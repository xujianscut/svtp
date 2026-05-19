"""Sparse Variational Student-t Processes (SVTP) — AAAI 2024.

Reference:
  Xu, J. and Zeng, D. "Sparse Variational Student-t Processes."
  Proceedings of the AAAI Conference on Artificial Intelligence, 38(14),
  16156–16163, 2024.

Implements the SVTP-MC formulation with proper Student-t variational posterior
and an MC-estimated KL regulariser:

  q(u) = ST(ν̃, m, S),   p(u) = ST(ν, 0, K_ZZ)

  reparameterised sampling   u = m + √((ν̃-2)/r) · L_S ε
                              r ~ Gamma(ν̃/2, 1/2),  ε ~ N(0, I)

  KL(q || p) = C(ν, ν̃, S)  −  (ν̃+M)/2 · L_1  +  (ν+M)/2 · L_2_mc
       L_1    = ψ((ν̃+M)/2) − ψ(ν̃/2)               (closed form, Lemma 5)
       L_2_mc = E_q[log(1 + uᵀK⁻¹u / (ν-2))]        (MC estimate)

  per-point Student-t likelihood
       p(y_i | u) ~ ST(ν+M, μ_i, Σ_ii + σ²)
       μ_i = K_xi,Z K⁻¹ u   and   Σ_ii = (ν+β-2)/(ν+M-2) · (K_xixi − …)

Compared against a standard SVGP baseline on UCI Yacht / Energy / Boston,
clean and with 5% ±5σ outliers injected.
"""
import argparse
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------- data
def load_uci(name, outlier_frac=0.0, outlier_scale=3.0, seed=0, standardize_y=True):
    """Paper setup: X standardized, y kept on raw scale; +3σ outliers as in paper."""
    if name == "boston":
        d = fetch_openml("boston", version=1, as_frame=False)
    elif name == "yacht":
        d = fetch_openml("yacht_hydrodynamics", version=1, as_frame=False)
    elif name == "energy":
        d = fetch_openml("energy_efficiency", version=1, as_frame=False)
    else:
        raise ValueError(name)
    X = np.asarray(d.data, dtype=np.float32)
    y = np.asarray(d.target, dtype=np.float32)
    if y.ndim > 1:
        y = y[:, 0]
    rng = np.random.RandomState(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu_x) / sd_x
    Xte = (Xte - mu_x) / sd_x
    if standardize_y:
        mu_y, sd_y = ytr.mean(), ytr.std() + 1e-8
        ytr = (ytr - mu_y) / sd_y
        yte = (yte - mu_y) / sd_y
    sd_y_train = float(ytr.std() + 1e-8)
    n_out = int(outlier_frac * len(ytr))
    if n_out > 0:
        idx = rng.choice(len(ytr), n_out, replace=False)
        ytr = ytr.copy()
        ytr[idx] += rng.choice([-1, 1], size=n_out).astype(np.float32) * outlier_scale * sd_y_train
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xte), torch.tensor(yte))


# ----------------------------------------------------------------- RBF kernel
def rbf(X1, X2, log_ls, log_var):
    ls = torch.exp(log_ls)
    var = torch.exp(log_var)
    X1s, X2s = X1 / ls, X2 / ls
    d2 = (X1s ** 2).sum(-1, keepdim=True) + (X2s ** 2).sum(-1).unsqueeze(0) - 2 * X1s @ X2s.T
    return var * torch.exp(-0.5 * d2.clamp(min=0))


# ----------------------------------------------------------------------- SVGP
class SVGP(nn.Module):
    def __init__(self, D, M, noise_init=0.1):
        super().__init__()
        self.D, self.M = D, M
        self.Z = nn.Parameter(torch.randn(M, D) * 0.5)
        self.log_ls = nn.Parameter(torch.zeros(D))
        self.log_var = nn.Parameter(torch.zeros(1))
        self.m = nn.Parameter(torch.zeros(M))
        self.L_raw = nn.Parameter(torch.eye(M) * 0.1)
        self.log_noise = nn.Parameter(torch.log(torch.tensor(noise_init)), requires_grad=False)

    def _kernels(self, X):
        jitter = 1e-3
        Kuu = rbf(self.Z, self.Z, self.log_ls, self.log_var) + jitter * torch.eye(self.M, device=X.device)
        Kuf = rbf(self.Z, X, self.log_ls, self.log_var)
        Kff_diag = torch.exp(self.log_var) * torch.ones(X.shape[0], device=X.device)
        return Kuu, Kuf, Kff_diag

    def get_L(self):
        return torch.tril(self.L_raw) + 1e-6 * torch.eye(self.M, device=self.L_raw.device)

    def predict(self, X):
        Kuu, Kuf, Kff_diag = self._kernels(X)
        Luu = torch.linalg.cholesky(Kuu)
        Linv_Kuf = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
        Linv_m = torch.linalg.solve_triangular(Luu, self.m.unsqueeze(-1), upper=False).squeeze(-1)
        mu_f = Linv_Kuf.T @ Linv_m
        Kuuinv_Kuf = torch.linalg.solve_triangular(Luu.T, Linv_Kuf, upper=True)
        L = self.get_L()
        S = L @ L.T
        var_f = Kff_diag - (Linv_Kuf ** 2).sum(0) + (Kuuinv_Kuf * (S @ Kuuinv_Kuf)).sum(0)
        return mu_f, var_f.clamp(min=1e-8), Luu, L

    def elbo(self, X, y, scale):
        mu_f, var_f, Luu, L = self.predict(X)
        noise2 = torch.exp(self.log_noise) ** 2
        log_lik = -0.5 * ((y - mu_f) ** 2 + var_f) / noise2 - 0.5 * torch.log(2 * math.pi * noise2)
        log_lik = log_lik.sum() * scale
        Linv_L = torch.linalg.solve_triangular(Luu, L, upper=False)
        Linv_m = torch.linalg.solve_triangular(Luu, self.m.unsqueeze(-1), upper=False).squeeze(-1)
        kl = 0.5 * ((Linv_L ** 2).sum() + (Linv_m ** 2).sum() - self.M
                    + 2 * torch.log(torch.diagonal(Luu)).sum()
                    - 2 * torch.log(torch.abs(torch.diagonal(L))).sum())
        return log_lik - kl

    def predict_mean(self, X):
        return self.predict(X)[0]


# -------------------------------------------------------------------- SVTP-MC
class SVTP_MC(nn.Module):
    """Full SVTP-MC: Student-t q(u), Student-t p(u), MC KL, Student-t likelihood."""

    def __init__(self, D, M, nu_init=5.0, nu_q_init=5.0, noise_init=0.1, n_mc=8):
        super().__init__()
        self.D, self.M, self.n_mc = D, M, n_mc
        self.Z = nn.Parameter(torch.randn(M, D) * 0.5)
        self.log_ls = nn.Parameter(torch.zeros(D))
        self.log_var = nn.Parameter(torch.zeros(1))
        self.m = nn.Parameter(torch.zeros(M))
        self.L_raw = nn.Parameter(torch.eye(M) * 0.1)
        self.log_noise = nn.Parameter(torch.log(torch.tensor(noise_init)), requires_grad=False)
        # Fix nu and nu_q (paper-style): learnable nu is unstable with MC KL
        self.log_nu_offset = nn.Parameter(torch.log(torch.tensor(nu_init - 2.1)), requires_grad=False)
        self.log_nu_q_offset = nn.Parameter(torch.log(torch.tensor(nu_q_init - 2.1)), requires_grad=False)

    @property
    def nu(self):
        return torch.exp(self.log_nu_offset) + 2.1

    @property
    def nu_q(self):
        return torch.exp(self.log_nu_q_offset) + 2.1

    def get_L(self):
        return torch.tril(self.L_raw) + 1e-6 * torch.eye(self.M, device=self.L_raw.device)

    def _kernels(self, X):
        jitter = 1e-3
        Kuu = rbf(self.Z, self.Z, self.log_ls, self.log_var) + jitter * torch.eye(self.M, device=X.device)
        Kuf = rbf(self.Z, X, self.log_ls, self.log_var)
        Kff_diag = torch.exp(self.log_var) * torch.ones(X.shape[0], device=X.device)
        return Kuu, Kuf, Kff_diag

    def sample_q_u(self, n_samples):
        """u_i = m + √((ν̃-2)/r_i) · L_S ε_i,  r_i ~ Gamma(ν̃/2, 1/2)."""
        L_S = self.get_L()
        nu_q = self.nu_q
        r = torch.distributions.Gamma(nu_q / 2, 0.5).rsample((n_samples,))
        eps = torch.randn(n_samples, self.M, device=self.m.device)
        scale = ((nu_q - 2) / r).sqrt().unsqueeze(-1)   # (n_samples, 1)
        return self.m + scale * (eps @ L_S.T)            # (n_samples, M)

    def predict_given_u(self, X, u):
        Kuu, Kuf, Kff_diag = self._kernels(X)
        Luu = torch.linalg.cholesky(Kuu)
        Linv_Kuf = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
        Linv_u = torch.linalg.solve_triangular(Luu, u.unsqueeze(-1), upper=False).squeeze(-1)
        mu_f = Linv_Kuf.T @ Linv_u
        beta = (Linv_u ** 2).sum()
        sigma_diag = (Kff_diag - (Linv_Kuf ** 2).sum(0)).clamp(min=1e-8)
        scale_factor = (self.nu + beta - 2) / (self.nu + self.M - 2)
        return mu_f, scale_factor * sigma_diag, beta, Luu

    def elbo(self, X, y, scale):
        # === likelihood: 1 MC sample of u, per-point Student-t with df=ν+M ===
        u_lik = self.sample_q_u(1).squeeze(0)
        mu_f, sigma_f, beta, Luu = self.predict_given_u(X, u_lik)
        df = self.nu + self.M
        noise2 = torch.exp(self.log_noise) ** 2
        disp = sigma_f + noise2
        z2 = (y - mu_f) ** 2 / ((df - 2) * disp)
        log_lik = (
            torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2)
            - 0.5 * torch.log((df - 2) * math.pi * disp)
            - 0.5 * (df + 1) * torch.log1p(z2)
        )
        log_lik = log_lik.sum() * scale

        # === KL(q || p): const − (ν̃+M)/2 · L_1 + (ν+M)/2 · L_2_mc ===
        L_S = self.get_L()
        nu, nu_q, M = self.nu, self.nu_q, self.M
        log_det_K = 2 * torch.log(torch.diagonal(Luu)).sum()
        log_det_S = 2 * torch.log(torch.abs(torch.diagonal(L_S))).sum()
        const = (
            torch.lgamma((nu_q + M) / 2) - torch.lgamma(nu_q / 2)
            - torch.lgamma((nu + M) / 2) + torch.lgamma(nu / 2)
            + (M / 2) * (torch.log(nu - 2) - torch.log(nu_q - 2))
            + 0.5 * (log_det_K - log_det_S)
        )
        L_1 = torch.digamma((nu_q + M) / 2) - torch.digamma(nu_q / 2)
        u_mc = self.sample_q_u(self.n_mc)                                   # (n_mc, M)
        Linv_u_mc = torch.linalg.solve_triangular(Luu, u_mc.T, upper=False)  # (M, n_mc)
        quad_mc = (Linv_u_mc ** 2).sum(0)                                    # (n_mc,)
        L_2_mc = torch.log1p(quad_mc / (nu - 2)).mean()
        kl = const - (nu_q + M) / 2 * L_1 + (nu + M) / 2 * L_2_mc
        return log_lik - kl

    def predict_mean(self, X):
        """Deterministic predictive mean uses variational mean m directly."""
        Kuu, Kuf, _ = self._kernels(X)
        Luu = torch.linalg.cholesky(Kuu)
        Linv_Kuf = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
        Linv_m = torch.linalg.solve_triangular(Luu, self.m.unsqueeze(-1), upper=False).squeeze(-1)
        return Linv_Kuf.T @ Linv_m


# ---------------------------------------------------------------------- train
def train_model(model, X_tr, y_tr, n_iter=2000, batch_size=256, lr=0.01, log_every=500):
    n = X_tr.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = min(batch_size, n)
    for it in range(n_iter):
        idx = torch.randperm(n, device=X_tr.device)[:bs]
        opt.zero_grad()
        loss = -model.elbo(X_tr[idx], y_tr[idx], n / bs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (it + 1) % log_every == 0:
            print(f"      iter {it+1:5d}  loss {loss.item():.3f}", flush=True)


def eval_model(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        mu = model.predict_mean(X_te)
        mse = ((mu - y_te) ** 2).mean().item()
    return mse


def run_one(name, X_tr, y_tr, X_te, y_te, M, n_iter, device, seed):
    D = X_tr.shape[1]
    out = {}
    for tag, ModelCls in [("SVGP", SVGP), ("SVTP-MC", SVTP_MC)]:
        torch.manual_seed(seed)
        model = ModelCls(D, M).to(device)
        train_model(model, X_tr.to(device), y_tr.to(device), n_iter=n_iter)
        mse = eval_model(model, X_te.to(device), y_te.to(device))
        extra = f"  (nu={model.nu.item():.2f}, nu_q={model.nu_q.item():.2f})" if isinstance(model, SVTP_MC) else ""
        print(f"    {tag:8s}  MSE {mse:.4f}{extra}", flush=True)
        out[tag] = mse
    return out


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n", flush=True)
    seeds = [0, 1, 2]
    results = {}
    for ds in ["yacht", "energy", "boston"]:
        for outlier_frac, outlier_scale in [(0.0, 0.0), (0.05, 5.0)]:
            tag = "clean" if outlier_frac == 0 else "outlier5%(±5σ)"
            key = f"{ds:7s} / {tag}"
            results[key] = []
            for seed in seeds:
                X_tr, y_tr, X_te, y_te = load_uci(
                    ds, outlier_frac=outlier_frac,
                    outlier_scale=outlier_scale, seed=seed,
                )
                M = max(32, len(X_tr) // 4)
                print(f"\n[{key}] seed={seed}  n_tr={len(X_tr)}  D={X_tr.shape[1]}  M={M}", flush=True)
                results[key].append(run_one(key, X_tr, y_tr, X_te, y_te,
                                            M=M, n_iter=args.n_iter, device=device, seed=seed))

    print("\n\n========== SUMMARY (mean ± std over 3 seeds) ==========", flush=True)
    print(f"{'Dataset':22s}  {'SVGP MSE':>16s}  {'SVTP-MC MSE':>16s}  {'SVTP/SVGP':>10s}")
    for name, rs in results.items():
        svgp = np.array([r["SVGP"] for r in rs])
        svtp = np.array([r["SVTP-MC"] for r in rs])
        ratio = svtp.mean() / svgp.mean()
        print(f"{name:22s}  {svgp.mean():7.4f} ± {svgp.std():.4f}    "
              f"{svtp.mean():7.4f} ± {svtp.std():.4f}    {ratio:10.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_iter", type=int, default=3000)
    main(p.parse_args())
