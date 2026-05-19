"""SVTP with natural-gradient learning via the beta-link Fisher information.

Reference:
  Xu, J., Zeng, D. and Paisley, J. "Sparse Variational Student-t Processes for
  Heavy-Tailed Modeling." IEEE Transactions on Neural Networks and Learning
  Systems (2026), pp. 1-14, doi:10.1109/TNNLS.2026.3673350. (Journal
  extension of the AAAI 2024 SVTP paper.)

Key new ingredient: closed-form Fisher information matrix for the multivariate
Student-t variational posterior, expressed via beta functions ("beta link").
Using a diagonal covariance approximation S = diag(σ_1², ..., σ_M²), the
diagonal entries of F^m are (Eq. 36 of the paper):

   F^m_ii = (1 / (M σ_i²)) · (ν̃+M)/(ν̃-2) · B((M+3)/2, (ν̃+1)/2) / B(M/2, ν̃/2)

We use this to precondition the gradient on m (the dominant variational
parameter block) and keep Adam for kernel / inducing-location / σ_i / ν̃.

We compare SVTP-NatGrad against SVTP-MC (plain Adam on all parameters) on the
same UCI benchmark — same datasets and outlier injection — and report MSE vs
gradient step to demonstrate the convergence-speed advantage claimed in the
paper.
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
def load_uci(name, outlier_frac=0.0, outlier_scale=5.0, seed=0):
    """X standardized, y standardized; ±outlier_scale·σ outliers injected on training y."""
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
    mu_y, sd_y = ytr.mean(), ytr.std() + 1e-8
    Xtr = (Xtr - mu_x) / sd_x
    Xte = (Xte - mu_x) / sd_x
    ytr = (ytr - mu_y) / sd_y
    yte = (yte - mu_y) / sd_y
    n_out = int(outlier_frac * len(ytr))
    if n_out > 0:
        idx = rng.choice(len(ytr), n_out, replace=False)
        ytr = ytr.copy()
        ytr[idx] += rng.choice([-1, 1], size=n_out).astype(np.float32) * outlier_scale
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xte), torch.tensor(yte))


# ----------------------------------------------------------------- RBF kernel
def rbf(X1, X2, log_ls, log_var):
    ls = torch.exp(log_ls)
    var = torch.exp(log_var)
    X1s, X2s = X1 / ls, X2 / ls
    d2 = (X1s ** 2).sum(-1, keepdim=True) + (X2s ** 2).sum(-1).unsqueeze(0) - 2 * X1s @ X2s.T
    return var * torch.exp(-0.5 * d2.clamp(min=0))


# ----------------------------------------------------- SVTP with diagonal S
class SVTP_Diag(nn.Module):
    """Shared backbone for both SVTP-MC (Adam) and SVTP-NatGrad.

    Diagonal variational covariance S = diag(exp(2·log_sigma)).
    Fixed ν, ν̃ for optimization stability.
    """

    def __init__(self, D, M, nu_init=5.0, nu_q_init=5.0, noise_init=0.1, n_mc=8):
        super().__init__()
        self.D, self.M, self.n_mc = D, M, n_mc
        self.Z = nn.Parameter(torch.randn(M, D) * 0.5)
        self.log_ls = nn.Parameter(torch.zeros(D))
        self.log_var = nn.Parameter(torch.zeros(1))
        self.m = nn.Parameter(torch.zeros(M))
        self.log_sigma = nn.Parameter(torch.zeros(M))   # diagonal S^{1/2}
        self.log_noise = nn.Parameter(torch.log(torch.tensor(noise_init)), requires_grad=False)
        self.log_nu_offset = nn.Parameter(torch.log(torch.tensor(nu_init - 2.1)), requires_grad=False)
        self.log_nu_q_offset = nn.Parameter(torch.log(torch.tensor(nu_q_init - 2.1)), requires_grad=False)

    @property
    def nu(self): return torch.exp(self.log_nu_offset) + 2.1
    @property
    def nu_q(self): return torch.exp(self.log_nu_q_offset) + 2.1
    @property
    def sigma(self): return torch.exp(self.log_sigma)
    @property
    def sigma_sq(self): return torch.exp(2 * self.log_sigma)

    def _kernels(self, X):
        jitter = 1e-3
        Kuu = rbf(self.Z, self.Z, self.log_ls, self.log_var) + jitter * torch.eye(self.M, device=X.device)
        Kuf = rbf(self.Z, X, self.log_ls, self.log_var)
        Kff_diag = torch.exp(self.log_var) * torch.ones(X.shape[0], device=X.device)
        return Kuu, Kuf, Kff_diag

    def sample_q_u(self, n_samples):
        """u_i = m + √((ν̃-2)/r_i) · σ ⊙ ε_i,  r_i ~ Gamma(ν̃/2, 1/2)."""
        nu_q = self.nu_q
        r = torch.distributions.Gamma(nu_q / 2, 0.5).rsample((n_samples,))
        eps = torch.randn(n_samples, self.M, device=self.m.device)
        scale = ((nu_q - 2) / r).sqrt().unsqueeze(-1)
        return self.m + scale * (self.sigma.unsqueeze(0) * eps)

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

        # KL term (diagonal S): const − (ν̃+M)/2 L_1 + (ν+M)/2 L_2_mc
        L_S_diag = self.sigma           # √S
        nu, nu_q, M = self.nu, self.nu_q, self.M
        log_det_K = 2 * torch.log(torch.diagonal(Luu)).sum()
        log_det_S = 2 * torch.log(L_S_diag.abs()).sum()
        const = (
            torch.lgamma((nu_q + M) / 2) - torch.lgamma(nu_q / 2)
            - torch.lgamma((nu + M) / 2) + torch.lgamma(nu / 2)
            + (M / 2) * (torch.log(nu - 2) - torch.log(nu_q - 2))
            + 0.5 * (log_det_K - log_det_S)
        )
        L_1 = torch.digamma((nu_q + M) / 2) - torch.digamma(nu_q / 2)
        # MC: sample u and compute uᵀK⁻¹u
        u_mc = self.sample_q_u(self.n_mc)               # (n_mc, M)
        Linv_u_mc = torch.linalg.solve_triangular(Luu, u_mc.T, upper=False)
        quad_mc = (Linv_u_mc ** 2).sum(0)               # (n_mc,)
        L_2_mc = torch.log1p(quad_mc / (nu - 2)).mean()
        kl = const - (nu_q + M) / 2 * L_1 + (nu + M) / 2 * L_2_mc
        return log_lik - kl

    def predict_mean(self, X):
        Kuu, Kuf, _ = self._kernels(X)
        Luu = torch.linalg.cholesky(Kuu)
        Linv_Kuf = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
        Linv_m = torch.linalg.solve_triangular(Luu, self.m.unsqueeze(-1), upper=False).squeeze(-1)
        return Linv_Kuf.T @ Linv_m


# -------------------------------- Fisher information for natural gradient
def log_beta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def fisher_m_diag_inv(M, nu_q, sigma_sq):
    """1 / F^m_ii  from paper Eq. 36 (diagonal S)."""
    Mf = torch.tensor(float(M), dtype=sigma_sq.dtype, device=sigma_sq.device)
    log_F = (-torch.log(Mf) - torch.log(sigma_sq)
             + torch.log(nu_q + Mf) - torch.log(nu_q - 2)
             + log_beta((Mf + 3) / 2, (nu_q + 1) / 2)
             - log_beta(Mf / 2, nu_q / 2))
    return torch.exp(-log_F)


def fisher_logS_inv_coefs(M, nu_q):
    """Inverse of Fisher in log-sigma parametrisation, paper Eq. 39+40.

    For S=diag(σ²), the Fisher F^S(σ) has c_diag/σ_i² on the diagonal and
    c_off/(σ_i σ_j) off-diagonal. Under log_sigma parametrisation
    F^{ls}_ij = σ_i σ_j · F^S_ij so the σ factors drop out:

        F^{ls} = (c_diag − c_off) · I  +  c_off · 1·1ᵀ

    Sherman–Morrison gives (αI + β J)^{-1} = (1/α) I − β/(α(α+βM)) J.
    Returns (alpha_inv, mix_coef) so the natural gradient for log_sigma is
        nat_g = alpha_inv · g  +  mix_coef · sum(g) · 1
    """
    Mf = torch.tensor(float(M), dtype=nu_q.dtype, device=nu_q.device)
    R1 = torch.exp(log_beta((Mf + 3) / 2, (nu_q - 1) / 2) - log_beta(Mf / 2, nu_q / 2))
    R2 = torch.exp(log_beta((Mf + 5) / 2, (nu_q - 1) / 2) - log_beta(Mf / 2, nu_q / 2))
    a = 2 * (nu_q + Mf) / (Mf + 2) * R1
    b = (nu_q + Mf) ** 2 / ((Mf + 4) * (Mf + 2)) * R2
    c_diag = 1.0 - a + 5.0 * b
    c_off  = 1.0 - a +       b
    alpha = c_diag - c_off       # = 4·b
    apbm  = c_diag + c_off * (Mf - 1)
    alpha_inv = 1.0 / alpha
    mix_coef  = -c_off / (alpha * apbm)
    return alpha_inv, mix_coef


# ---------------------------------------------------------------------- train
def train_adam(model, X_tr, y_tr, X_te, y_te, n_iter=3000, batch_size=256, lr=0.01,
                eval_every=100, log_every=500):
    n = X_tr.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = min(batch_size, n)
    history = []
    for it in range(n_iter):
        idx = torch.randperm(n, device=X_tr.device)[:bs]
        opt.zero_grad()
        loss = -model.elbo(X_tr[idx], y_tr[idx], n / bs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (it + 1) % eval_every == 0 or it == 0:
            with torch.no_grad():
                mse = ((model.predict_mean(X_te) - y_te) ** 2).mean().item()
            history.append((it + 1, loss.item(), mse))
            if (it + 1) % log_every == 0 or it == 0:
                print(f"      iter {it+1:5d}  loss {loss.item():.3f}  test_MSE {mse:.4f}", flush=True)
    return history


def train_natgrad(model, X_tr, y_tr, X_te, y_te, n_iter=3000, batch_size=256,
                   lr_nat=0.01, lr_adam=0.01, eval_every=100, log_every=500):
    """Hybrid optimisation: natural gradient on m, Adam on the rest.

    Empirically the "partial" recipe — NatGrad only on the variational mean m
    via the diagonal F^m (paper Eq. 36) — gives the most reliable speedup. A
    full version that also applies the Sherman–Morrison inverse of F^{logS}
    (Eqs. 39+40) to log_sigma is included in `fisher_logS_inv_coefs` for
    reference, but in our experiments it under-performs Adam on log_sigma.
    """
    n = X_tr.shape[0]
    bs = min(batch_size, n)
    adam_params = [p for p in (model.Z, model.log_ls, model.log_var, model.log_sigma)
                    if p.requires_grad]
    opt_adam = torch.optim.Adam(adam_params, lr=lr_adam)

    history = []
    for it in range(n_iter):
        idx = torch.randperm(n, device=X_tr.device)[:bs]
        opt_adam.zero_grad()
        model.m.grad = None
        loss = -model.elbo(X_tr[idx], y_tr[idx], n / bs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adam_params, 5.0)
        opt_adam.step()

        with torch.no_grad():
            F_inv_m = fisher_m_diag_inv(model.M, model.nu_q, model.sigma_sq.detach())
            nat_g_m = F_inv_m * model.m.grad
            nat_g_m = torch.clamp(nat_g_m, -5.0, 5.0)
            model.m.data -= lr_nat * nat_g_m

        if (it + 1) % eval_every == 0 or it == 0:
            with torch.no_grad():
                mse = ((model.predict_mean(X_te) - y_te) ** 2).mean().item()
            history.append((it + 1, loss.item(), mse))
            if (it + 1) % log_every == 0 or it == 0:
                print(f"      iter {it+1:5d}  loss {loss.item():.3f}  test_MSE {mse:.4f}", flush=True)
    return history


# -------------------------------------------------------------- experiment
def run_one(name, X_tr, y_tr, X_te, y_te, M, n_iter, device, seed):
    D = X_tr.shape[1]
    out = {}
    X_tr_d, y_tr_d = X_tr.to(device), y_tr.to(device)
    X_te_d, y_te_d = X_te.to(device), y_te.to(device)

    print(f"    [SVTP-MC (Adam)]", flush=True)
    torch.manual_seed(seed)
    model = SVTP_Diag(D, M).to(device)
    hist_adam = train_adam(model, X_tr_d, y_tr_d, X_te_d, y_te_d, n_iter=n_iter)
    out["SVTP-MC"] = (hist_adam[-1][2], hist_adam)

    print(f"    [SVTP-NatGrad (NatGrad on m, Adam on rest)]", flush=True)
    torch.manual_seed(seed)
    model = SVTP_Diag(D, M).to(device)
    hist_nat = train_natgrad(model, X_tr_d, y_tr_d, X_te_d, y_te_d, n_iter=n_iter)
    out["SVTP-NatGrad"] = (hist_nat[-1][2], hist_nat)
    return out


def find_iters_to_target(history, target_mse):
    for it, _, mse in history:
        if mse <= target_mse:
            return it
    return None


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n", flush=True)
    results = {}
    seeds = [0, 1, 2]
    for ds in ["yacht", "energy", "boston"]:
        for outlier_frac, scale in [(0.0, 0.0), (0.05, 5.0)]:
            tag = "clean" if outlier_frac == 0 else "outlier5%(±5σ)"
            key = f"{ds:7s} / {tag}"
            results[key] = []
            for seed in seeds:
                X_tr, y_tr, X_te, y_te = load_uci(
                    ds, outlier_frac=outlier_frac, outlier_scale=scale, seed=seed,
                )
                M = max(32, len(X_tr) // 4)
                print(f"\n[{key}] seed={seed}  n_tr={len(X_tr)}  D={X_tr.shape[1]}  M={M}", flush=True)
                results[key].append(run_one(key, X_tr, y_tr, X_te, y_te,
                                            M=M, n_iter=args.n_iter, device=device, seed=seed))

    print("\n\n========== FINAL MSE (mean ± std over 3 seeds) ==========", flush=True)
    print(f"{'Dataset':25s}  {'SVTP-MC':>20s}  {'SVTP-NatGrad':>20s}  {'NatGrad/MC':>10s}")
    for name, rs in results.items():
        mc = np.array([r["SVTP-MC"][0] for r in rs])
        ng = np.array([r["SVTP-NatGrad"][0] for r in rs])
        ratio = ng.mean() / mc.mean()
        print(f"{name:25s}  {mc.mean():7.4f} ± {mc.std():.4f}    "
              f"{ng.mean():7.4f} ± {ng.std():.4f}    {ratio:10.3f}")

    # Convergence speed: iterations to reach 1.5× of the better final MSE
    print("\n========== CONVERGENCE (iters to reach 1.5× best final MSE) ==========", flush=True)
    print(f"{'Dataset':25s}  {'best target':>12s}  {'SVTP-MC':>10s}  {'SVTP-NatGrad':>13s}  {'speedup':>8s}")
    for name, rs in results.items():
        targets, iters_mc, iters_ng = [], [], []
        for r in rs:
            best = min(r["SVTP-MC"][0], r["SVTP-NatGrad"][0])
            target = best * 1.5
            iter_mc = find_iters_to_target(r["SVTP-MC"][1], target)
            iter_ng = find_iters_to_target(r["SVTP-NatGrad"][1], target)
            targets.append(target)
            iters_mc.append(iter_mc if iter_mc else args.n_iter)
            iters_ng.append(iter_ng if iter_ng else args.n_iter)
        speedup = np.mean(iters_mc) / np.mean(iters_ng) if np.mean(iters_ng) > 0 else float("nan")
        print(f"{name:25s}  {np.mean(targets):12.4f}  {np.mean(iters_mc):10.0f}  "
              f"{np.mean(iters_ng):13.0f}  {speedup:8.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_iter", type=int, default=3000)
    main(p.parse_args())
