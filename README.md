# Sparse Variational Student-t Processes (SVTP)

[![AAAI 2024](https://img.shields.io/badge/AAAI-2024-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29547)
[![TNNLS 2026](https://img.shields.io/badge/IEEE%20TNNLS-2026-blue)](https://doi.org/10.1109/TNNLS.2026.3673350)
[![DOI](https://img.shields.io/badge/DOI-10.1609%2Faaai.v38i14.29547-orange)](https://doi.org/10.1609/aaai.v38i14.29547)

Official implementation of:

- **AAAI 2024**: *"Sparse Variational Student-t Processes"* (SVTP)
- **IEEE TNNLS 2026**: *"Sparse Variational Student-t Processes for
  Heavy-Tailed Modeling"* — journal extension with closed-form Fisher
  information ("beta link") for natural-gradient optimisation.

SVTP extends the sparse-inducing-point framework of SVGP to Student-t
processes, giving heavier tails in both the prior and the likelihood. This
makes the model substantially more robust to outliers and heavy-tailed
observation noise while preserving SVGP's `O(nm² + m³)` complexity.

---

## Method overview

For inducing-variable Student-t inference with $\mathbf U=\{u_i\}_{i=1}^M$:

- **Student-t prior**     $p(\mathbf U) = \mathcal{ST}(\nu,\,\mathbf 0,\,K_{ZZ})$
- **Student-t posterior** $q(\mathbf U) = \mathcal{ST}(\tilde\nu,\,\mathbf m,\,\mathbf S)$
- **Reparameterised sampling**
  $\mathbf u = \mathbf m + \sqrt{(\tilde\nu-2)/r}\,L_S\,\boldsymbol\varepsilon$,
  with $r\sim\mathrm{Gamma}(\tilde\nu/2, 1/2)$, $\boldsymbol\varepsilon\sim\mathcal N(\mathbf 0, I)$.

ELBO terms:

$$
\mathcal L = \mathbb E_{q}[\log p(\mathbf y\mid \mathbf u)] - \mathrm{KL}(q\Vert p)
$$

$$
\mathrm{KL}(q\Vert p) = C(\nu,\tilde\nu,\mathbf S) - \tfrac{\tilde\nu+M}{2}\,L_1 + \tfrac{\nu+M}{2}\,\widehat{L_2}_{\text{MC}}
$$

with $L_1 = \psi(\tfrac{\tilde\nu+M}{2}) - \psi(\tfrac{\tilde\nu}{2})$ closed-form
(Lemma 5), and $\widehat{L_2}_{\text{MC}}$ a Monte-Carlo estimate of
$\mathbb E_q[\log(1 + \mathbf u^\top K_{ZZ}^{-1}\mathbf u / (\nu-2))]$.

The likelihood is per-point Student-t with degrees of freedom $\nu+M$ — the
key term producing outlier robustness via the **log compression** of the
squared error (paper Eq. 25):

$$
\log p(y_i\mid \mathbf u) \propto -\tfrac{\nu+M+1}{2}\,\log\!\Big(1 + \tfrac{(y_i-\mu_i)^2}{(\nu+M-2)\,\Sigma_{ii}}\Big).
$$

---

## Repository layout

```
.
├── svtp.py            SVGP baseline + SVTP-MC + UCI benchmark (AAAI 2024)
└── svtp_natgrad.py    SVTP with natural-gradient learning (TNNLS 2026)
```

`svtp.py` self-contains the kernel, the variational posteriors for SVGP and
SVTP-MC, and the full benchmark across Yacht / Energy / Boston with optional
outlier injection.

`svtp_natgrad.py` reuses the same backbone with a *diagonal* variational
covariance $S = \mathrm{diag}(\sigma_1^2, \dots, \sigma_M^2)$ and implements
the closed-form Fisher information matrix from the TNNLS extension. The
variational mean $\mathbf m$ is updated by

$$
\mathbf m \leftarrow \mathbf m - \eta_{\mathrm{nat}} \cdot (F^{\mathbf m})^{-1}\,\nabla_{\mathbf m}\mathcal L
$$

with $F^{\mathbf m}_{ii} = \frac{1}{M\sigma_i^2}\cdot\frac{\tilde\nu+M}{\tilde\nu-2}\cdot\frac{B((M{+}3)/2,(\tilde\nu{+}1)/2)}{B(M/2,\tilde\nu/2)}$ from Eq. 36
of the paper, and the remaining parameters stay on Adam. The Sherman–Morrison
form of $F^{\log S}$ (Eqs. 39 + 40) is also provided as
`fisher_logS_inv_coefs` for reference, but in our reproduction it
under-performs Adam on `log_sigma`, so the shipped recipe is **NatGrad on
$\mathbf m$ + Adam on the rest**.

---

## Requirements

```bash
pip install torch numpy scikit-learn
```

(Optional) GPU is auto-detected; the script also runs on CPU.

---

## Quick start

```bash
# AAAI 2024 SVTP-MC vs SVGP baseline
python svtp.py

# TNNLS 2026 natural-gradient variant vs SVTP-MC (Adam)
python svtp_natgrad.py
```

Each script runs all 3 UCI datasets × {clean, outlier 5%(±5σ)} × 3 seeds and
prints a summary table at the end.

To shorten a smoke run:

```bash
python svtp.py --n_iter 500
```

---

## Key implementation details

| Choice | Default | Rationale |
|---|---|---|
| Inducing points $M$ | $\max(32, n/4)$ | follows the paper's "$m = n/4$" rule |
| Batch size | 256 | minibatch SGD on small UCI sets |
| Optimizer | Adam, lr 0.01 | paper hyperparameter |
| Iterations | 3000 | enough to converge on these datasets |
| Jitter | 1e-3 | larger than 1e-4 to avoid Cholesky failure under outliers |
| Observation noise $\sigma$ | 0.10, **fixed** | stops SVGP from absorbing outliers via inflated noise |
| Student-t df $\nu, \tilde\nu$ | 5.0, **fixed** | learnable $\nu$ + MC KL is unstable; paper-style |
| MC samples for $L_2$ | 8 | unbiased; SGD noise dominates |
| Outlier injection | 5% of training $y$ shifted by $\pm 5\sigma_y$ | matches the spirit of "Concrete\_Outliers / Kin8nm\_Outliers" in the paper |

---

## Citation

```bibtex
@inproceedings{xu2024sparse,
  title     = {Sparse variational student-t processes},
  author    = {Xu, Jian and Zeng, Delu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {38},
  number    = {14},
  pages     = {16156--16163},
  year      = {2024}
}

@ARTICLE{11441992,
  author  = {Xu, Jian and Zeng, Delu and Paisley, John},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  title   = {Sparse Variational Student-t Processes for Heavy-Tailed Modeling},
  year    = {2026},
  pages   = {1-14},
  doi     = {10.1109/TNNLS.2026.3673350}
}
```

---

## Contact

Questions / issues: open a GitHub issue or contact the corresponding author
(see paper).
