# Sparse Variational Student-t Processes (SVTP)

[![AAAI 2024](https://img.shields.io/badge/AAAI-2024-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/29547)
[![DOI](https://img.shields.io/badge/DOI-10.1609%2Faaai.v38i14.29547-orange)](https://doi.org/10.1609/aaai.v38i14.29547)

Official implementation of the **AAAI 2024** paper *"Sparse Variational
Student-t Processes"* (SVTP).

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
└── svtp.py        Single-file implementation: SVGP baseline + SVTP-MC + UCI bench
```

The script self-contains the kernel, the variational posteriors for both
models, and the full benchmark across Yacht / Energy / Boston with optional
outlier injection.

---

## Requirements

```bash
pip install torch numpy scikit-learn
```

(Optional) GPU is auto-detected; the script also runs on CPU.

---

## Quick start

```bash
python svtp.py
```

This runs all 3 UCI datasets × {clean, outlier 5%(±5σ)} × {SVGP, SVTP-MC} × 3
seeds and prints a summary table at the end.

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
```

---

## Contact

Questions / issues: open a GitHub issue or contact the corresponding author
(see paper).
