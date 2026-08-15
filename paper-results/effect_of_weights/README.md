# Effect of GHCP merger weights (App. D.4)

Paper Poisson design of Sec. 3.1 unless noted. Restricted GHCP \(\eta=0.5\),
\(\alpha=0.1\), \(K=20\), \(B=1000\), \(o\in\{0,5,10,15,20\}\). Target index 35
(same Poisson protocol). All weights in a replicate share the same draw and
donor/quantile seeds.

Merger (paper (3)):
\(\widetilde\mu=(1-\lambda_{\mathrm{local}})\widehat\mu^{\mathrm{global}}+\lambda_{\mathrm{local}}\overline Y\),
where \(\overline Y\) uses the first \(\tau=\lfloor o/2\rfloor\) outcomes. The
code evaluates a **fixed** \(\lambda_{\mathrm{local}}\) grid instead of the
default \(\tau/(|S_{\mathrm{train}}|+\tau)\). At \(o=0\), \(\tau=0\) so all
\(\lambda_{\mathrm{local}}\) coincide (pure global).

Paper appendix uses:

| Folder | Setting |
|--------|---------|
| [`gamma5p0/`](gamma5p0/) | Original DGP \(\gamma=5\), \(U\sim\mathrm{Unif}([1,5]^5)\), RF, \(\lambda_{\mathrm{local}}\in\{k/7\}_{k=1}^{6}\) (2×3 figures) |
| [`ud_narrow_gamma0/`](ud_narrow_gamma0/) | \(\gamma=0\), \(U_{1:d-1}\sim\mathrm{Unif}(1,5)\), \(U_d\sim\mathrm{Unif}(1,2)\); RF **and** Bayes \(E[Y\mid X,U]\); \(\lambda_{\mathrm{local}}\in\{1/7,3/7,4/7,6/7\}\) (2×2) |

Extra (not in App. D.4 text): [`gamma0p2/`](gamma0p2/) (\(\gamma=1/5\)), [`gamma0p0/`](gamma0p0/) (\(\gamma=0\), original \(U\) range).

```bash
.venv/bin/python code/marginal/run_effect_of_weights.py --B 1000 --n_workers 6 --gamma 5
.venv/bin/python code/marginal/plot_effect_of_weights.py --gamma 5

.venv/bin/python code/marginal/run_effect_of_weights_ud_narrow.py --B 1000 --n_workers 6
.venv/bin/python code/marginal/plot_effect_of_weights_ud_narrow.py
```

Raw CSVs: `../results/dgp/effect_of_weights_*`.
