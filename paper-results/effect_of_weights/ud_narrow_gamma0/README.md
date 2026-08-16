# Effect of weights: narrow $U_d$, $\gamma=0$ (App. D.4)

Poi(25) DGP with $U_{1:d-1}\sim\mathrm{Unif}(1,5)$, $U_d\sim\mathrm{Unif}(1,2)$, $B_j=0$.
Restricted GHCP ($\eta=0.5$) over $\lambda_{\mathrm{local}}\in\{1/7,3/7,4/7,6/7\}$ for RF and Bayes $E[Y\mid X,U]$, $o\in\{0,5,10,15,20\}$, target index 35.

2×2 panels (coverage with SE; twin width boxplots). Ticks are $o$; xlabel is “Test group size, $o$”.

```bash
.venv/bin/python code/marginal/run_effect_of_weights_ud_narrow.py --B 1000 --n_workers 6
.venv/bin/python code/marginal/plot_effect_of_weights_ud_narrow.py
```
