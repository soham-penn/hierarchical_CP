# Size-shift sensitivity (App. D.3)

Paper Poisson DGP of Sec. 3.1 except the latent intercept. Restricted GHCP
(\(\eta=0.5\)), \(\gamma=5\), \(\alpha=0.1\), \(K=20\), \(o\in\{0,5,10,15,20\}\).
RF 50 trees / min leaf 5. \(B=1000\). Destination: this folder +
`../results/dgp/size_shift_gamma5p0_poissonNmean25_alpha10/`.

Independently of \(U_j\), draw \(M_j\sim\mathrm{Poi}(25)\) and
\(\varepsilon_j\sim N(0,1)\). For the \(K\) reference groups set \(N_j=M_j\) and

\[
B_j=\gamma\Bigl(\sqrt{1-\xi^2}\,\varepsilon_j+\xi\frac{M_j-25}{5}\Bigr),\qquad \lvert\xi\rvert<1.
\]

This keeps \(\mathrm{Var}(B_j)=\gamma^2\) and
\(\operatorname{Corr}(B_j,N_j)=\xi\) on the reference groups. Observation model
unchanged. \(\xi=0\) is the original Poisson DGP.

**Test group.** \(M_{K+1}\) is used only to form \(B_{K+1}\) (Assumption A2 fails
through \(B\), not through a shared \(N_{\mathrm{test}}\)). The observed test
stream matches the Poisson GHCP protocol: length \(36\) (target **index 35**),
history = first \(o\) rows, \(\tau=\lfloor o/2\rfloor\). Same data
\((U,M,\varepsilon,X)\) is reused across
\(\xi\in\{-0.75,0,0.5,0.75\}\) (2×2 figures and `size_shift_table.tex`).

A fresh run writes exactly \(\xi\in\{-0.75,0,0.5,0.75\}\) (20{,}000 finite
intervals: 4 \(\xi\) × 5 \(o\) × 1000) and replaces any previous trial CSV.
Pass `--merge-existing-xi` only if you want to keep other \(\xi\) from an
earlier grid. Chunking matches the Poisson Sec. 3.1 runner: 40 chunks of 25,
`np.random.seed(457 + 1000*i)`. `--n_workers` is concurrency only.
The plotter also keeps only those four \(\xi\) if an older trial file is still on disk.

```bash
.venv/bin/python code/marginal/run_size_shift_sensitivity.py --B 1000 --n_workers 6
.venv/bin/python code/marginal/plot_size_shift_sensitivity.py
```

Figures: `coverage_by_xi.pdf`, `width_by_xi.pdf`. Numbers:
`summary_by_xi_o.csv` (same four \(\xi\); `summary_by_xi_o_panels.csv` is a copy).
LaTeX: `size_shift_table.tex`.
