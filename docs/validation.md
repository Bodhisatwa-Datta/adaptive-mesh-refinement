# Validation

The uniform-grid advection solver is validated using the exact periodic translation

$$
u(x,t)=u_0\!\left((x-at-x_{\min})\bmod L+x_{\min}\right),
\qquad L=x_{\max}-x_{\min}.
$$

The reported discrete norms are

$$
L_1=\frac{1}{N}\sum_i|e_i|,\qquad
L_2=\sqrt{\frac{1}{N}\sum_i e_i^2},\qquad
L_\infty=\max_i|e_i|.
$$

Observed order between grids with spacing $h$ and $h/2$ is

$$
p=\frac{\log(E_h/E_{h/2})}{\log 2}.
$$

Run `python examples/advection_1d/run_gaussian.py` after installing the package. The script calculates fresh results rather than relying on embedded benchmark data, writes them to `benchmarks/convergence/advection_1d_gaussian.csv`, and creates `figures/gaussian_advection_convergence.png`.

For $N=50,100,200,400$ at $t=0.5$, the measured $L_1$ errors are respectively $2.9581\times10^{-2}$, $1.5982\times10^{-2}$, $8.2487\times10^{-3}$, and $4.2252\times10^{-3}$. The successive observed orders are $0.888$, $0.954$, and $0.965$, approaching the theoretical first-order rate. Absolute mass errors are between $2.78\times10^{-17}$ and $8.33\times10^{-17}$.

## AMR infrastructure checks

Transfer tests use refinement ratios two, three, and four. They verify for every parent cell that prolongation followed by restriction recovers the original average and that

$$
\sum_j U_j^f\Delta x_f = \sum_i U_i^c\Delta x_c.
$$

Hierarchy tests separately verify coordinate alignment, valid index ranges, parent/child links, rejection of overlapping siblings, active-cell counting, multilevel trees, and restriction before derefinement. Gradient tests cover uniform-state preservation, periodic discontinuities, normalized scale invariance, buffer wrapping, bounded buffers, and region merging.

## Static AMR advection check

The static-patch benchmark uses a 64-cell root, refinement ratio two, $a=1$, $C_{\mathrm{CFL}}=0.8$, and final time $t=0.1$. The fine region is selected from the initial Gaussian by the configured gradient threshold and eight buffer cells. `examples/amr_1d/run_static_advection.py` writes all measurements to `benchmarks/uniform_vs_amr/static_advection_1d.csv`.

Measured $L_1$ errors are $5.2767\times10^{-3}$ for uniform $N=64$, $3.1261\times10^{-3}$ for 100 active AMR cells, and $2.6692\times10^{-3}$ for uniform $N=128$. The corresponding actual cell-update counts are 512, 2176, and 2048. The AMR count uses all 136 stored cells because covered root cells are advanced before restriction. This exposes the cost of global fine timesteps and redundant covered-cell updates in the initial synchronized algorithm.

The uniform calculations conserve mass to roundoff, while the static AMR calculation has signed mass change $+1.8991\times10^{-4}$. This is an expected limitation of synchronization by restriction without refluxing. The result is recorded as evidence for the need for flux correction; it is not presented as a conservative AMR result or as a runtime-efficiency result.
