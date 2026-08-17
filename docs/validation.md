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

## Dynamic refinement check

`examples/amr_1d/run_dynamic_advection.py` transports the Gaussian from $x=0.25$ to $x=0.55$ over $t=0.3$. Regridding occurs every four fine-grid updates using refinement threshold 3.0, derefinement threshold 1.5, six buffer cells, and a four-cell merge gap.

Dynamic AMR obtains $L_1=8.0916\times10^{-3}$ with 99 final active cells, compared with $7.7606\times10^{-3}$ for uniform $N=128$. A fixed initial patch gives $L_1=1.7063\times10^{-2}$ after the Gaussian leaves its refined region, demonstrating the need for patch motion.

The synchronized dynamic calculation performs 6424 cell updates versus 6144 for uniform $N=128$. Subcycling reduces this to 4888 updates and gives $L_1=8.0001\times10^{-3}$. Without refluxing its mass change is $-1.4829\times10^{-6}$.

With refluxing enabled, the error is $L_1=8.0015\times10^{-3}$ and total mass change is $-2.78\times10^{-17}$. The largest mass change from an individual regrid is $8.33\times10^{-17}$. Tests cover positive and negative velocities, synchronized and subcycled refluxing, uniform-state preservation, and fine patches meeting across the periodic boundary.

The refluxed AMR calculation uses fewer cell updates than uniform $N=128$ for nearly equivalent error, but this alone is not a runtime result. Repeated wall-clock measurements are required before claiming an efficiency advantage.

## Repeated accuracy and runtime study

`benchmarks/performance/run_advection_benchmark.py` measures base resolutions 32, 64, and 128. Each case receives one untimed warm-up followed by seven complete initialization-and-integration timings using `time.perf_counter`. The script records Python, NumPy, and platform information beside the CSV.

Across the three resolutions, refluxed AMR $L_1$ errors are $1.5748\times10^{-2}$, $8.0015\times10^{-3}$, and $4.0309\times10^{-3}$. Successive orders are approximately 0.98 and 0.99. Equivalent uniform fine-grid errors are slightly lower.

AMR update counts are 1512, 4888, and 17152, compared with 1536, 6144, and 24576 on the fine uniform grids. Nevertheless, measured median AMR runtimes are 0.00654 s, 0.01008 s, and 0.02104 s, roughly 8–10 times the corresponding uniform fine-grid runtimes in this environment.

Therefore the present implementation demonstrates accuracy and update-count compression, but not wall-clock acceleration. The likely cause is Python-level hierarchy, ghost-fill, regridding, and small-array overhead relative to the cheap advection stencil. The raw samples' minimum and maximum, benchmark scope, and environment metadata are retained for reproducibility.

## Smooth Burgers convergence

Before shock formation, Burgers' equation has the characteristic solution

$$
u(x,t)=u_0(\xi),\qquad x=\xi+u_0(\xi)t.
$$

For $u_0(x)=0.5+0.2\sin(2\pi x)$, the first shock time is

$$
t_s=-\frac{1}{\min_x u_0'(x)}=\frac{1}{0.4\pi}\approx0.7958.
$$

The validation uses $t=0.2<t_s$ and solves the characteristic-foot equation by Newton iteration. At resolutions 50, 100, 200, and 400, measured $L_1$ errors are $2.3678\times10^{-3}$, $1.1509\times10^{-3}$, $5.5984\times10^{-4}$, and $2.7838\times10^{-4}$. Successive orders are 1.041, 1.040, and 1.008. Periodic mass changes remain at floating-point roundoff.

## Burgers shock refinement

The same sinusoidal initial condition is evolved to $t=1.0$, beyond the analytical shock time. Because the classical characteristic solution is no longer valid, errors are calculated against a uniform $N=2048$ Rusanov reference and are explicitly labelled as numerical-reference errors.

Uniform $N=64$ gives $L_1=1.0511\times10^{-2}$. Dynamic refluxed AMR with a 64-cell base finishes with 72 active cells and gives $L_1=5.9122\times10^{-3}$. Uniform $N=128$ gives $L_1=5.3813\times10^{-3}$. Corresponding update counts are 3584, 10704, and 14336.

The AMR solution remains within the initial range $[0.3,0.7]$, regridding changes mass only at roundoff, and total refluxed mass change is zero at displayed precision. The final two fine patches meet across the periodic origin and resolve the single wrapped shock.

## Periodic Gaussian diffusion

`examples/diffusion_1d/run_gaussian_validation.py` evolves a periodic image-sum Gaussian with $D=0.01$ to $t=0.05$ and compares cell-centre values with the analytical broadened Gaussian. Resolutions 50, 100, 200, and 400 give measured $L_1$ errors $4.5593\times10^{-4}$, $1.1416\times10^{-4}$, $2.9152\times10^{-5}$, and $7.2649\times10^{-6}$. The successive observed orders are 1.998, 1.969, and 2.005. Mass changes remain at or below $2.78\times10^{-17}$.

The script writes `benchmarks/convergence/diffusion_1d_gaussian.csv` and `figures/diffusion_gaussian_convergence.png`. Automated tests also check the stability limit, exact final time, uniform-state preservation, zero diffusivity, and rejection of unstable timesteps.

## Dynamic AMR diffusion

`examples/diffusion_1d/run_amr_diffusion.py` compares a dynamic AMR calculation on a 64-cell root against uniform 64- and 128-cell grids. The AMR calculation uses refinement ratio two, $r^2=4$ fine substeps per coarse step, conservative limited-linear patch initialization, linear coarse-fine ghost interpolation, restriction, and diffusive refluxing.

Uniform $N=64$ gives $L_1=2.8558\times10^{-4}$ with 384 cell updates. Dynamic AMR finishes with 98 active cells and gives $L_1=1.8816\times10^{-4}$ with 1920 cell updates. Uniform $N=128$ gives $L_1=6.9880\times10^{-5}$ with 2688 updates. The AMR mass change is $2.78\times10^{-17}$ and every recorded regrid preserves mass to roundoff.

This validates improvement relative to the base grid and conservation across moving coarse-fine interfaces. It does not show fine-grid-equivalent accuracy: the uniform 128-cell result remains more accurate. The update-count reduction also does not establish a runtime advantage; a repeated diffusion timing study remains future work.
