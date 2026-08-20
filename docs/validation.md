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

## Two-dimensional advection

`examples/advection_2d/run_gaussian_validation.py` translates an anisotropic periodic Gaussian with velocity $(0.7,-0.4)$ to $t=0.25$ on square grids with 24, 48, 96, and 192 cells per direction. The measured $L_1$ errors are $1.2771\times10^{-2}$, $7.2312\times10^{-3}$, $3.8980\times10^{-3}$, and $2.0384\times10^{-3}$. Successive orders are 0.821, 0.892, and 0.935, approaching the expected first-order rate. Absolute mass errors are no greater than $6.94\times10^{-18}$.

Tests additionally exercise every combination of velocity signs, a stationary field, exact one-cell translation at CFL one, the multidimensional CFL rejection path, coordinate orientation, and field-shape validation. The benchmark writes `benchmarks/convergence/advection_2d_gaussian.csv` and `figures/advection_2d_convergence.png`.

## Two-dimensional diffusion

`examples/diffusion_2d/run_fourier_validation.py` evolves exact finite-volume averages of a separable periodic Fourier product with modes $(1,2)$, diffusivity $D=0.01$, and final time $t=0.05$. On grids with 40, 80, 160, and 320 cells per direction, the measured $L_1$ errors are $3.9123\times10^{-5}$, $9.7104\times10^{-6}$, $2.4232\times10^{-6}$, and $6.0554\times10^{-7}$. Successive orders are 2.010, 2.003, and 2.001. The measured mass change is exactly zero at all four resolutions.

Independent tests check the multidimensional parabolic stability limit, discrete maximum principle, zero-diffusivity behavior, and exact one-step amplification for several discrete Fourier-mode pairs. The benchmark writes `benchmarks/convergence/diffusion_2d_fourier.csv` and `figures/diffusion_2d_convergence.png`.

## Static two-dimensional AMR infrastructure

Two-dimensional transfer tests use refinement ratios two, three, and four. They verify that repeating every coarse average over an $r\times r$ block and restricting by the block mean exactly recovers the coarse array and preserves its area integral. Hierarchy tests cover parent-edge alignment, rectangular geometry, touching siblings, overlap rejection, nested levels, stored and active cell counts, restriction before derefinement, and composite-mass preservation after fine data have changed.

The static example computes buffered gradient-magnitude flags from a periodic Gaussian rather than hard-coding a patch. On a 64-by-64 base grid, the resulting parent-cell box is $(17,47)\times(21,43)$ and produces 6,076 active cells across the composite grid. `examples/amr_2d/build_static_hierarchy.py` regenerates `figures/gradient_selected_amr_hierarchy_2d.png`.

The separated-feature example verifies deterministic multi-box clustering. Two Gaussian features produce parent boxes $(6,29)\times(11,34)$ and $(35,58)\times(30,53)$. The resulting hierarchy contains 7,270 active and 8,328 stored cells, compared with 10,648 active and 12,832 stored cells for one enclosing box: reductions of 31.7% and 35.1%, respectively. Tests also cover deterministic ordering, configurable gap merging, separation across periodic domain edges, hierarchy attachment, and conservation during multi-patch replacement. `examples/amr_2d/build_multi_patch_hierarchy.py` writes `benchmarks/uniform_vs_amr/multi_box_clustering_2d.csv` and `figures/multi_patch_hierarchy_2d.png`.

## Static two-dimensional AMR advection

`examples/amr_2d/run_static_advection.py` compares a 32-by-32 root grid, a gradient-selected static level-one patch, and a uniform 64-by-64 grid for diagonal Gaussian transport. Their measured $L_1$ errors are $3.4998\times10^{-3}$, $2.0378\times10^{-3}$, and $1.7999\times10^{-3}$. The AMR calculation uses 17,808 cell updates versus 28,672 on the uniform fine grid and has a signed mass change of $6.94\times10^{-18}$.

Automated tests verify all velocity-sign combinations, exact uniform-state preservation, equivalence to the uniform solver when no patches exist, face-area averaging, fine-fine interface exclusion, periodic patch boundaries, and explicit rejection of deeper time-dependent levels. The benchmark writes `benchmarks/uniform_vs_amr/advection_2d_static_amr.csv` and `figures/static_amr_advection_2d.png`.

## Dynamic two-dimensional AMR advection

`examples/amr_2d/run_dynamic_advection.py` transports a Gaussian diagonally to $t=0.5$ with root-level timesteps, two fine substeps per root step, refluxing, and patch replacement every two root steps. The rectangular patch moves from $(2,17)\times(2,17)$ to $(10,28)\times(5,23)$. The coarse, static AMR, dynamic AMR, and uniform fine $L_1$ errors are respectively $1.1877\times10^{-2}$, $8.9020\times10^{-3}$, $7.4983\times10^{-3}$, and $7.3439\times10^{-3}$.

Dynamic AMR uses 61,808 cell updates compared with 147,456 for uniform 64. Its nine recorded regrid events individually preserve mass to roundoff, and the final signed mass change is $3.47\times10^{-18}$. Tests separately verify subcycled uniform states, root and fine CFL accounting, accumulated flux-register conservation, overlap-data retention, hysteresis, complete derefinement, and diagonal box motion. The benchmark writes `benchmarks/uniform_vs_amr/advection_2d_dynamic_amr.csv` and `figures/dynamic_amr_advection_2d.png`.

## Dynamic two-dimensional AMR diffusion

`examples/amr_2d/run_dynamic_diffusion.py` evolves analytical cell averages of a localized periodic Gaussian to $t=0.1$ with $D=0.01$. The refinement box expands from $(7,25)\times(7,25)$ to $(5,27)\times(5,27)$. Uniform 32, static AMR, dynamic AMR, and uniform 64 give $L_1$ errors $1.5008\times10^{-4}$, $6.8002\times10^{-5}$, $6.0601\times10^{-5}$, and $3.8134\times10^{-5}$.

Dynamic AMR uses 47,360 cell updates compared with 86,016 for uniform 64. Its two regrid events use conservative quadratic initialization and retain overlap data. Bilinear parent ghost interpolation, $r^2=4$ fine substeps, shared 2D refluxing, and the complete integration preserve mass to roundoff. The benchmark writes `benchmarks/uniform_vs_amr/diffusion_2d_dynamic_amr.csv` and `figures/dynamic_amr_diffusion_2d.png`.

## Repeated two-dimensional performance study

`benchmarks/performance/run_2d_benchmark.py` measures advection and diffusion at base resolutions 24, 32, and 48. Each uniform coarse, dynamic AMR, and uniform twice-base case receives one untimed warm-up and seven complete timings. A separate complete run records peak traced allocations without contaminating the runtime samples. The timed scope includes initialization, initial regridding, integration, and diagnostics.

For advection, dynamic AMR reduces uniform fine-grid update counts from 62,208, 147,456, and 497,664 to 32,368, 61,808, and 175,984. The corresponding median runtimes are 0.07408 s, 0.10192 s, and 0.23677 s for AMR versus 0.00347 s, 0.00554 s, and 0.01163 s for the uniform fine grids. Traced peaks are 0.19, 0.26, and 0.60 MiB for AMR versus 0.20, 0.35, and 0.78 MiB for uniform fine.

For diffusion, dynamic AMR reduces uniform fine-grid update counts from 27,648, 86,016, and 433,152 to 16,192, 47,360, and 166,208. Median runtimes are 0.05324 s, 0.13946 s, and 0.36828 s for AMR versus 0.00417 s, 0.00481 s, and 0.01311 s for uniform fine. Traced peaks are 0.17, 0.27, and 0.46 MiB for AMR versus 0.15, 0.26, and 0.57 MiB for uniform fine.

Thus the current Python implementation converts fewer cell updates into neither lower runtime nor uniformly lower traced allocation. `tracemalloc` does not measure total process resident memory, and the recorded timings are specific to the platform captured in `benchmarks/performance/two_dimensional_benchmark_metadata.json`. Raw results are in `benchmarks/performance/two_dimensional_accuracy_runtime_memory.csv`; the script regenerates `figures/two_dimensional_performance.png`.

## Second-order advection

`examples/advection_1d/run_second_order_validation.py` transports a smooth periodic sinusoid with $a=1$, $C_{\mathrm{CFL}}=0.6$, and final time $t=0.5$. It compares the first-order upwind solver with MC-limited MUSCL reconstruction and SSP-RK2 time integration.

At resolutions 40, 80, 160, and 320, second-order $L_1$ errors are $1.1950\times10^{-2}$, $3.2556\times10^{-3}$, $8.6125\times10^{-4}$, and $2.2050\times10^{-4}$. Successive orders are 1.876, 1.918, and 1.966. The corresponding first-order errors are $6.0513\times10^{-2}$, $3.0818\times10^{-2}$, $1.5556\times10^{-2}$, and $7.8157\times10^{-3}$. At the finest resolution the higher-order error is about 35 times smaller. Signed mass changes remain below $8\times10^{-17}$.

Automated tests additionally cover positive, negative, and zero velocities, CFL rejection, and bounded transport of a discontinuous square pulse. The benchmark writes `benchmarks/convergence/advection_1d_second_order.csv` and `figures/advection_second_order_convergence.png`.

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

`examples/burgers_1d/run_second_order_validation.py` repeats the pre-shock study with MC-limited MUSCL reconstruction, local Rusanov fluxes, and SSP-RK2. At the same resolutions, $L_1$ errors are $3.3435\times10^{-4}$, $8.7400\times10^{-5}$, $2.1226\times10^{-5}$, and $5.3484\times10^{-6}$. Successive orders are 1.936, 2.042, and 1.989. At 400 cells this is approximately 69 times smaller than the first-order error measured with the same $C_{\mathrm{CFL}}=0.6$. Mass remains conserved to roundoff, and a separate test verifies no new extrema through $t=1.0$, after shock formation.

## Burgers shock refinement

The same sinusoidal initial condition is evolved to $t=1.0$, beyond the analytical shock time. Because the classical characteristic solution is no longer valid, errors are calculated against a uniform $N=2048$ Rusanov reference and are explicitly labelled as numerical-reference errors.

Uniform $N=64$ gives $L_1=1.0511\times10^{-2}$. Dynamic refluxed AMR with a 64-cell base finishes with 72 active cells and gives $L_1=5.9122\times10^{-3}$. Uniform $N=128$ gives $L_1=5.3813\times10^{-3}$. Corresponding update counts are 3584, 10704, and 14336.

The AMR solution remains within the initial range $[0.3,0.7]$, regridding changes mass only at roundoff, and total refluxed mass change is zero at displayed precision. The final two fine patches meet across the periodic origin and resolve the single wrapped shock.

## Periodic Gaussian diffusion

`examples/diffusion_1d/run_gaussian_validation.py` evolves a periodic image-sum Gaussian with $D=0.01$ to $t=0.05$. The initial state and analytical comparison are exact integrals over each finite-volume cell, not point samples at cell centres. Resolutions 50, 100, 200, and 400 give measured $L_1$ errors $4.4887\times10^{-4}$, $1.1375\times10^{-4}$, $2.9126\times10^{-5}$, and $7.2632\times10^{-6}$. The successive observed orders are 1.980, 1.965, and 2.004. Mass changes remain at or below $2.78\times10^{-17}$.

The script writes `benchmarks/convergence/diffusion_1d_gaussian.csv` and `figures/diffusion_gaussian_convergence.png`. Automated tests also check the stability limit, exact final time, uniform-state preservation, zero diffusivity, and rejection of unstable timesteps.

## Periodic Fourier-mode diffusion

`examples/diffusion_1d/run_fourier_validation.py` provides an independent smooth test using mode $m=2$. For wave number $k=2\pi m/L$, the exact state

$$
u(x,t)=\bar{u}+A e^{-Dk^2t}\sin(k(x-x_{\min})+\phi)
$$

is integrated analytically over every finite-volume cell. At resolutions 40, 80, 160, and 320, measured $L_1$ errors are $1.0903\times10^{-4}$, $2.6869\times10^{-5}$, $6.6934\times10^{-6}$, and $1.6719\times10^{-6}$. Successive orders are 2.021, 2.005, and 2.001, and the displayed mass change is zero for every grid. The script writes `benchmarks/convergence/diffusion_1d_fourier.csv` and `figures/diffusion_fourier_convergence.png`.

## Dynamic AMR diffusion

`examples/diffusion_1d/run_amr_diffusion.py` compares a dynamic AMR calculation on a 64-cell root against uniform 64- and 128-cell grids. The AMR calculation uses refinement ratio two, $r^2=4$ fine substeps per coarse step, smooth conservative quadratic patch initialization, linear coarse-fine ghost interpolation, restriction, and diffusive refluxing.

Uniform $N=64$ gives $L_1=2.8294\times10^{-4}$ with 384 cell updates. Dynamic AMR finishes with 98 active cells and gives $L_1=8.2265\times10^{-5}$ with 1920 cell updates. Uniform $N=128$ gives $L_1=6.9719\times10^{-5}$ with 2688 updates. The AMR mass change is $2.78\times10^{-17}$ and every recorded regrid preserves mass to roundoff.

This validates improvement relative to the base grid and conservation across moving coarse-fine interfaces. AMR is within 18% of the uniform fine-grid error while using 29% fewer updates, although the uniform 128-cell result remains more accurate.

## Repeated diffusion performance study

`benchmarks/performance/run_diffusion_benchmark.py` measures base resolutions 32, 64, and 128 against uniform grids at the base and twice-base resolutions. The recorded study uses one untimed warm-up and 15 complete timings per case. Timed work includes grid and hierarchy initialization, initial regridding, integration, and error diagnostics. Python, NumPy, platform, timer, and numerical parameters are stored in `benchmarks/performance/diffusion_accuracy_runtime_metadata.json`.

Dynamic AMR errors are $4.1981\times10^{-4}$, $8.2265\times10^{-5}$, and $1.9494\times10^{-5}$, giving successive observed orders of approximately 2.35 and 2.08. Uniform fine errors are $2.8294\times10^{-4}$, $6.9719\times10^{-5}$, and $1.7723\times10^{-5}$. The AMR-to-fine error ratio improves from 1.48 to 1.18 and 1.10 as the base grid is refined.

AMR uses 384, 1920, and 12192 cell updates, compared with 384, 2688, and 20992 for the uniform fine grids. The larger two cases reduce updates by approximately 29% and 42%. Median AMR runtimes are 0.002413 s, 0.004853 s, and 0.014023 s, while uniform fine medians are 0.000804 s, 0.001449 s, and 0.003683 s. AMR is therefore about 3.0–3.8 times slower in this environment despite the update-count reductions. Analytical cell-average evaluation is included in the timed initialization and diagnostics for every method.

The associated sensitivity study evaluates all combinations of refinement thresholds 0.5, 1.0, and 2.0 and buffer widths 2, 4, and 8 on the 64-cell root. Most configurations reach an $L_1$ plateau near $8.22\times10^{-5}$. The least-refined case—threshold 2.0 with a two-cell buffer—covers 43.75% of the root and gives $1.1245\times10^{-4}$. Increasing that buffer to four and eight cells lowers the error to $8.4841\times10^{-5}$ and $8.2246\times10^{-5}$. This isolates insufficient feature coverage as a failure mode only when restrictive flagging is paired with a narrow buffer.

`benchmarks/performance/diffusion_prolongation_comparison.csv` holds a controlled transfer comparison at base resolution 64. Piecewise-constant initialization gives $L_1=5.3031\times10^{-4}$, limited linear gives $8.8310\times10^{-5}$, and smooth conservative quadratic gives $8.2265\times10^{-5}$. All three preserve mass; quadratic transfer is selected for this smooth validation but is explicitly not monotonicity preserving.

Across the scaling, sensitivity, and transfer cases, total mass and individual regrid mass changes remain at floating-point roundoff. Raw scaling measurements, sensitivity measurements, transfer measurements, minimum and maximum timing samples, and the generated figure are retained for reproducibility.
