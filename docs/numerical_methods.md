# Numerical methods

## One-dimensional linear advection

The implemented equation is

$$
\frac{\partial u}{\partial t} + a\frac{\partial u}{\partial x}=0,
$$

with constant velocity $a$ on a periodic domain. Values $U_i$ represent cell averages on a uniform, cell-centred mesh. Integrating over a cell gives

$$
U_i^{n+1}=U_i^n-\frac{\Delta t}{\Delta x}
\left(F_{i+1/2}-F_{i-1/2}\right).
$$

The first-order upwind flux is

$$
F_{i+1/2}=\begin{cases}
aU_i, & a\geq 0,\\
aU_{i+1}, & a<0.
\end{cases}
$$

Time integration uses forward Euler. The timestep is controlled by
$\Delta t \leq C_{\mathrm{CFL}}\Delta x/|a|$, with $0<C_{\mathrm{CFL}}\leq1$.
The last timestep is shortened so the requested final time is reached exactly.

Periodic ghost cells provide the interface states at both physical boundaries. Because every flux enters one cell and leaves another, the discrete mass $\sum_i U_i\Delta x$ is conserved to roundoff.

This baseline method is deliberately first order. Its numerical diffusion is especially visible for discontinuous profiles; the separate higher-order method below provides the validated uniform-grid upgrade.

## Two-dimensional linear advection

The two-dimensional baseline solves

$$
\frac{\partial u}{\partial t}+a\frac{\partial u}{\partial x}
+b\frac{\partial u}{\partial y}=0
$$

on a periodic Cartesian grid. Arrays have shape $(N_y,N_x)$, so axis zero is the y direction and axis one is the x direction. Donor-cell upwinding is applied independently to the conservative x and y flux differences in one unsplit forward-Euler update. Stability requires

$$
\Delta t\left(\frac{|a|}{\Delta x}+\frac{|b|}{\Delta y}\right)
\leq C_{\mathrm{CFL}},\qquad 0<C_{\mathrm{CFL}}\leq1.
$$

Periodic neighbors are obtained by array shifts. The summed flux differences telescope in both directions, conserving $\sum_{i,j}U_{i,j}\Delta x\Delta y$ to floating-point roundoff.

## Two-dimensional explicit diffusion

The Cartesian diffusion solver applies the centred five-point Laplacian,

$$
L(U)_{i,j}=D\left(
\frac{U_{i,j+1}-2U_{i,j}+U_{i,j-1}}{\Delta x^2}
+\frac{U_{i+1,j}-2U_{i,j}+U_{i-1,j}}{\Delta y^2}
\right),
$$

with periodic indexing and forward Euler integration. The discrete maximum principle and linear stability require

$$
2D\Delta t\left(\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}\right)
\leq C_{\mathrm{stab}},\qquad 0<C_{\mathrm{stab}}\leq1.
$$

The implementation also exposes the exact discrete amplification factor for a Fourier mode $(m_x,m_y)$, allowing the stencil and time update to be tested independently of the continuous analytical solution.

On rectangular AMR patches, diffusion uses bilinear interpolation of synchronized or time-interpolated parent cell averages at fine ghost-cell centres. A ratio-$r$ child takes $r^2$ fine steps per root step. Diffusive x- and y-face fluxes are accumulated over those substeps and passed through the same face-averaged reflux operator used by 2D advection.

## Second-order linear advection

The higher-order uniform solver reconstructs interface states with a piecewise-linear MUSCL profile. Its dimensionless slope is the monotonized-central value

$$
s_i=\operatorname{minmod}\left(2(U_i-U_{i-1}),\frac{U_{i+1}-U_{i-1}}{2},2(U_{i+1}-U_i)\right).
$$

The left and right states at interface $i+1/2$ are

$$
U_{i+1/2}^{L}=U_i+\frac{s_i}{2},\qquad
U_{i+1/2}^{R}=U_{i+1}-\frac{s_{i+1}}{2},
$$

and the upwind state is selected from the sign of $a$. Time integration uses the two-stage strong-stability-preserving Runge–Kutta method

$$
U^{(1)}=U^n+\Delta t L(U^n),\qquad
U^{n+1}=\frac12U^n+\frac12\left[U^{(1)}+\Delta t L(U^{(1)})\right].
$$

The method is conservative, second order for smooth solutions, and bounded for the tested square pulse under the configured CFL limit. Limiting reduces the local order near smooth extrema and discontinuities.

The monotonized-central calculation is implemented once in the PDE-independent reconstruction module. Second-order advection, second-order Burgers, and conservative limited-linear AMR prolongation use the same tested limiter implementation.

For a periodic sequence of cell averages, the discrete total variation is

\[
\operatorname{TV}(U) = \sum_i \lvert U_{i+1} - U_i \rvert,
\]

where the final difference wraps from the last cell to the first. Regression tests verify that this quantity does not increase for limited square-pulse advection or post-shock Burgers evolution.

## Synchronized AMR update

The first time-dependent AMR implementation uses one global timestep on all levels,

$$
\Delta t = C_{\mathrm{CFL}}\frac{\min_\ell\Delta x_\ell}{|a|}.
$$

For each timestep, the update order is:

1. Fill every fine patch's ghost cells from the old-time hierarchy.
2. Advance the periodic root grid by one forward-Euler step.
3. Advance every fine patch by the same timestep.
4. Restrict updated fine averages onto covered coarse cells.

At a fine boundary, an adjacent same-level patch supplies data first. If no fine patch covers the ghost-cell centre, piecewise-constant interpolation from the coarse parent is used. All ghost values are collected before any patch is updated, so every level uses data at the same old time.

This ordering synchronizes solution values but not interface fluxes. Until refluxing is implemented, the coarse flux and fine flux at a coarse-fine interface need not agree, so composite mass is measured after every benchmark rather than assumed to be conserved.

When dynamic regridding is enabled, the refinement decision is made after synchronization at a configurable number of timesteps. The next update therefore begins with a hierarchy whose fine patches were constructed from a single synchronized time. Regridding does not alter the physical time and is measured separately from the mass change caused by PDE evolution.

## Temporal subcycling

With refinement ratio $r$, the coarse level takes a CFL-controlled step $\Delta t_c$ while level one takes

$$
\Delta t_f=\frac{\Delta t_c}{r}
$$

for each of its $r$ substeps. The root is first advanced provisionally from $U_c^n$ to $U_c^{n+1}$. At fine substep $m=0,\ldots,r-1$, parent data for coarse-fine ghosts is interpolated to the fine substep's start time:

$$
U_c^{n+m/r}=\left(1-\frac{m}{r}\right)U_c^n+\frac{m}{r}U_c^{n+1}.
$$

All fine-patch ghosts are collected before any fine patch is advanced, preserving simultaneous same-level updates. After $r$ fine steps, fine averages are restricted onto the provisional root solution.

## Flux correction

For a coarse-fine interface, define the time-integrated coarse flux

$$
I_c=\Delta t_c F_c
$$

and the accumulated fine flux

$$
I_f=\sum_{m=0}^{r-1}\Delta t_f F_f^{(m)}.
$$

At the left edge of a fine patch, the adjacent uncovered coarse cell is corrected by

$$
U_{s-1}\leftarrow U_{s-1}+\frac{I_c-I_f}{\Delta x_c}.
$$

At the right edge, the correction is

$$
U_e\leftarrow U_e+\frac{I_f-I_c}{\Delta x_c}.
$$

No correction is applied where the neighbouring parent cell is also covered by a fine patch, because that is a fine-fine rather than coarse-fine interface. The same logic is applied across the periodic domain boundary.

### Two-dimensional synchronized update and refluxing

The first rectangular 2D AMR evolution uses the finest-grid global timestep. Before either level advances, each fine patch receives one ghost layer on all four sides. A same-level sibling supplies a ghost wherever it covers that ghost-cell centre; all remaining values come from the periodic parent field. Root and fine patches then take one donor-cell step from the same old time, and fine block averages overwrite covered root cells.

For refluxing, the fine flux density replacing one coarse face is the arithmetic mean of the $r$ aligned fine-face fluxes. On a vertical interface,

$$
\bar I_f^x=\frac{1}{r}\sum_{q=0}^{r-1}\Delta t F_{f,q}^x,
$$

with the analogous expression on horizontal faces. The neighboring uncovered coarse average receives $(I_c^x-\bar I_f^x)/\Delta x_c$ on a patch's left edge and the sign-reversed correction on its right edge; bottom and top use $\Delta y_c$. Corrections are skipped cell by cell when the neighboring parent cell is covered by another fine patch.

For 2D temporal subcycling, the root takes one multidimensional CFL step while level one takes $r$ steps of size $\Delta t_c/r$. Parent values at fine substep $m$ are linearly interpolated between the old and provisional new root fields. The x- and y-face fine flux arrays are integrated across all substeps before face averaging and reflux correction.

## Inviscid Burgers' equation

The uniform-grid nonlinear solver advances

$$
\frac{\partial u}{\partial t}+\frac{\partial f(u)}{\partial x}=0,
\qquad f(u)=\frac{u^2}{2}.
$$

For left and right interface states $u_L$ and $u_R$, the local Rusanov flux is

$$
F_{i+1/2}=\frac{1}{2}\left[f(u_L)+f(u_R)\right]
-\frac{1}{2}\alpha_{i+1/2}(u_R-u_L),
$$

where

$$
\alpha_{i+1/2}=\max(|u_L|,|u_R|).
$$

The timestep is recomputed from the evolving maximum characteristic speed,

$$
\Delta t\leq C_{\mathrm{CFL}}\frac{\Delta x}{\max_i|u_i|}.
$$

The Rusanov method is robust through shocks and conservative, but its first-order dissipation broadens discontinuities.

The AMR Burgers solver follows the same synchronization, subcycling, and flux-register sequence as linear advection. Its coarse timestep is recomputed using the largest absolute state on any stored patch. For subcycling,

$$
\Delta t_c\leq C_{\mathrm{CFL}}\frac{\Delta x_c}{\max_{\ell,i}|U_{\ell,i}|},
\qquad \Delta t_f=\frac{\Delta t_c}{r}.
$$

Rusanov fluxes at both sides of each fine patch are accumulated over the fine substeps and used for reflux correction.

The second-order uniform Burgers solver applies the same MC-limited MUSCL reconstruction and SSP-RK2 formula used by second-order linear advection. Reconstructed left and right states enter the local Rusanov flux, and the timestep is recomputed from the evolving maximum absolute cell state. The limiter maintains bounded shock evolution in the tested case while retaining second-order convergence before shock formation.

## Explicit diffusion

The diffusion solver advances

$$
\frac{\partial u}{\partial t}=D\frac{\partial^2u}{\partial x^2}
$$

in conservative form with centred interface fluxes

$$
F_{i+1/2}=-D\frac{U_{i+1}-U_i}{\Delta x},
\qquad
U_i^{n+1}=U_i^n-\frac{\Delta t}{\Delta x}
\left(F_{i+1/2}-F_{i-1/2}\right).
$$

Forward Euler is stable for

$$
\Delta t\leq C_D\frac{\Delta x^2}{2D},
\qquad 0<C_D\leq1.
$$

For discrete periodic Fourier mode $m$ on an $N$-cell grid, one timestep has the exact amplification factor

$$
G_m=1-4\frac{D\Delta t}{\Delta x^2}\sin^2\left(\frac{\pi m}{N}\right).
$$

The solver exposes this factor as a diagnostic, and automated tests verify the complete update against it for low, intermediate, and Nyquist modes. The stability restriction follows from requiring the most oscillatory mode to remain bounded.

The final step is shortened to reach the requested time exactly. Periodic interface fluxes telescope, so the uniform solver conserves discrete mass to roundoff.

Diffusion validation treats every $U_i$ as a finite-volume average. The analytical periodic Gaussian is integrated between the cell edges for both initialization and error calculation. This avoids mixing point samples with cell averages during conservative AMR prolongation.

For a refinement ratio $r$, the fine diffusive stability limit is $r^2$ smaller than the coarse limit. The AMR solver therefore takes $r^2$ fine substeps per coarse step. Parent states are interpolated linearly in time at each fine substep and linearly in space at coarse-fine ghost centres. Time-integrated diffusive fluxes are accumulated in the same interface register used by the hyperbolic solvers, then refluxed after restriction.
