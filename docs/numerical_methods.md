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

The method is deliberately first order. Its numerical diffusion is especially visible for discontinuous profiles; higher-order reconstruction is deferred until the uniform baseline is fully established.

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

The final step is shortened to reach the requested time exactly. Periodic interface fluxes telescope, so the uniform solver conserves discrete mass to roundoff.

Diffusion validation treats every $U_i$ as a finite-volume average. The analytical periodic Gaussian is integrated between the cell edges for both initialization and error calculation. This avoids mixing point samples with cell averages during conservative AMR prolongation.

For a refinement ratio $r$, the fine diffusive stability limit is $r^2$ smaller than the coarse limit. The AMR solver therefore takes $r^2$ fine substeps per coarse step. Parent states are interpolated linearly in time at each fine substep and linearly in space at coarse-fine ghost centres. Time-integrated diffusive fluxes are accumulated in the same interface register used by the hyperbolic solvers, then refluxed after restriction.
