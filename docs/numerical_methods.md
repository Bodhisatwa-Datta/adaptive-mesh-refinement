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
