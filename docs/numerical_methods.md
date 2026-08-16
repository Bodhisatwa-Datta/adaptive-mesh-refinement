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
