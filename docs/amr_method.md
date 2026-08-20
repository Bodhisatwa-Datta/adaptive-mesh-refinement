# Adaptive mesh refinement method

## Current hierarchy

The one-dimensional hierarchy is a tree rooted at a uniform base patch. A child patch covers a half-open range $[i_s,i_e)$ of cells in its parent and has physical bounds aligned exactly with the corresponding parent cell edges. For refinement ratio $r$, the child contains

$$
N_f = r(i_e-i_s)
$$

valid cells and has spacing $\Delta x_f=\Delta x_c/r$. Ratios greater than one are supported, with $r=2$ used by default. Sibling patches may touch but cannot overlap. A patch may itself own children, so the same representation supports more than two levels.

`n_stored_cells` counts arrays on every level, including covered coarse cells. `n_active_cells` counts leaf cells and excludes coarse cells covered by children. Keeping these definitions separate prevents ambiguous cost reporting later.

## Rectangular two-dimensional hierarchy

The 2D extension uses arrays with shape $(N_y,N_x)$ and represents a child region as two half-open parent ranges, $[j_s,j_e)\times[i_s,i_e)$. For the same refinement ratio $r$ in each coordinate, the child shape is

$$
(N_y^f,N_x^f)=\left(r(j_e-j_s),r(i_e-i_s)\right).
$$

Physical bounds coincide with the selected parent cell edges. Rectangular siblings may touch along an edge or corner but cannot overlap in area. The tree supports nested static patches and counts active leaf cells by subtracting each non-overlapping child rectangle from its parent.

Piecewise-constant 2D prolongation copies each parent average to its $r\times r$ fine children. Restriction applies the block average

$$
U_{i,j}^c=\frac{1}{r^2}\sum_{p=0}^{r-1}\sum_{q=0}^{r-1}
U_{ri+p,rj+q}^f,
$$

so the area integral is preserved exactly apart from floating-point summation. The 2D composite-mass diagnostic excludes rectangular parent regions covered by children at every tree level.

## Refinement criterion

The absolute centred-gradient indicator is

$$
R_i=\frac{|U_{i+1}-U_{i-1}|}{2\Delta x}.
$$

The optional dimensionless form is

$$
R_i^{\mathrm{norm}}=
\frac{|U_{i+1}-U_{i-1}|}{|U_i|+\epsilon}.
$$

Cells are flagged only when $R_i$ is strictly greater than a configurable threshold. Flags can be expanded by a configurable number of buffer cells, with either periodic wrapping or clipping at physical boundaries. Contiguous ranges are extracted deterministically, and ranges separated by a configurable small gap may be merged.

In two dimensions, centred differences in each coordinate form the gradient-magnitude indicator. Flags can be square-buffered with periodic wrapping or bounded clipping. Deterministic eight-connected-component clustering converts each separated flagged region into its own half-open rectangular box. Connectivity itself does not wrap across periodic boundaries, so features near opposite edges remain separate patches. An optional nonnegative merge gap combines boxes whose separation in both coordinates is no greater than the configured number of parent cells.

## Conservative prolongation

The robust baseline is piecewise constant. Every one of the $r$ fine children receives the parent average:

$$
U_{ir+j}^{f}=U_i^c,\qquad j=0,\ldots,r-1.
$$

Consequently,

$$
\frac{1}{r}\sum_{j=0}^{r-1}U_{ir+j}^{f}=U_i^c.
$$

This method is conservative and robust but only first-order accurate.

The hierarchy also supports conservative piecewise-linear reconstruction. A monotonized-central limiter computes a dimensionless parent-cell slope $s_i$. Fine child $j$ receives

$$
U_{ir+j}^{f}=U_i^c+s_i\left(\frac{j+1/2}{r}-\frac12\right).
$$

The subcell offsets sum to zero, so the children retain the parent average exactly. The limiter prevents the reconstruction from creating new extrema.

For smooth fields, an unlimited quadratic reconstruction is also available. With

$$
b_i=\frac{U_{i+1}^c-U_{i-1}^c}{2},\qquad
a_i=\frac{U_{i+1}^c-2U_i^c+U_{i-1}^c}{2},
$$

and $\delta_j=(j+1/2)/r-1/2$, the fine average is

$$
U_{ir+j}^f=U_i^c+b_i\delta_j+a_i\left[\delta_j^2+\frac{1/r^2-1}{12}\right].
$$

Both bracketed terms average to zero over the $r$ children, so the transfer is conservative and exact for quadratic cell-average data. Unlike the limited-linear method, it can create new extrema and is therefore used only for the smooth Gaussian diffusion benchmark. In the measured 64-cell AMR case it reduces $L_1$ from $8.83\times10^{-5}$ to $8.23\times10^{-5}$.

## Restriction and derefinement

Fine values are restricted using the arithmetic average

$$
U_i^c=\frac{1}{r}\sum_{j=0}^{r-1}U_{ir+j}^{f}.
$$

Because $\Delta x_f=\Delta x_c/r$, the integrated quantity is unchanged. Removing a fine leaf patch restricts its current values onto the covered parent cells by default before detaching it from the hierarchy.

## Coarse-fine ghost cells

The time-dependent solvers require one ghost cell on each side of a fine patch. A ghost-cell centre covered by an attached fine patch receives that fine value, including across the periodic domain boundary. Otherwise, the value is interpolated from the root grid. Advection and Burgers retain piecewise-constant interpolation; diffusion uses periodic linear interpolation evaluated at the actual fine ghost-cell centre.

The current implementation deliberately supports this operation only between the root and level one. Deeper time-dependent levels require recursive spatial and temporal interpolation and are rejected explicitly.

## Composite solution

Diagnostics count only leaf cells. If $\mathcal{L}$ is the set of cells not covered by a finer patch, composite mass is

$$
M_{\mathrm{AMR}}=\sum_{i\in\mathcal{L}}U_i\Delta x_i.
$$

Covered coarse cells remain stored for synchronization but are excluded from physical integrals and error norms.

## Dynamic regridding

At a configured interval, the synchronized root solution is evaluated with the gradient indicator. Cells above the refinement threshold are flagged. Cells already covered by level one remain eligible down to a separate, lower derefinement threshold:

$$
R_{\mathrm{derefine}} \leq R_{\mathrm{refine}}.
$$

The union of newly flagged and retained cells is buffered and converted into merged parent-cell ranges in 1D or connected-component boxes in 2D. If these regions differ from the current layout, level one is replaced conservatively:

1. Restrict every old fine patch onto the root.
2. Initialize each requested patch by conservative prolongation.
3. Copy old fine values directly wherever old and new patches overlap.
4. Attach the replacement patches and record the regrid mass change.

Restriction makes the root integral equal to the old composite integral. Prolongation preserves each root average, while copied overlap data already has the same restricted average. The replacement therefore preserves composite mass to roundoff.

## Not yet implemented

More than one time-dependent fine level and higher-order flux reconstruction within AMR remain future work. Refluxing is implemented for one-level 1D and 2D advection and diffusion, as well as 1D Burgers flow. Conservative quadratic prolongation and bilinear coarse-parent ghost interpolation support smooth 2D diffusion. Multilevel recursive space-time interpolation and more efficient decomposition of nonrectangular connected 2D flag sets are not yet implemented.
