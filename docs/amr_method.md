# Adaptive mesh refinement method

## Current hierarchy

The one-dimensional hierarchy is a tree rooted at a uniform base patch. A child patch covers a half-open range $[i_s,i_e)$ of cells in its parent and has physical bounds aligned exactly with the corresponding parent cell edges. For refinement ratio $r$, the child contains

$$
N_f = r(i_e-i_s)
$$

valid cells and has spacing $\Delta x_f=\Delta x_c/r$. Ratios greater than one are supported, with $r=2$ used by default. Sibling patches may touch but cannot overlap. A patch may itself own children, so the same representation supports more than two levels.

`n_stored_cells` counts arrays on every level, including covered coarse cells. `n_active_cells` counts leaf cells and excludes coarse cells covered by children. Keeping these definitions separate prevents ambiguous cost reporting later.

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

## Conservative prolongation

The initial prolongation method is piecewise constant. Every one of the $r$ fine children receives the parent average:

$$
U_{ir+j}^{f}=U_i^c,\qquad j=0,\ldots,r-1.
$$

Consequently,

$$
\frac{1}{r}\sum_{j=0}^{r-1}U_{ir+j}^{f}=U_i^c.
$$

This method is conservative and robust but only first-order accurate. Higher-order conservative interpolation is deferred.

## Restriction and derefinement

Fine values are restricted using the arithmetic average

$$
U_i^c=\frac{1}{r}\sum_{j=0}^{r-1}U_{ir+j}^{f}.
$$

Because $\Delta x_f=\Delta x_c/r$, the integrated quantity is unchanged. Removing a fine leaf patch restricts its current values onto the covered parent cells by default before detaching it from the hierarchy.

## Not yet implemented

The hierarchy is static and is not coupled to a PDE solver. Coarse-fine ghost filling, synchronized level updates, temporal subcycling, dynamic regridding with hysteresis, and refluxing remain future milestones.
