# Adaptive mesh refinement method

Adaptive mesh refinement is intentionally not implemented in Phase 1. The current uniform-grid solver establishes a verified conservative baseline against which later AMR results will be compared.

The next phase will introduce a coarse grid and fine child patches with conservative prolongation and restriction. Coarse-fine ghost filling, subcycling, and refluxing will be added only after those data structures and transfer operations have independent tests.

