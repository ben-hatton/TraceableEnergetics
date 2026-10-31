# History of the reference state of available energy

How the Lorenz reference state came to be implemented, and the changes of course on the way. The implementation itself is
described in `ape_reference_state.tex` (physics, algorithms, decisions, and in its appendix what is not implemented).
This file replaced the implementation plan on 2026-11-01.

## Milestones

1. **Sorted rearrangement with its certificate, the reference heights and the available energy** (designed 2026-10-23,
   committed 2026-10-26, 2704e5e). Changes from the plan, with the user:
   - the horizontal-mean reference state, implemented first, was removed (2026-10-26): it is not a goal of the project;
   - the slot resolution, planned as a required input (`slots_per_row` in AN2D, `slots = K` equal-mass slots in FC2D),
     became the model rows in both models, FC2D's slots with the masses of the rows of the state. A state at rest is
     then its own reference state in FC2D too, with E_P - E_* of O(Δz⁴) rather than the O(Δz²) accepted at first;
   - the outputs were cut down (2026-10-26): no Σ_*, N_*², placement height or flags of multiple minima; the parcel's own
     height is a candidate for z_* rather than a second minimum (it flagged most parcels of a resting state);
   - AN2D beyond the end rows: v_* follows the compression of the end slot's parcels (2026-10-26). A constant v_* first
     gave every parcel of a neutral fluid a spurious Π = O(Δz²) at a wall, 10 % of the available energy of the rising
     bubble;
   - FC2D in an exactly neutral layer: ties of the coarse search at the wiggles of the monotone cubic put z_* at the
     bottom wall (accepted, 2026-10-26: Π₂ ~ 6e-10 J/kg, gone after the first time step);
   - root finding by Roots.A42 (Alefeld–Potra–Shi), because Brent's method stalled in BigFloat.
2. **Exact rearrangement** (2026-10-27, e7b4b27): the transportation simplex, generic in F, from the sorted placement
   only. HiGHS was ruled out as Float64-only. The warm start from the previous snapshot, banded pricing and incremental
   tree updates were deferred.
3. **∂ₜP_* from the diabatic tendencies** (2026-10-27, 1cb2465). FC2D's ∂ₜP_* at fixed height by dual numbers through the
   monotone cubic, after comparing a hand-written tangent (the same to 1e-16) and the continuous formula (0.4 to 0.7 %
   off); AN2D's ∂ₜb_e beyond the end rows neglected (at most 2.6e-3 of max|∂ₜπ_*|). The test of the weights under strong
   thermobaricity moved to milestone 2, as it needs the exact solver.
4. **Budgets of ACE and APE** (2026-10-27, 15ba704): in Bernoulli form, exact at each cell by the product rule of the operators,
   with the forms of the energy notes and the residuals of the manipulation as diagnostics.

## Later changes

- **Lorenz profile from the slot volumes** (2026-10-27, 15ba704): first the duals of the placement (π_* = -ψ_k), which change
  abruptly when the parcel split between two rows changes and make split parcels' z_* jump, so that the generation, ∫ρΠ
  and C_APE - C_b,2* were rough (by two orders of magnitude in the lock exchange). The duals now only certify.
- **Float32 and BigFloat tests dropped** (2026-10-27, 063f198): only Float64 matters for now; the code stays generic in F.
  The exact rearrangement needs Float64 anyway (Float32 cannot resolve thermobaric reordering).
- **Source organised by topic** (2026-10-27, d2b89f7): `src/energetics/`, `src/reference_state/`, `src/numerics/`.
- **Reference states built from shared parts** (2026-10-28, df11a35): `Parcels`, `Slots`, `ColumnPlacement`, with one
  implementation of the steps common to AN2D and FC2D.
- **Incremental tree updates** of the simplex and candidates sorted once per pass (2026-10-31): 455 ms to 50 ms in the
  hardest case (K = 80, N = 1600), with the same placements, pivots and duals.

## Facts worth keeping

- **Prototypes** in `prototypes/` gave reference results before the Julia code: `duals.py` (π_* = -ψ in AN2D; Σ m Π =
  E_P - E_* with z_* on the row centres), `dtpstar.py` (FC2D ∂ₜP_* against differences, 5e-6), `weights.py` (the
  N_*⁻²-weighted mean against the plain one under thermobaricity).
- **Interpolation is hand-written:** `Interpolations.CubicHermite` is Float64-only and non-public, `MonotonicInterpolation`
  allocates and takes no prescribed slopes, and neither differentiates with respect to the nodes, which FC2D's ∂ₜP_*
  needs.
- **Upstream limitations found** (not fixed; no permission to change those packages): ClimFluids `NonlinearBinaryFluid`
  returns Float64 for Float32 parameters; CFAnelastics `AnelasticReference` computes p_R in Float64 on a Float32 grid,
  AN2D cannot be built in BigFloat (FFT Poisson solver) and its Poisson solver fails in Float32 (an FFTW plan on a
  misaligned array); CFCompressible `balanced_density` fails in BigFloat.
