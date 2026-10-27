# Plan: the reference state of available potential energy

This is a handoff for a new session in this repository. Nothing in it has been implemented yet. It was written on
2026-10-23, at the end of a design session with the user.

**Read these first:**
- this file;
- `AGENTS.md`, whose rules apply throughout;
- `notes/ape_reference_state.tex`, the specification (cited below by section name).

## 1. Goal

Implement in `src/` the computation of the Lorenz reference state of available energy, for both models:
AN2D (CFAnelastics) and FC2D (CFCompressible).

- **The Lorenz reference state**: the rearrangement of the mass of the cells, each keeping its (ϑ, q), that has the least
  potential energy. (A horizontal-mean reference state was implemented in milestone 1 and removed with the user on
  2026-10-26: it is not a goal of the project.)
- **What is computed:**
  - per cell: the reference height z_*, the static energy Σ at z_*, and N_*²;
  - profiles: the reference pressure p_*(z, t) and the reference specific volume v_*(z, t);
  - the time derivative ∂_t p_*;
  - later, the available-energy budget diagnostics: Π = Π₁ + Π₂ and its budget terms.

The budget these quantities feed is defined in `~/Documents/latex/projects/energy_budgets/energy.tex`. That file is
outside this repository: read it, never edit it.

## 2. Settle these before writing code

1. **Uncommitted base (resolved).** The energy-budget diagnostics were committed (6e4ffae, f4d630b).
2. **Exact solver (decided by the user on 2026-10-23).** A pure-Julia transportation simplex, generic in F, warm-started
   from the sorted placement and from the previous snapshot. The note ("Solver 3: exact rearrangement", "Decisions") is
   updated. The design at §5 and a `slot` dimension in `DataFile` were confirmed, and Roots.jl was added.
3. **Recording during `loop` (decided by the user: a `book` keyword on `loop`; done).**
   - `CFAnelastics.loop` (0.1.3) and `CFCompressible.loop` (0.1.4) take `book = diagnostics(model)`, the book handed to
     `outputs`. Pass the book of `TraceableEnergetics.diagnostics`, which adds the energy budget to the diagnostics of
     the model package: `loop(state, model, time_scheme, params; outputs, book = diagnostics(model))`.
   - `examples/energy_budget/run.jl` records the energy budget this way. `examples/vortex_schemes/run.jl` still uses
     its older workaround, an output, `Record`, that holds its own book.
   - New reference-state diagnostics merged into the book of `TraceableEnergetics.diagnostics` are therefore recorded
     during a run with no further change.

## 2a. Status (2026-10-23): milestone 1 implemented, awaiting the user's review

- Code: the numerical methods in `src/numerics/` (`interpolation.jl`: Hermite interpolant and monotone slopes;
  `quadrature.jl`: Gauss–Legendre; `minimisation.jl`: global minimum of a function of one variable; `placement.jl`:
  north-west corner rule and staircase duals), the physics in `src/reference_state.jl` (the reference profile, the
  placement and its certificate, P_* and v_*: `reference_state!`, whose results are held by the `LorenzReference`) and
  `src/available_energy.jl` (static energy, LNB residual, N_*², and the kernels of the diagnostics z_*, Π, Σ m Π, E_P
  - E_* and E_*, which are not held by the reference state: as for the energetics, those on the cells write into arrays
  allocated by the book), for both models; the diagnostics entries in `src/diagnostics.jl`; `diagnostics`,
  `diagnostic_info` and `DataFile` take `reference_state`; locations `(nothing, Cell())` (z profiles) and `(Slot(),)`
  (slot profiles); the flags are two diagnostics, `placement_flags` and `height_flags`. Tests: `test/reference_state.jl`.
- Simplified with the user (2026-10-26): outputs are z_*, Π and the flags on the cells, P_*, v_* on z, the slot
  profiles, Σ m Π, E_*, E_P - E_*, the count of certificate violations, and P_top and the volume residual (FC2D). Σ_*,
  N_*², the placement height and the other counts are no longer outputs (N_*² is still computed by `stratification`).
  The parcel's own height is a candidate for z_* but no longer counts as a second minimum (it flagged most parcels of a
  resting state with `MULTIPLE_MINIMA`). π_* is no longer an output: it is (P_* - P_R)/ρ_R, and stays the AN2D
  profile (interpolated directly, as the duals give it), with nodes at the walls (below).
- Choices made in the implementation, to confirm:
  - differences of h are integrals of v by 3-point Gauss–Legendre (exact for NonlinearBinaryFluid), pressures are
    interpolated as anomalies from an anchor, so that Float32 keeps the 1e-4 J/kg differences;
  - at a degenerate boundary of the staircase the step of ψ is the middle of its allowed interval;
  - π_* node slopes (AN2D Lorenz) are g (ρ_R v_*,k - 1) with the slot-mean v at P_R(z_k);
  - P_* (FC2D Lorenz) is PCHIP through the slot edges, with second-order (three-point) end slopes;
  - root finding by Roots.A42 (Alefeld–Potra–Shi), because Brent's method stalls in BigFloat.
- Open questions for the user:
  - AN2D extension beyond the end nodes, as nodes of π_* at the walls rather than an extension at each evaluation, so
    that the nodes span the box as those of P_* in FC2D: v_* follows the compressibility of the end slot's own parcels,
    v_* = v_e + b_e (P_R - P_e) (decided with the user, 2026-10-26). The constant v_* chosen first gave every parcel of
    a neutral fluid a spurious Π = O(Δz²) at a wall (10 % of the available energy of examples/energy_budget); now a
    neutral or resting fluid has Π at round-off for a uniform ρ_R, O(Δz⁴) otherwise;
  - FC2D resting mismatch: O(Δz²) from the model grid and O(1/K²) from the slots, which nearly cancel at K = M_z,
    leaving about 1e-7 M g L_z that behaves like O(Δz).
  - FC2D reference heights in an exactly neutral layer (accepted by the user, 2026-10-26, no change): identical parcels
    with N_*² = 0 have a static energy along P_* that is flat up to the wiggles of the monotone cubic (node slopes only
    second-order accurate), about 1e-10 J/kg at M_z = 32. The coarse search sees ties at the slot edges and takes the
    lowest, the bottom wall, so at t = 0 of examples/energy_budget 1484 of 2048 parcels have z_* = 0. Their Π₂ is about
    6e-10 J/kg (2e-7 J/m in all, against 1.4e-2 J/m for the bubble), and the ties vanish after the first time step. The
    same wiggles give ∫ρΠ at rest of 2e-5, 2.5e-6, 9e-9 and 6e-10 J/m at M_z = 8, 16, 32 and 64.
- Upstream limitations found (not fixed, no permission): ClimFluids NonlinearBinaryFluid returns Float64 for Float32
  parameters (Float64 literals); CFAnelastics `AnelasticReference` computes p_R in Float64 on a Float32 grid
  (`cellpoint`'s 0.5), and AN2D cannot be built in BigFloat (FFT Poisson solver); CFCompressible `balanced_density`
  fails in BigFloat (broadcast into an `undef` array).

## 3. Decisions already taken with the user

All of these are recorded in the note's "Decisions" section.

**Scope and structure**
- New files go in `src/`. Work proceeds in milestones (§6), with a check-in with the user after each.
- The reference state is computed online, as CookBook entries, at the outputs of `loop`. These are at t = 0 and at the
  end of each slice, as set by `Nslice` and `slice_size`. There is also a plain function for offline use.
- Follow the existing diagnostics pattern: define the entries in their own function and merge them into the book of
  `diagnostics(model; ...)`, as `energetics_entries` and `add_energetics!` do in `src/diagnostics.jl`.

**Physics choices**
- The available-energy budget is evaluated from the continuous formulas, and residuals are reported, as
  `conversion_residual` does. Exact closure at each cell is not required.
- The Lorenz profile is taken from the slot-mean specific volumes, in both models; the duals certify the placement only
  (changed with the user on 2026-10-27: the dual profile of AN2D changes abruptly when the parcel split between two rows
  changes, and makes the reference heights of split parcels jump between rows, which made G, ∫ρΠ and C_APE - C_b,2*
  rough; in AN2D π_* integrates -v_* dP_R - g dz with v_* linear in P_R between the rows).
- The reference height used in the budgets is the LNB z_*: the global minimiser of Σ(ϑ, q, z, p_*(z, t)) over
  [0, L_z]. It is neither Tailleux (2013)'s nearest root nor the placement height ẑ.
- ∂_t p_* is computed by default from the diabatic tendencies: an N_*⁻²-weighted mean of the parcels at each level,
  valid for any equation of state. Differences of successive snapshots serve as a check.
- FC2D's equal-mass slots give an O(Δz²) mismatch of E_P - E_* for a resting state. This is accepted, but must be reported
  wherever FC2D results appear.

**Inputs from the user**
- The slot resolution is a **required** keyword, chosen by the user: `slots_per_row = s` in AN2D (K = s M_z), and
  `slots = K` in FC2D.

**Precision**
- Any F: Float32, Float64, Float128 or BigFloat. See "Precision" in `AGENTS.md`.

**Names**
- Quantities: `p_star`, `z_star`, `v_star`, `N2_star`, `pi_star` (= (p_* − p_R)/ρ_R, AN2D) and `p_top` (FC2D).
- Types: `LorenzReference`. Solver types: `SortedRearrangement` and `ExactRearrangement`.
- Words: "rearrangement", "placement" and "placement height". Never "transport", which means advection in a fluid model.

**Dependencies**
- Use existing packages, unless our own code removes a bottleneck or the package breaks the precision rule.
- Roots.jl (Brent) and Base `sortperm!` are fine. Both are generic in F.
- Interpolation must be hand-written:
  - `Interpolations.CubicHermite` is Float64-only and non-public;
  - `MonotonicInterpolation` reallocates on construction and can't take prescribed slopes;
  - neither gives derivatives with respect to the node positions, which FC2D's ∂_t p_* needs.

**Output**
- Profiles are 1D variables: on the z dimension for row profiles, and on a new `slot` dimension for slot profiles. They
  are not full cell fields, so `DataFile` must be extended for them.
- Flags and counts are fields of the result and scalar diagnostics, with no logging inside loops. They cover:
  - split, wall-pinned and multiple-minimum parcels;
  - the certificate status;
  - slots where the weights were replaced.

**Comments**
- Write equations in full. Never cite the notes.

## 4. Facts established (don't rediscover)

**Thermodynamics**
- `fluid(:p, :consvar, :q).volume_functions(p, ϑ, q)` returns `(v, ∂v/∂p, ∂v/∂ϑ, ∂v/∂q)`. This holds in ClimFluids 0.1.6
  for `NonlinearBinaryFluid`. `fwdd_volume_functions` (ForwardDiff) covers other fluids.
- Other functions: `specific_enthalpy`, `specific_volume`, `conjugate_variable` (h_ϑ), `modified_chemical_potential`
  (h_q) and `sound_speed2`.
- ClimFluids promotes to the type of the fluid's parameters: a Float64 fluid returns Float64 even for Float32 inputs. So
  require `eltype(fluid) == F`.

**Model state**
- AN2D: `state.m[:, :, 1:2]` is (ρ_R ϑ, ρ_R q).
- `model.reference` (`AnelasticReference`) holds `ρ_R` and `p_R` at the cell centres of the full z axis, halo included,
  and `pref`. The discrete hydrostatic balance is δ_z p_R = −g ℑ_z ρ_R.
- FC2D: `state.m[:, :, 1:3]` is (ρ, ρϑ, ρq).
- The geopotential is Φ = g(z − z0), with `z0 = model.space.z0`.

**Recipe inputs** (model diagnostics that both models provide)
- `conservative_variable`, `composition`, `density`, `pressure`, `temperature`;
- `conjugate_variable`, `modified_chemical_potential`;
- `conservative_variable_production` (σ_ϑ), `div_consflux` (∇·J_ϑ), `div_qflux` (∇·J_q);
- `mass_flux_x`, `mass_flux_z`, `dρconsvar`, `dρq`;
- FC2D only: `dρ`. AN2D only: `kinematic_pressure_anomaly`.
- The diabatic material rates follow from these: ρ Dϑ/Dt = `conservative_variable_production` − `div_consflux`, and
  ρ Dq/Dt = −`div_qflux`. Here ρ = ρ_R in AN2D.

**CookBook mechanics**
- Recipes declare their inputs by argument name.
- Output arrays are allocated on first use by `out(name)`; see `add_energetics!`.
- `entry(long_name, units, location, recipe)` is defined in `src/energetics.jl`.
- `DataFile` handles locations that are tuples of `Cell()`/`Face()`, and `()` for numbers. It doesn't handle 1D
  locations yet.

**Precision of costs**
- h is about 10⁴ J/kg. The cost differences that decide the ordering are about 10⁻⁴ J/kg, below Float32 resolution.
- So measure each parcel's costs relative to its own level, c_nk − c_n,own. Adding a constant per parcel doesn't change
  the optimum.
- Sum E_P − E_* parcel by parcel, Σ_n m_n (c_n,own − u_n), not as a difference of E_P and E_*.

**Verified numerically** with the Python prototypes in `notes/prototypes/`. They are reference results for the Julia
tests.
- `duals.py` (exact LP):
  - In AN2D, π_* = −ψ: the duals of the slot constraints are the reduced reference pressure, up to a constant.
  - With z_* restricted to the row centres, Σ_n m_n A_n = E_P − E_* (to 2×10⁻¹¹).
  - For a parcel split between rows k and k′, π_k − π_k′ = c_nk′ − c_nk exactly.
  - The horizontal-mean profile gives a larger Σ m Π, as weak duality requires.
  - In FC2D, ψ_k ≈ −Φ(z_k) to O(Δz).
- `dtpstar.py`: the FC2D formula for ∂_t p_* matches finite differences of the full construction to 5×10⁻⁶. The formula
  covers ∂_t p_top, the slot tendencies ∂_t v_*, and the motion of the slot edges.
- `weights.py`: the test case has strong thermobaricity, with two density-compensated parcel families of different
  v_P. There, the exact LP's finite-difference derivative of v_* matches the N_*⁻²-weighted level mean to 0.6%, while
  the plain mean is 20% off. The finite difference includes the reordering of parcels.

## 5. Design (confirm with the user at the start of milestone 1)

**Files**
- `src/rearrangement.jl`:
  - slots and costs;
  - the sorted placement;
  - duals along the sorted placement (a staircase);
  - the certificate;
  - the exact rearrangement (milestone 2).
- `src/reference_profiles.jl`:
  - a cubic Hermite interpolant, generic in F and updated in place, giving values, derivatives and derivatives with
    respect to the node positions;
  - the LNB residual;
  - the global minimum of Σ along the profile: a coarse pass over the slot centres, then refinement with Brent's method;
  - N_*².
- `src/reference_state.jl`:
  - `LorenzReference` and its constructors;
  - `reference_state!`, with the AN2D and FC2D specifics;
  - `reference_state_entries` and `add_reference_state!`.
- `src/reference_tendency.jl` (milestone 3).
- `test/reference_state.jl`, included from `test/runtests.jl`.
- `src/output.jl`: 1D locations and a `slot` dimension in `DataFile`. The location convention for 1D profiles is a
  design point to check with the user.

**API sketch**
```julia
ref = LorenzReference(model, SortedRearrangement(); slots_per_row = 1)      # AN2D (required keyword)
ref = LorenzReference(model, ExactRearrangement(; band = 2); slots = 128)   # FC2D (required keyword)
reference_state!(ref, model, state)                                         # in place; returns ref
reference_tendency!(ref, model, state, ϑdot, qdot)                          # ∂ₜp_star from the material rates
book = diagnostics(model; reference, reference_state = ref)                 # merges the new entries
```

**Workspace** (allocated once, at construction)
- O(N) vectors:
  - the sort key, the permutation and u;
  - per-cell outputs as cell arrays from `alloc`, so that they can be diagnostics.
- The placement as sparse triplets (parcel, slot, mass), at most N+K−1 of them. Never an N×K matrix.
- O(K) slot arrays: mass, pressure, ψ, v_*, edges and Hermite data.
- Nothing state-sized is allocated per call.

**Loops**
- Loops over parcels are loops over interior cells with `@with mgr, let ... @vec for i in ..., j in ...`, for threads
  and SIMD. The inner loop over slots is serial.
- The O(NK) certificate pass doubles as the coarse search for the LNB.

**Per snapshot, Lorenz state, AN2D**
1. Parcels are the interior cells, with m_n = ρ_R,j Δx Δz.
2. Slots are the rows, s per row, with M_k = M_x ρ_R,k Δx Δz / s.
3. Costs are c_nk = h(p_R,k, ϑ_n, q_n) + g(z_k − z0), measured relative to the parcel's own row.
4. The sort key is v(p_key, ϑ, q), with p_key = p_R at mid-depth.
5. Fill the slots from the top in that order; this gives the placement.
6. Compute the duals along the staircase.
7. Certificate: c_nk − u_n − ψ_k ≥ 0 for all parcel–slot pairs.
8. If the certificate fails and the solver is `ExactRearrangement`, pivot.
9. Profile: π_*,k = −ψ_k, with a free constant. The fallback is the slot-mean v_* with the trapezoidal hydrostatic rule.
10. Interpolate p_R and π_* with Hermite interpolants; for p_R, use node slopes −g ρ_R. Then derive
    ρ_R(z) = −p_R′/g and v_* = (π_*′/g + 1)/ρ_R.
11. Per parcel, z_* is the global minimum of S(z) = h(p_R(z), ϑ, q) + Φ(z) + π_*(z). N_*² = −dr/dz, with
    r = −dS/dz = g ρ_R (v(p_R(z)) − v_*(z)).

**Per snapshot, Lorenz state, FC2D**
1. K slots of equal mass M/K, counted from the top, with p_k = p_top + g(k − ½) M/(K L_x).
2. Costs are c_nk = h(p_k, ϑ, q).
3. Outer Newton iteration on p_top, with residual R = Σ x v − L_x L_z and R′ = Σ x ∂v/∂p. Stop at a small multiple of
   eps(F).
4. Geometry: V_k = Σ_n x_nk v_nk. Each slot has thickness V_k/L_x, stacked from the bottom, and ρ_*,k = M_k/V_k.
5. Interpolate P_* through the edge pressures with a monotone C¹ Hermite interpolant, and set ρ_* = −P_*′/g.
6. Find z_* by the same LNB search.

**Horizontal mean:** see the note's section "Horizontal-mean reference state".

**∂_t p_\* (milestone 3):** see the note's section "Time derivative of the reference pressure". It covers the weighted
formulas, their discrete form, the lid term and the motion of the nodes.

## 6. Milestones (check in with the user after each)

1. Sorted rearrangement with its certificate and the LNB, for AN2D and FC2D. Includes CookBook
   entries, `DataFile` support for profiles, and the tests of §7.
2. Exact rearrangement, generic in F, warm-started from the sorted placement and from the previous snapshot.
3. ∂_t p_*: from the diabatic tendencies, plus an offline check by differences of snapshots, reading `p_star` from a
   `DataFile`.
4. The available-energy budget entries from `energy.tex` (sections "Available energy" and "Available energy budgets"),
   with residuals reported. They are:
   - Π, Π₁, Π₂;
   - the fluxes;
   - the heating terms at the intermediate state and at the reference state;
   - the R terms.

   Status (2026-10-26): done apart from the R terms and the tendencies ∂ₜ(ρΠ₁), ∂ₜ(ρΠ₂), which wait for milestone 3
   (decided with the user). Decided with the user: the budgets in Bernoulli form, exact at each cell by the product rule
   of the operators (as the budget of ρP), with the forms of the energy notes (C_b,1*, C_b,2*, C_D*, ∇·J_C*) and the
   residuals of the manipulation reported; names with the energy prefix and `_at_p_star`/`_at_z_star`. Kernels in
   `src/available_energy_budget.jl`, entries `available_energy_budget_entries` in `src/diagnostics.jl`, tests in
   `test/available_energy_budget.jl` (closure at each cell with the tendency at a fixed reference state computed in the
   test). Found:
   - the conversion C = C_ACE + C_APE + C_bg, with C_bg of the background zero in the continuous equations; C_bg and
     C_APE - C_b,2* converge at about second order in the mean but first order in the maximum (v_* is only continuous,
     so z_* has kinks in (ϑ, q)): about 2 % of the conversion at M_z = 32;
   - in an exactly neutral layer z_* is arbitrary, and although Π₂ does not depend on it, Σ*_ϑ = h_ϑ(P_*(z_*)) does, so
     the heatings at the reference state Q*_ϑ, and the generation G, depend on how ties are broken: at t = 0 of
     examples/energy_budget G is -1.1e-4 W/m in AN2D (ties at the top of the neutral layer) and -2.7e-4 W/m in FC2D (at
     the bottom wall or the parcel's own height); from the first output on, the parcels differ and the models agree.
   - examples/thermohaline_lock_exchange (isochoric warm-salty against cold-fresh, k_T = 1e-4, k_q = 3e-5): FC2D starts
     with ACE = 12 % of the APE, exchanged with K by sound waves (16 to 32 Hz, undamped) through C_ACE; G_ACE follows the
     acoustic pressure but averages to zero; heat diffusion destroys available energy and salt diffusion generates it,
     of similar sizes; the APE budgets of AN2D and FC2D nearly coincide. The global budgets are recorded every 2.5 ms
     (`IntegralSeries`), the slow ones plotted as block means over 0.25 s. The sorted rearrangement fails the
     certificate for up to 9 parcels (thermobaricity), a case for milestone 2.

## 7. Tests

`AGENTS.md` requires tests to cover limiting and extreme cases, in several float types, and to find where things break.

**Milestone 1**
- **Resting state**, horizontally uniform and stably stratified:
  - the placement is the identity, ẑ = z and E_* = E_P;
  - Π = 0, with the dual profile and z_* on the row centres;
  - to round-off in AN2D with s = 1, and to O(Δz²) in FC2D and in AN2D with s > 1, shown converging.
- **Fully mixed fluid**, all parcels equal: ties, degenerate duals, N_*² = 0, and Π = 0.
- **Analytic cases:** an inverted column and a two-layer step, which have an analytic E_P − E_*.
- **Separable EOS** (γ = 0): the certificate holds, and E_* matches a brute-force LP solution on a small N.
- **Strong thermobaricity:** strong enough that the certificate fails; it must be detected and reported. The exact
  solver comes in milestone 2.
- **Degenerate sizes and strong compressibility:** M_x = 1, K = 1 and K > N, and very strong compressibility in FC2D.
- **Invariance** under:
  - a permutation of the parcels (a periodic shift in x);
  - a change of z0;
  - a constant added to π_*.
- **Weak duality:** any profile gives Σ m Π ≥ E_P − E_*, with equality for the optimal duals (AN2D, s = 1).
- **LNB:**
  - r(z_*) = 0 at interior minima, N_*² ≥ 0 and Π ≥ 0;
  - wall minima are flagged, with r of the outward sign.
- **Precision:** Float32, Float64, and BigFloat at small sizes. Find the stratification below which Float32 fails.
- **Allocations:** check `reference_state!` after its first call. Allocations must be bounded, with nothing state-sized.

**Milestone 3**
- the weighting test of `weights.py`;
- ∂_t p_top and the box volume;
- zero tendency without diffusion;
- ∫ R_A dV = 0.

## 8. Working rules

`AGENTS.md` applies in full. In short:
- Never commit; give the user the commands.
- Check in before design decisions.
- Keep AN/FC parity.
- Docstrings give units and sign conventions.
- Comments write the equations out in full and explain each nontrivial line.
