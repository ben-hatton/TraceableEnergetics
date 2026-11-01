# TraceableEnergetics — test cases, tutorials and validation experiments

Plan of the tutorials, verification tests and validation experiments of TraceableEnergetics, written with Claude (Oct 31, 2026). Exported from the Claude doc https://claude.ai/artifact/MjNmubjTZPv1wo3srNqESB; this file is the copy agents work from.

## Overview

The plan below has 40 experiments in nine groups, 17 of them high priority, run from 23 example scripts; most are checked against an exact discrete answer rather than by eye.

**What the code does.** FC2D (CFCompressible) and AN2D (CFAnelastics) are energy-consistent finite-volume models of a binary fluid in a 2D box on a C-grid. They share every kernel except mass conservation and pressure. FC2D advances (ρ, ρϑ, ρq, u, w); AN2D advances (ρ\_Rϑ, ρ\_Rq, u, w) under ∇·(ρ\_R u) = 0 with a Poisson solve. Both use the thermodynamic form of the momentum forcing, three vortex-force schemes (`VortexEnergyCons`, `VortexEnstrophyCons`, `VortexEnergyEnstrophyCons`), Stokes viscosity, and `ThermoFluxConsistent` or `ThermoFluxDowngradient` tracer fluxes. Walls can be free-slip, no-slip, moving or stressed, with flux, value or heat-flux tracer conditions. Time stepping is RK4 (or Kinnmark–Gray), with Δt fixed at the start of each output slice from a CFL bound. TraceableEnergetics adds model-independent setups, the energy budget, the Lorenz reference state (AE = ACE + APE) with its budgets, and outputs (NetCDF/JLD2, integral series, videos).

**What the default fluid makes visible.** With v0 = 10⁻³ m³ kg⁻¹ and α\_p = 10⁻⁶ Pa⁻¹, the sound speed is c = √(v0/α\_p) ≈ 31.6 m s⁻¹. A 0.1 m s⁻¹ flow then has Ma ≈ 3×10⁻³, and density varies by g L\_z/c² ≈ 0.5 % over the 0.5 m box. Thermodynamic corrections are tiny: the adiabatic gradient is α\_T g T/c\_p0 ≈ 1.4×10⁻⁴ K m⁻¹ (7×10⁻⁵ K over the box), and barodiffusion gives Δq ≈ 2×10⁻⁶ over the box. Consistency tests therefore need Float64 differences at those levels, or a deliberately magnified fluid (F1).

**Three methods used throughout.**

1. Compare with the discrete solution. With `Centred2`, a mode cos(kx) sees k̂ = (2/Δx) sin(kΔx/2), so diffusion rates, wave frequencies and growth rates have discrete formulae the model should match to time-stepping error, not merely to O(Δx²).
2. Linearise `tendencies!` about a base state on a small grid and take the eigenvalues of the Jacobian (W4). This gives the discrete dispersion relation and growth rates without deriving them.
3. Refine space and time separately (N2).

Every experiment lives in examples/ as a Literate script that a user can run interactively; related experiments share a script (see Where each experiment lives). An agent implementing the plan starts with the Implementation guide for agents, near the end.

| ID | Experiment | Main thing tested | Models | Reference value | Priority |
| --- | --- | --- | --- | --- | --- |
| T1 | Building both models from one recipe | every argument; rest stays at rest | both | tendencies at round-off | High |
| T2 | Choosing diagnostics | books, entries, sampling vs acoustic periods | both | integrals vs direct sums | Medium |
| T3 | Saving, reading, restarting, plotting | DataFile, IntegralSeries, videos, restart | both | bitwise restart | Medium |
| T4 | FC rest state as AN reference | hydrostatic consistency of the two models | both | P\_FC − P\_R per level | High |
| N1 | Sine diffusion and viscous heating | fluxes, walls, ρε, heating return | both | exp(−κk̂²t), closed-form heating | High |
| N2 | Convergence order | 2nd order in space, 4th in time | both | p = 2, p = 4 | High |
| N3 | Energy drift and round-off | RK4 energy error, eps(F) scaling | both | slope 5 in Δt; ∝ eps(F) | High |
| N4 | Manufactured solutions | every nonlinear term | both | p = 2 | Medium |
| N5 | Symmetry, reversibility, reproducibility | indexing, hidden dissipation, threads | both | round-off | Medium |
| N6 | Cellular (Taylor–Green) flow | vortex force, viscosity | both | steady; exp(−2νK̂²t) | Medium |
| N7 | Couette flow with viscous heating | moving wall, wall work, heat out | both | parabolic T profile | Medium |
| W1 | Internal gravity wave mode | N², buoyancy staggering | both | discrete IGW relation | High |
| W2 | Acoustic mode and sound speed | c from ClimFluids | FC | ω = c(k̂² + m̂²)^½ | High |
| W3 | Lamb wave | horizontal acoustics with w ≡ 0 | FC (absent in AN) | ω = ck̂ | Medium |
| W4 | Jacobian eigenvalues | discrete normal modes | both | method | High |
| I1 | Kelvin–Helmholtz growth and Ri cut-off | shear instability, J = 1/4 | both | σ = 0.19 U/δ; J = α(1 − α) | High |
| I2 | Rayleigh–Bénard onset | convective threshold | both | 1707.76 / 657.51 | High |
| I3 | Holmboe instability | sharp interface, Ri\_min | both | unstable for J > 1/4 | Low |
| I4 | Salt-finger growth | k\_T ≠ k\_q, binary fluid | both | linear finger theory | Medium |
| I5 | Rayleigh–Taylor single mode | strong buoyancy, bubble speed | both | σ² = Agk; Goncharov U\_b | Medium |
| A1 | Lamb's hydrostatic adjustment | ACE release, acoustic filtering | both | AN: no motion | High |
| A2 | Low-Mach convergence | FC → AN as c → ∞ | both | difference ∝ Ma² | High |
| A3 | Balanced vs unbalanced initial states | spurious acoustic energy | both | ACE at t = 0 | Medium |
| F1 | Diffusion to equilibrium | equilibrium of each flux law | both | isothermal T, barodiffusive q | High |
| F2 | Entropy vs θ as consvar | invariance of the consistent fluxes | both | consistent θ and s runs identical | Medium |
| F3 | Laminar-to-turbulent flux sweep | when the flux law matters | both | 1/Cox scaling | Medium |
| F4 | AN buoyancy schemes | energy residual ∝ (ρ′/ρ)² | AN | 0, 0, O((ρ′/ρ)²) | Medium |
| E1 | Cabbeling of a compensated front | nonlinear EOS, diabatic APE source | both | erf² density anomaly; rest if α\_TT = 0 | High |
| E2 | Analytic APE checks | reference state and available-energy diagnostics | both | ΔρgH²L/8; ρ0N²⟨ζ²⟩/2; α\_PΔp²/2 | High |
| E3 | Deep regime | ρ\_R varying by \~30 % | both | energy closure; Alboussière–Ricard Ra\_c | Medium |
| E4 | Thermobaric instability | pressure-dependent expansion | both | threshold d\_c = S/(α\_TγΔθρ0g) | Medium |
| B1 | Lock exchange | gravity current, spurious mixing | both | Fr = 1/2; ΔRPE | High |
| B2 | Lid-driven cavity | no-slip, moving lid, corners | both | Ghia; Botella–Peyret | Medium |
| B3 | Differentially heated cavity | value walls, heat transport | both | Nu ≈ 8.8 at Ra = 10⁶ | Medium |
| B4 | Dipole–wall collision | vorticity at no-slip walls | both | Clercx–Bruneau data | Medium |
| B5 | Under-resolved shear layer | under-resolution, vortex schemes | both | resolved run (after Minion–Brown) | Low |
| B6 | Skamarock–Klemp wave packet | dispersive IGW, advection | both | analytic linear solution | Medium |
| B7 | Horizontal convection | Paparella–Young bound | both | ε ≤ κ Δb/H | Medium |
| B8 | Rayleigh–Bénard Nu–Ra | exact dissipation–Nu identities | both | Shraiman–Siggia | Medium |
| S1 | Thermohaline RT–KH in a forced box | all pinch points at once | both | budgets to round-off | High |

## Tutorials (T1–T4)

Each tutorial should end in assertions, so it doubles as a smoke test; the repo already has most of the pieces in examples/.

### T1 · Building both models from one recipe

- **Goal:** one Literate script that builds FC2D and AN2D from a shared `physics` named tuple, explains each argument, and ends with checks.
- **Setup:** `Tank2D(Lx = 1, Lz = 0.5, g = 10)`, the seawater-like `NonlinearBinaryFluid` of the examples, `Box2D` 32×16 (Periodic, Bounded), closed free-slip walls. Rest state θ = T0 + 1 K · z/L\_z, q = q0 through `setup(model; θ, q)`; FC2D also needs `pref`.
- **Expected:** momentum tendencies at round-off (the canonical-setup bound, 100 eps c\_p0 T0/Δ); after 10³ steps |u| stays at round-off; mass, ∫ρθ, ∫ρq and E constant to round-off; `tendencies!` allocates 0 bytes on a serial manager.
- **Then:** the same script with `NonlinearBinaryFluid(:entropy, …)`, to show that setups are independent of the model and the entropic variable. Stay in Float64: the code is designed for it, and other float types are untested (N3).
- **Pertinence:** examples/kelvin\_helmholtz/run.jl already shows every piece, but mixes model building with a physics experiment and checks nothing.

| Choice | Options in the code | Used in examples |
| --- | --- | --- |
| Topology | (Periodic, Bounded), (Bounded, Bounded); AN2D needs bounded z | (Periodic, Bounded) |
| Tangential wall | FreeSlip, NoStress, NoSlip, TangentialVelocity(f), Stress(f) | FreeSlip |
| Tracer wall | Flux(f), Value(f), HeatFlux(f) for consvar | Flux(0) |
| Vortex force | VortexEnergyCons, VortexEnstrophyCons, VortexEnergyEnstrophyCons | VortexEnergyCons |
| Tracer fluxes | ThermoFluxConsistent, ThermoFluxDowngradient | Consistent |
| AN forcing | ThermodynamicForcing, ConsistentBuoyancy, LegacyBuoyancy | ThermodynamicForcing |
| Time scheme | RungeKutta4, KinnmarkGray (5 stages, order 2 or 3) | RungeKutta4 |
| Float type | Float64 (designed for); Float32 and BigFloat untested | Float64 |

*References:* the anelastic system and its energy budget, [Ogura & Phillips 1962](https://doi.org/10.1175/1520-0469%281962%29019%3C0173:SAODAS%3E2.0.CO;2) and [Bannon 1996, anelastic](https://doi.org/10.1175/1520-0469%281996%29053%3C3618:OTAAFA%3E2.0.CO;2); the consistent formulation shared by both models, [Tailleux, Dubos & Hatton 2026](https://doi.org/10.48550/arXiv.2512.18461).

### T2 · Choosing diagnostics

- **Goal:** list, choose, add and sample diagnostics.
- **Steps:** `diagnostic_info(model)` gives names, units and grid locations; `diagnostics(model; reference)` adds the energy budget to the model's own diagnostics; a new diagnostic is an `entry(long_name, units, location, recipe)` merged into the book. The available-energy diagnostics are left out: they come in only when `reference_state` is passed.
- **Worked additions:** the gradient Richardson number on w-faces (I1, I3); the amplitude of horizontal Fourier mode k of w, for growth rates (I1, I2, I4, I5); the realised Courant number max|u|Δt/Δx (S1).
- **Checks:** `integrated_K` equals Σ ρK ΔxΔz computed from the state; each location matches its coordinates (already tested); a new entry allocates nothing after its first call.
- **Sampling:** outputs happen only at t = 0 and slice ends. FC2D exchanges K and P at acoustic frequencies (16–32 Hz in the lock exchange), so fields every 0.5 s alias them; integrals need `IntegralSeries` at ≤ 1/10 of the shortest acoustic period. `slice_size` also sets how often Δt is recomputed (S1).
- **Pertinence:** medium for the science, high for anyone else using the code.

*References:* energy-budget terms for a compressible binary fluid, [Tailleux 2009](https://doi.org/10.1017/S002211200999111X); entropy production as a diagnostic, [Gassmann & Herzog 2015](https://doi.org/10.1002/qj.2404).

### T3 · Saving, reading, restarting, plotting

- **Already there:** examples/recording writes NetCDF and JLD2 with `DataFile`, reads both back with `read_diagnostic`, and records videos.
- **Add:** (1) open the NetCDF in Python (xarray) to check that staggered locations and halos are understandable outside Julia; (2) pair a fine `IntegralSeries` with `Thinned` field files; (3) a restart: save (m, u, w, t), reload, continue, and compare with an uninterrupted run; (4) figures of |budget residual|/scale on a log axis, Hovmöller diagrams of horizontal means, and spectra, drawn with CairoMakie from the repo's own dependencies.
- **Expected:** the restarted run equals the continuous one bitwise on a serial manager with `FFTW.ESTIMATE`; `FFTW.MEASURE` may change round-off between runs (documented in the Poisson solver).
- **Pertinence:** I found no restart in the repo; FC2D runs are long because of the acoustic Δt, so it will be needed.

*References:* Literate.jl tutorials in a Julia fluid model, [Ramadhan et al. 2020](https://doi.org/10.21105/joss.02018).

### T4 · FC rest state as AN reference profile

- **Routes:** (a) profiles θ(z), q(z) → `CFCompressible.balanced_density(fc; consvar, q, pref)` → interior column → `AnelasticReference(domain, space, ρ_column; pref)`, as thermohaline\_lock\_exchange/models.jl does; (b) a relaxed FC state: perturb (a), run FC2D with large viscosity until K < tolerance, then average ρ over x; (c) the Lorenz reference state P\_\*(z) of an FC state, so that AN2D is posed about the minimum-energy state.
- **Checks:** (1) the FC rest state stays at rest to round-off over 10³ steps; (2) the background buoyancy b = g(ρ\_R v(P\_R, θ, q) − 1) of AN2D vanishes; (3) P\_FC(z) − P\_R(z) per level; (4) 𝒰\_FC and 𝒰\_AN of the same rest state agree, since 𝒰\_AN is 𝒰\_FC with (ρ, P) → (ρ\_R, P\_R).
- **Expected:** (1) round-off. (2)–(3) differ from zero at O(Δz²): FC2D balances the discrete thermodynamic forcing, while AN2D integrates ∂p\_R/∂z = −g ℑ\_z ρ\_R by the trapezoidal rule. Confirm with 16, 32 and 64 levels. This residual is horizontally uniform, so the AN pressure absorbs it and nothing moves.
- **Pertinence:** high. Every FC/AN comparison depends on it; a mismatch shows up as spurious initial ACE, or as FC–AN differences that do not shrink with c (A2). Route (c) ties the anelastic reference directly to your APE work.

*References:* reference states for anelastic models, [Bannon 1996, anelastic](https://doi.org/10.1175/1520-0469%281996%29053%3C3618:OTAAFA%3E2.0.CO;2); why the reference profile matters for energy conservation, [Brown, Vasil & Zweibel 2012](https://doi.org/10.1088/0004-637X/756/2/109).

## Numerical verification (N1–N7)

The diffusion, viscosity and wall operators can be checked against exact discrete eigenmodes to round-off; order of accuracy needs space and time refined separately.

On a uniform C-grid, a mode of wavenumber k is seen by the centred operators as

```latex
\hat k^2 = \frac{4}{\Delta^2}\sin^2\!\left(\frac{k\Delta}{2}\right) = k^2\left(1 - \frac{k^2\Delta^2}{12} + O(k^4\Delta^4)\right)
```

### N1 · Analytic diffusion of a sine profile, with viscous heating

- **Setup:** a column, `Box2D(Mx = 4, Mz = 32)` (Periodic, Bounded), L\_z = 0.5 m, insulating free-slip walls. Use g = 0 (or 10⁻⁶ if `Tank2D` needs g > 0) and `pref = p0`, so P = P0 everywhere, T = θ, h\_θ = c\_p = c\_p0, and the consistent and down-gradient laws coincide. AN2D with uniform ρ\_R = ρ0.
- **ICs and exact solutions:** (a) θ = T0 + A cos(kz), k = π/L\_z, decays as exp(−κ\_T k̂²t); (b) q = q0 + B cos(kz) decays as exp(−κ\_q k̂²t); (c) u = U cos(kz) decays as exp(−ν k̂²t) with ν = μ/ρ0, and its dissipation heats the fluid. With `Value` walls and `NoSlip`, use sin(kz) modes: the ghost-cell closures make them exact discrete eigenvectors too.
- **Heating in (c):** θ\_t = κ θ\_zz + ε/c\_p0 with ε = (νU²k²/2)(1 − cos 2kz) e^{−2νk²t}, so

```latex
\theta = \theta_0 + \frac{U^2}{4c_{p0}}\left(1-e^{-2\nu k^2 t}\right) - \frac{\nu U^2}{4c_{p0}}\,\frac{e^{-2\nu k^2 t}-e^{-4\kappa k^2 t}}{2\kappa-\nu}\,\cos 2kz
```

- **Expected:** decay rates equal the discrete ones to the RK4 error, and the continuum ones to O(Δz²). The mean rise U²/(4c\_p0) (6×10⁻⁷ K for U = 0.1 m s⁻¹) is exact, because energy is conserved; the cos 2kz amplitude matches at O(Δz²). FC2D adds a small expansion flow, w ≈ α\_Tκ∂\_zθ, so it agrees to O(α\_T A).
- **Tests:** tracer and momentum fluxes, wall closures, ρε, the return of dissipation as heat (σ\_ϑ), and closure of ∫(ρK + 𝒰).
- **Pertinence:** high. It is the cheapest way to pin every irreversible operator, and a natural test/ entry.

*References:* verification against exact solutions, [Roache 2002](https://doi.org/10.1115/1.1436090) and [Oberkampf & Roy 2010](https://doi.org/10.1017/CBO9780511760396).

### N2 · Convergence order

- **Caution:** halving Δx and Δt together gives E ≈ C\_xΔx² + C\_tΔt⁴, so the observed order tends to 2 and the 4th order in time is never seen. The code also sets Δt ∝ Δx (advective, acoustic) or ∝ Δx² (diffusive) unless `dt_max` binds.
- **Space:** fix Δt through `dt_max` so the temporal error is negligible; grids 32, 64, 128, 256; compare with an analytic solution (N1, W1, N4) or with the finest grid after conservative coarsening (2×2 cell means; pairs of faces for u, w). Expect p = 2 in L2, and possibly 1 near walls in L∞ (ghost-cell stresses, hydrostatic extrapolation of the wall pressure).
- **Time:** fix the grid; take Δt = 2⁻ⁿ via `dt_max` (it must divide `slice_size`); compare with Δt/16 on the same grid. Expect p = 4 for the solution and p = 5 for the energy error (N3).
- **Flows:** flows that stay smooth for the whole run and, where possible, have an exact solution: N1 (diffusion), W1 and W2 (buoyancy and acoustic coupling), a Gaussian vortex carried by a uniform current along a periodic channel (advection and the vortex force; exact for ν = 0 up to an exponentially small wall effect when its radius is ≪ L\_z), and N4 for every term at once. Avoid flows that roll up or go unstable: once they do, the measured error is set by the instability, not by the grid.
- **FC2D:** the acoustic Δt makes its temporal error invisible; to see p = 4, raise `cfl` towards 1 and measure acoustic phase (W2).
- **Pertinence:** high, with the caveats above. Combined space–time refinement is discussed by [Veluri, Roy & Luke 2012](https://doi.org/10.1016/j.compfluid.2012.04.028).

### N3 · Energy drift of RK4, and round-off

- **Theory:** for a skew-symmetric linear system, RK4 multiplies the energy of a mode of frequency ω by

```latex
|R(iy)|^2 = 1 - \frac{y^6}{72} + \frac{y^8}{576}, \qquad y = \omega\Delta t
```

so the relative energy lost over a time T is about Tω⁶Δt⁵/72: one order higher than the solution error.

- **Run:** inviscid, non-diffusive runs whose frequency content is known. (a) A superposition of a few W1 gravity modes (AN2D) or W2 acoustic modes (FC2D): each mode should lose Tω⁶Δt⁵/72 of its energy, so both the slope and the prefactor are predicted. (b) A smooth nonlinear flow, the N6 cellular flow with a weak θ perturbation, for a few turnover times. Record E(t) − E(0) − ∫(wall input) dt at several Δt and expect slope 5. Avoid shear layers and bubbles here: their cascade to the grid scale keeps raising the fastest resolved frequency, so the prefactor drifts during the run. In FC2D at `cfl = 0.9`, the fastest acoustic modes have y up to 0.9√8 ≈ 2.55, so RK4 multiplies their amplitude by about 0.53 per step: the time scheme filters the shortest sound waves, and that numerical sink appears in the energy budget.
- **Precision ladder:** the code is designed for Float64, and it is not yet known what works in Float32 or BigFloat. Finding out is a first job: try a short run in each, record what fails and where, and where a run works check that the semi-discrete residual scales with eps(F). One obstacle is known: AN2D's Poisson solver uses FFTW, which supports only Float32 and Float64.
- **Pertinence:** high. It turns "conserves energy" into a quantitative statement about each error source.

*References:* energy behaviour of non-symplectic Runge–Kutta methods, [Hairer, Lubich & Wanner 2006](https://doi.org/10.1007/3-540-30666-8).

### N4 · Manufactured solutions

- **How, without touching the packages:** a wrapper type in examples/ holding the model and a source S(x, z, t), whose `CFTimeSchemes.tendencies!` calls the model and adds S to the interior of `dstate`, forwarding `update!` and the CFL bound. For AN2D, take momentum sources as a discrete curl, ρ\_R S = curl χ, so they need no projection.
- **Fields:** θ = T0 + a sin(2πx/L\_x) cos(πz/L\_z) cos t, q likewise, and (u, w) from a streamfunction compatible with the walls; sources by Symbolics.jl.
- **Expected:** p = 2 in space for every field, including the vortex force, B, K and the barodiffusive flux.
- **Pertinence:** medium. It is the only test of all nonlinear terms at once ([Roy et al. 2004](https://doi.org/10.1002/fld.660)), but costs the wrapper and symbolic sources.

### N5 · Symmetry, reversibility, reproducibility

- **Mirror:** the single-mode Rayleigh–Taylor run of I5, centred at L\_x/2 in a closed box, stays mirror-symmetric; the asymmetry starts at round-off and grows no faster than the flow's own instabilities. RT is the usual choice because its secondary shear instabilities amplify any asymmetry within a few growth times, so an index error shows quickly.
- **Translation:** in periodic x, shifting the initial state by one cell shifts the solution by one cell, bitwise on a serial manager.
- **Transposition:** with g = 0 and walls all round, rotating the initial state by 90° rotates the solution; this catches direction-specific index errors in τ\_xz and the vortex force.
- **Time reversal:** with ν = κ = 0, run to T, flip (u, w) → (−u, −w), run T again. The centred semi-discrete system is reversible, so the initial state returns to within the RK4 error (∝ Δt⁴); an O(1) mismatch reveals hidden dissipation.
- **Reproducibility:** PlainCPU vs VectorizedCPU vs MultiThread should agree bitwise, with `FFTW.ESTIMATE` for AN2D.
- **Pertinence:** medium; cheap and good at finding index bugs.

*References:* time reversal as a benchmark for energy-conserving schemes, [Duponcheel, Orlandi & Winckelmans 2008](https://doi.org/10.1016/j.jcp.2008.06.020).

### N6 · Cellular (Taylor–Green) flow

- **Setup:** g = 0 or uniform θ and q, free-slip walls; ψ = ψ0 sin(kx) sin(mz), k = 2π/L\_x, m = π/L\_z.
- **Expected:** ω ∝ ψ, so the flow is a steady Euler solution and decays as K ∝ exp(−2ν(k̂² + m̂²)t). Unsteadiness of the inviscid run measures the truncation error of each vortex scheme; FC2D adds an O(Ma²) acoustic transient.
- **Pertinence:** medium; a clean test of the vortex force on its own.

*References:* [Taylor & Green 1937](https://doi.org/10.1098/rspa.1937.0036); the 2D cellular solution is exact for any ν.

### N7 · Couette flow with viscous heating

- **Setup:** g = 0, bottom `NoSlip` + `Value(T0)`, top `TangentialVelocity(U)` + `Value(T0)`, u(z, 0) = Uz/H + A sin(πz/H).
- **Expected:** the transient decays as exp(−νm̂²t), an exact discrete eigenvector. The steady state has θ = T0 + (νU²/2κc\_p0)(z/H)(1 − z/H), and the work of the moving wall, μU²/H per unit area, leaves exactly as heat through the two walls. With ν = κ and U = 1 m s⁻¹ the peak is about 3×10⁻⁵ K.
- **Pertinence:** medium. It is the only steady check of dE/dt = wall input; the existing test checks it at one instant.

*References:* Couette flow with dissipation heating, [Schlichting & Gersten 2017](https://doi.org/10.1007/978-3-662-52919-5).

## Linear waves (W1–W4)

Wave frequencies should be compared with the discrete dispersion relation, best obtained from the model's own Jacobian (W4); at the default parameters the FC–AN difference in gravity-wave frequency is only \~10⁻⁷.

### W1 · Internal gravity wave in both models

- **Setup:** (Periodic, Bounded), free-slip walls, κ and ν small. Uniform N² = gα\_T ∂\_zθ, e.g. N² = 10⁻² s⁻², i.e. ∂\_zθ = N²/(gα\_T) = 5 K m⁻¹. Set α\_TT = γ = 0 for this test, or cabbeling makes N² vary by \~10 % across the box. Standing mode: θ′ = Θ cos(kx) sin(mz), u = w = 0, k = 2π/L\_x, m = π/L\_z; FC2D from `balanced_density`.
- **Expected (continuum):** ω² = N²k²/(k² + m²) for AN2D. For FC2D with uniform c, the gravity branch of

```latex
\omega^4 - \omega^2\left[N^2 + c^2\left(k^2+m^2\right)\right] + c^2N^2k^2 = 0
```

differs from it by a relative N²m²/(c²K⁴) ≈ 10⁻⁷ here, plus density-scale-height effects of order (gL\_z/c²)².

- **Expected (discrete):** k → k̂, m → m̂, and an averaging factor from the vertical staggering. Buoyancy lives at cell centres and w on z-faces, a Lorenz-type placement, so the leading guess is ω² = N² cos²(mΔz/2) k̂²/(k̂² + m̂²), with a zero-frequency computational mode at mΔz = π. The exact factor depends on how `ThermodynamicForcing` averages ϑ∇h\_ϑ to faces: confirm it with W4 rather than trusting this formula ([Thuburn & Woollings 2005](https://doi.org/10.1016/j.jcp.2004.08.018); for general equations of state, [Thuburn](https://doi.org/10.1002/qj.2971)).
- **Measure:** project w onto cos(kx) sin(mz) every \~T/20 and fit frequency and decay over ≥ 20 periods. Repeat for m = π/L\_z … π/Δz.
- **Pertinence:** high. It is the first test of the buoyancy coupling, and the 2Δz computational mode matters for mixing studies. To see a real FC–AN difference, reduce c (W4 or A2), and compare against the anelastic analyses of [Brown, Vasil & Zweibel 2012](https://doi.org/10.1088/0004-637X/756/2/109) and [Davies et al. 2003](https://doi.org/10.1256/qj.02.1951).

### W2 · Acoustic mode and the sound speed (FC2D)

- **Setup:** closed free-slip box, uniform θ and q (g = 0, or keep g: corrections are O(gL/c²)). IC: ρ′ = ερ0 cos(πx/L\_x) cos(πz/L\_z) at fixed θ and q, u = w = 0.
- **Expected:** ω = c(k̂² + m̂²)^{1/2} with c² from ClimFluids' `sound_speed2`, ≈ v0/α\_p = 10³ m² s⁻². Mode (1,1) in the 1 × 0.5 m box: ω ≈ 222 rad s⁻¹ (35 Hz). On 64×32 the k̂ error is ≈ 4×10⁻⁴ and the RK4 phase error (≈ y⁴/120) ≈ 10⁻⁶.
- **Damping:** the viscous amplitude decay is (2/3)νK² ≈ 3×10⁻⁵ s⁻¹ for μ = 10⁻³ Pa s; the RK4 energy loss at `cfl = 0.9` is ≈ 3×10⁻⁵ s⁻¹, the same order. Measure both by halving Δt.
- **Measure:** a least-squares (Prony) fit to a point pressure or to the mode projection, sampled at \~1 ms over ≥ 100 periods.
- **Pertinence:** high. It checks that the model's pressure P(v, ϑ, q) and ClimFluids' `sound_speed2` are mutually consistent; they are separate code paths.

*References:* discrete acoustic dispersion on staggered grids, [Thuburn & Woollings 2005](https://doi.org/10.1016/j.jcp.2004.08.018); acoustic modes in compressible and sound-proof sets, [Davies et al. 2003](https://doi.org/10.1256/qj.02.1951).

### W3 · Lamb wave (FC2D only)

- **Setup:** periodic x, rigid lids, neutral; p′ = P cos(kx) e^{−gz/c²}, w = 0, set through ρ′ = p′/c² at fixed θ and q.
- **Expected:** w stays at truncation level, and ω = ck̂ for every k (non-dispersive). In AN2D the corresponding velocity is purely divergent, so the projection removes it entirely: AN2D has no Lamb mode.
- **Pertinence:** medium. It is the cleanest filtered mode, and it tests whether FC2D's discrete hydrostatic balance generates spurious w under horizontal acoustics ([Fanelli & Bannon 2005](https://doi.org/10.1175/JAS3517.1) on Lamb waves and available energy).

### W4 · Discrete normal modes from the Jacobian

- **Method:** pick a base state s₀ (stratified rest, shear layer, conduction profile). Build J = ∂F/∂s by centred differences of `tendencies!`, one degree of freedom at a time (≈ 2,500 on 32×16), or matrix-free with an Arnoldi solver for larger grids. Use Float64 with a relative step of \~10⁻⁷; BigFloat would make the differences essentially exact, but whether the code runs in it is untested (N3). For AN2D, restrict it to divergence-free directions (a streamfunction basis), since the projection adds a null space.
- **Output:** the discrete dispersion relations (W1–W3), discrete growth rates (I1, I2, I4, I5), computational modes, and any spurious unstable mode of a stable base state.
- **Expected:** small-amplitude runs match the eigenvalues to the RK4 error; the eigenvalues approach linear theory at O(Δ²).
- **Already started:** examples/vortex\_schemes/output/linear.txt has such a study: FC2D eigenvalues are neutral to \~10⁻¹³, while AN2D shows growth rates \~10⁻⁵ labelled noise. Find their source (finite-difference step, or the projection) before relying on W4 for AN2D, then turn the study into a test.
- **Pertinence:** high. It makes W1, I1, I2, I4 and I5 quantitative to \~10⁻¹⁰ rather than \~10⁻³.

*References:* numerical normal-mode analysis of discretizations, [Thuburn & Woollings 2005](https://doi.org/10.1016/j.jcp.2004.08.018); for general equations of state, [Thuburn 2017](https://doi.org/10.1002/qj.2971).

## Instabilities (I1–I5)

Both requested instability tests are sound, but each needs a box sized to the right wavenumber and the right definition of its control parameter; as set up today, neither would land on the textbook threshold.

### I1 · Kelvin–Helmholtz growth rate and the Ri = 1/4 cut-off

- **Profile:** `kelvin_helmholtz` sets u = U tanh((z − L\_z/2)/δ) with U = Δu/2, and a θ step of the same thickness δ (Hazel's R = 1). The minimum gradient Richardson number is at the centre:

```latex
J = Ri_{\min} = \frac{2\,g\,\alpha_T\,\Delta\theta\,\delta}{\Delta u^2}
```

This is twice the "bulk Ri = gα\_TΔθδ/Δu²" quoted in examples/kelvin\_helmholtz\_budget/run.jl, so the 1/4 criterion applies to J, i.e. to a bulk value of 1/8.

- **Linear theory:** with J = 0 the fastest mode has α = kδ = 0.4446 and σ = 0.19 U/δ = 0.095 Δu/δ ([Michalke 1964, as reproduced in arXiv:2206.13205](https://doi.org/10.48550/arXiv.2206.13205)). For R = 1 the inviscid neutral curve is J = α(1 − α), so the last mode to stabilise is α = 1/2 at J = 1/4; for R ≤ √2 the minimum gradient Ri equals J, as Miles–Howard requires ([Alexakis 2005](https://doi.org/10.48550/arXiv.physics/0503241); [Miles 1961](https://doi.org/10.1017/S0022112061000305)).
- **A slip in the existing comment:** kelvin\_helmholtz\_budget/run.jl expects "about 0.18 Δu/δ ≈ 0.4 s⁻¹". Michalke's value is 0.095 Δu/δ ≈ 0.19 s⁻¹ for Δu = 0.1 m s⁻¹ and δ = 0.05 m, and the seeded mode (α = 2πδ/L\_x ≈ 0.31) grows more slowly still.
- **Design for "optimal instability":** L\_x = 1 m with kx = 3; δ = 0.4446/(6π) ≈ 0.0236 m for the J = 0 optimum, or δ = 1/(12π) ≈ 0.0265 m (α = 1/2) for the J sweep. Keep Δu = 0.1 m s⁻¹ (Re = Uδ/ν ≈ 1300 with μ = 10⁻³ Pa s) and raise α\_T to 2×10⁻³ K⁻¹ with α\_TT = γ = 0; then J = 0.106 Δθ, and J = 1/4 needs Δθ ≈ 2.4 K. Set k\_T = k\_q ≈ 10⁻⁶ m² s⁻¹: the examples' k\_T = 10⁻⁴ gives Pr = 0.01, and the density layer would spread faster than the billow grows. Resolve δ with ≥ 8 cells, check walls by doubling L\_z, sweep J = 0, 0.05 … 0.3.
- **Measure:** the amplitude of mode kx of w during the linear phase; or seed the eigenfunction from W4 to skip the transient.
- **Expected:** AN2D and FC2D agree to O(Ma²) and with a viscous, bounded Taylor–Goldstein solve ([Smyth & Carpenter 2019](https://doi.org/10.1017/9781108640084), open-access with code) to discretisation error. σ → 0 as J → 1/4 at α = 1/2; small viscosity and diffusion shift the marginal J slightly and can even destabilise ([Thorpe 2013](https://doi.org/10.1017/jfm.2013.378)). Compute J from the model's own N² and S² on faces, not from the formula.
- **Pertinence:** high; it validates the shear–buoyancy coupling and is the base flow for F3.

### I2 · Rayleigh–Bénard onset

- **Thresholds:** rigid walls Ra\_c = 1707.76 at kH = 3.117 (λ ≈ 2.016H); free-slip walls Ra\_c = 27π⁴/4 ≈ 657.5 at kH = π/√2 (λ = 2√2H); rigid–free 1100.65 at kH = 2.682 (Chandrasekhar 1961; [arXiv:nlin/0302057](https://doi.org/10.48550/arXiv.nlin/0302057)).
- **The box picks the mode:** with H = L\_z = 0.5 m, use L\_x = 1.008 m (rigid) or 1.414 m (free-slip). In the default L\_x = 1 m box the free-slip mode has kH = π, whose threshold is 8π⁴ ≈ 779, not 657.5.
- **Free-slip growth rate in closed form** (time in H²/κ, K² = k² + π²):

```latex
(s + K^2)\,(s + Pr\,K^2)\,K^2 = Ra\,Pr\,k^2
```

The discrete version replaces k², π² by k̂², m̂² and adds the buoyancy-averaging factor of W1; check it with W4.

- **Parameters:** ν = κ = 10⁻³ m² s⁻¹ (Pr = 1), so Ra = 1708 needs ΔT ≈ 0.68 K and the diffusive time is 250 s; α\_TT = γ = 0, so the conduction state is linear. `rayleigh_benard` with `Value` walls, Ra/Ra\_c = 0.9, 0.95, 1.05, 1.1.
- **Measure:** σ from the linear phase of mode k, then Ra\_c where σ = 0 by interpolation (as in arXiv:nlin/0302057). Expect the discrete Ra\_c within O(Δz²), a few tenths of a percent with 32 levels.
- **Thermodynamics:** with consistent fluxes the conduction state is linear in T, and Ra should use the superadiabatic difference ΔT − α\_T g T H/c\_p; here the correction is \~10⁻⁴, but not with a magnified fluid ([Alboussière & Ricard 2017](https://doi.org/10.1017/jfm.2017.108); FC/anelastic comparisons in [Curbelo et al. 2019](https://doi.org/10.1017/jfm.2019.420)).
- **Cost:** FC2D at onset needs \~10⁶ acoustic steps; use W4 for FC thresholds and keep nonlinear FC runs for supercritical Ra.
- **Pertinence:** high. The free-slip case offers an exact growth rate at any Ra, not only the threshold.

### I3 · Holmboe instability (guarding I1)

- **Setup:** as I1, but with the density interface thinner than the shear layer, R = δ\_u/δ\_ρ ≥ 2 (a custom setup; `kelvin_helmholtz` ties both to δ).
- **Expected:** instability for J > 1/4, because the gradient Ri dips below 1/4 off-centre once R > √2; two counter-propagating cusped waves ([Alexakis 2005](https://doi.org/10.48550/arXiv.physics/0503241)).
- **Pertinence:** low to medium. It shows that the 1/4 cut-off concerns the minimum gradient Ri, and it stresses centred advection of a sharp interface.

### I4 · Salt fingers (double diffusion)

- **Setup:** periodic x, `Value` walls holding uniform gradients, warm salty water over cold fresh, density ratio R\_ρ = α\_Tθ\_z/(α\_q q\_z) between 1 and κ\_T/κ\_q ([regimes](https://doi.org/10.48550/arXiv.2605.00335)). Linear theory for vertical fingers (K = k), with N\_T² = gα\_Tθ\_z and N\_q² = gα\_q q\_z:

```latex
(\sigma+\nu K^2)(\sigma+\kappa_T K^2)(\sigma+\kappa_q K^2)K^2 + k^2\left[N_T^2(\sigma+\kappa_q K^2) - N_q^2(\sigma+\kappa_T K^2)\right] = 0
```

- **Parameters:** the examples' diffusivities give Pr = 0.01, far from water; choose Pr = 7 with τ = κ\_q/κ\_T ≈ 0.1–0.3 so the salt field is resolvable in 2D. The fastest wavenumber scales as \[gα\_q q\_z(R\_ρ − 1)/(νκ\_T)\]^{1/4} ([Kunze 2003](https://doi.org/10.1016/S0079-6611%2803%2900027-2); Radko 2013).
- **Expected:** growth rate and wavenumber from the cubic (or W4) at the discretisation error. During growth the APE dissipation is negative: double diffusion converts background into available energy ([Tailleux 2024](https://doi.org/10.1017/jfm.2024.647); [Middleton & Taylor 2020](https://doi.org/10.1017/jfm.2020.259)), a direct test of your reference-state budgets.
- **Pertinence:** medium to high. It is the only test in which k\_T ≠ k\_q drives the dynamics.

### I5 · Rayleigh–Taylor, single mode

- **Setup:** heavy (salty: α\_qΔq = 0.04 gives A = 0.02) over light, tanh interface of thickness δ, perturbation a₀cos(kx) with ka₀ ≪ 1, free-slip or no-slip box.
- **Expected:** σ² = Agk for a sharp, inviscid interface, reduced by a diffuse interface (Duff, Harlow & Hirt 1962, as used by [Morgan et al. 2016](https://doi.org/10.1017/jfm.2016.46)) and by viscosity; then a terminal bubble velocity U\_b = \[2Ag/(3(1 + A)k)\]^{1/2} in 2D ([Goncharov 2002](https://doi.org/10.1103/PhysRevLett.88.134502); [Bian et al. 2020](https://doi.org/10.1016/j.physd.2019.132250)), with re-acceleration later at high Reynolds number.
- **Tests:** the fastest P → K conversion of the suite, Δt adapting from rest, mirror symmetry, mixing.
- **Pertinence:** medium; also the core of S1.

## Acoustic filtering (A1–A3, with W2 and W3)

The cleanest demonstration of filtering is a horizontally uniform heating: FC2D rings with sound whose energy is exactly the initial ACE, while AN2D does not move at all.

### A1 · Lamb's hydrostatic adjustment

- **Setup:** a column (`Mx = 4`, periodic) or the full box, at rest on a stratified background (T4). At t = 0 a layer z₁ < z < z₂ is warmed by Δθ at constant density, setting (ρ, ρθ, ρq) = (ρ\_b, ρ\_bθ\_new, ρ\_bq) as the lock exchange does.
- **AN2D:** the buoyancy is horizontally uniform, so the pressure absorbs it: u ≡ 0, ACE ≡ 0, APE ≈ 0.
- **FC2D:** the layer's pressure exceeds hydrostatic by Δp ≈ α\_TΔθ/α\_p (≈ 200 Pa for 1 K). Acoustic fronts bounce between the lids at ω\_n = cnπ/L\_z (n = 1: ≈ 199 rad s⁻¹, 32 Hz). The time-mean state is hydrostatically adjusted: the layer expands, the fluid above it rises uniformly, the fluid below is unchanged ([Bannon 1995](https://doi.org/10.1175/1520-0469%281995%29052%3C1743:HALP%3E2.0.CO;2); finite amplitude: [Bannon 1996](https://doi.org/10.1175/1520-0469%281996%29053%3C3606:NHA%3E2.0.CO;2)).
- **Energetics:** for a horizontally uniform, stable column, the adjusted state is the Lorenz reference state of the initial state (same parcels, same order, hydrostatic). So the initial ACE, ≈ α\_pΔp²/2 per unit volume (≈ 0.02 J m⁻³ for 1 K), is exactly the energy given to sound; APE stays ≈ 0. The sound then decays by bulk viscous dissipation (4μ/3)∇D and by RK4 damping, both \~10⁻⁵ s⁻¹ at the default parameters; raise μ to see the adjustment complete.
- **Pertinence:** high. It is one-dimensional and cheap, it shows filtering with no ambiguity, and it checks the ACE diagnostic against an analytic value.

### A2 · Low-Mach convergence of FC2D to AN2D

- **Setup:** FC2D with α\_p = 10⁻⁶, 2.5×10⁻⁷, 6.25×10⁻⁸ Pa⁻¹ (c ≈ 31.6, 63.2, 126.5 m s⁻¹) and AN2D about the matching reference state from T4, on flows that stay smooth, so that the FC–AN difference is not swamped by instability growth. Three stages: (1) the W1 standing gravity mode, whose FC–AN frequency difference is known in closed form; (2) the B6 gravity-wave packet, which spans many wavenumbers and has a Boussinesq analytic solution; (3) a steady Rayleigh–Bénard roll at Ra ≈ 3Ra\_c (I2), nonlinear but attracted to a steady state, compared through Nu, the peak w and the dissipations. All three start from balanced data (A3, case iii).
- **Expected:** with well-prepared (balanced) initial data, FC–AN differences in θ and u at fixed time fall as Ma² ∝ 1/c² ([Klainerman & Majda 1981](https://doi.org/10.1002/cpa.3160340405)); ill-prepared data add an O(Ma) fast acoustic part. In this box 1/c² also sets the density variation gL\_z/c², which AN2D keeps through ρ\_R(z), so both effects should fall together: slope −2 on a log–log plot against c. The pressure-dilatation conversion C\_D and the ACE should vanish at the same rate.
- **Cost:** doubles with each doubling of c.
- **Pertinence:** high. It is the dynamical form of "AN2D is traceable to FC2D" ([Klein et al. 2010](https://doi.org/10.1175/2010JAS3490.1) on the validity range of soundproof models).

### A3 · Balanced vs unbalanced initial states

- **Setup:** a localised Gaussian warm anomaly (radius ≈ L\_z/8) in uniform stratification, N² ≈ 10⁻² s⁻², in FC2D initialised three ways: (i) `balanced_density` per column (the default); (ii) at the background density, heated at constant volume; (iii) at the background pressure, heated isobarically. AN2D receives the same θ and q.
- **Expected:** (iii) emits almost no sound and has no ACE; (ii) has ACE ≈ α\_pΔp²/2 in the bubble and radiates it; (i) radiates from the horizontal pressure gradients at the bubble's edge. After the acoustic transient, all three settle into the same gravity-wave response and adjusted state as AN2D, to O(Ma²). This is the localised-heating problem of [Chagnon & Bannon 2005](https://doi.org/10.1175/JAS3503.1); unlike a rising bubble it stays smooth, so the comparison can run long after the sound has gone.
- **Measure:** ACE(0), the high-pass part of K, and the gravity-wave field and the adjusted θ profile against AN2D.
- **Pertinence:** medium. It tells you how to initialise FC2D for comparisons, and confirms that the initial ACE predicts the radiated acoustic energy.

W2 (acoustic standing mode) and W3 (Lamb wave) complete this group: both exist only in FC2D, and AN2D removes the Lamb wave's velocity by projection.

*References:* adjustment of a compressible atmosphere to imbalance, [Bannon 1995](https://doi.org/10.1175/1520-0469%281995%29052%3C1743:HALP%3E2.0.CO;2) and [Chagnon & Bannon 2005](https://doi.org/10.1175/JAS3503.1).

## Consistent vs down-gradient fluxes (F1–F4)

The two flux laws differ in three ways, and only one of them fades in the turbulent limit; F3 tests which. The down-gradient law also depends on the choice of entropic variable, but here it is only run with θ.

| Property | `ThermoFluxConsistent` | `ThermoFluxDowngradient` |
| --- | --- | --- |
| Heat flux | −ρc\_pκ\_T∇T, so J\_ϑ = J′*Q/h\_ϑ + (∂ϑ/∂q)*{P,T} J\_q | J\_ϑ = −ρκ\_T∇ϑ |
| Salt flux | −ρκ\_q(∇q − Γ∇P), with barodiffusion | −ρκ\_q∇q |
| Source of ρϑ | h\_ϑσ\_ϑ = ρε − J\_ϑ·∇h\_ϑ − J\_q·∇h\_q | none |
| Total energy | conserved | not conserved |
| Entropy production | ≥ 0 in the continuum; not guaranteed discretely | no guarantee |
| Depends on the choice of ϑ | no | yes |
| Equilibrium at rest | uniform T, ∇q = Γ∇P | uniform ϑ and q |

### F1 · Diffusion to thermodynamic equilibrium

- **Setup:** a column (`Mx = 4`, periodic), L\_z = 0.5 m, insulating and salt-tight walls, g = 10 m s⁻². Stable IC: θ = T0 + A(1 − cos πz/L\_z)/2 and q = q0 − Q(1 − cos πz/L\_z)/2, with A = 0.1 K and Q = 10⁻⁴. Both models, both flux laws. Take κ\_T = κ\_q = 10⁻³ m² s⁻¹ so FC2D reaches equilibrium (≈ 5L\_z²/π²κ ≈ 130 s) in \~3×10⁵ steps; the equilibrium does not depend on κ.
- **Expected, consistent law:** isothermal T, so θ increases upward at the adiabatic rate α\_T gT/c\_p0 ≈ 1.4×10⁻⁴ K m⁻¹. An isothermal liquid is stably stratified, with N² = g²α\_T²T/c\_p0 ≈ 2.8×10⁻⁷ s⁻². The salt reaches sedimentation equilibrium, d ln q/dz = −α\_q g/(R\_sT), i.e. a scale height of ≈ 9 km and Δq/q ≈ −5×10⁻⁵ over the box. The levels follow from conservation of E and ∫ρq. Total entropy should rise monotonically; check the sign of dS/dt at every output, since the discrete scheme does not guarantee it.
- **Expected, down-gradient law:** uniform θ and q, so T decreases upward by 7×10⁻⁵ K, which the real fluid would keep conducting away. E drifts by ∫(J\_ϑ·∇h\_ϑ + J\_q·∇h\_q) dV.
- **Discrete check:** at equilibrium the fluxes vanish to round-off, e.g. δ\_zq = Γ\_face δ\_zP for the consistent law.
- **Magnified fluid:** α\_T = 2×10⁻³ K⁻¹ and c\_p0 = 4 J kg⁻¹ K⁻¹ (with α\_TT = γ = 0) give ≈ 0.7 K of adiabatic θ difference over the box, since that difference is α\_T gTL\_z/c\_p0; M\_s × 10³ (a heavy solute, R\_s ≈ 0.26 J kg⁻¹ K⁻¹) gives Δq/q ≈ 5 %. Both make the two laws differ at O(1).
- **Pertinence:** high. It is the only test whose answer is set by thermodynamics alone (barodiffusion: TEOS-10 manual, appendix B).

### F2 · Invariance under the choice of entropic variable

- **Setup:** g = 0 and `pref = p0`, uniform T = T0, a salt step q = q0 + (Δq/2) tanh((z − L\_z/2)/δ) with Δq = 10⁻³, κ\_T = 10κ\_q. Three runs: the consistent law with consvar = θ and with consvar = s (`NonlinearBinaryFluid(:entropy, …)`), and the down-gradient law with consvar = θ as a reference.
- **Expected:** the two consistent runs agree to discretisation error, and T stays uniform (for this fluid the heat of mixing vanishes at P0). So the consistent law gives the same physical answer whichever entropic variable it carries. The down-gradient run with θ matches them too, because ∂θ/∂q|\_{P,T} = 0 here, so this setup does not separate the two laws; F1 and F3 do.
- **With gravity:** repeat with g = 10 m s⁻² and pref from the rest state. The two consistent runs should still agree, including the barodiffusive salt flux, which checks that both entropic variables carry the same Γ∇P term.
- **Pertinence:** medium. It checks that the consistent law is invariant under the choice of entropic variable, as it must be ([Gassmann & Herzog 2015](https://doi.org/10.1002/qj.2404) on entropy production by subgrid heat fluxes; [Dubos 2026](https://doi.org/10.48550/arXiv.2601.16957) on thermodynamically invariant heat fluxes).

### F3 · Laminar vs turbulent: when does the flux law matter?

- **Argument:** with consvar = θ, two differences are proportional to κ times a background gradient (the adiabatic term, and barodiffusion). The turbulent flux, by Osborn–Cox, is κ⟨|∇θ′|²⟩/θ̄\_z, so their relative weight falls like 1/Cox: **your hypothesis holds for these terms**. One other does not fade. The missing heating h\_ϑσ\_ϑ is set by dissipation rates, which stay finite as κ, ν → 0, so the down-gradient energy residual stays O(1) relative to the dissipation.
- **2D caveat:** without a forward energy cascade, viscous dissipation tends to zero as Re → ∞ in 2D (Kraichnan & Montgomery 1980), but scalar variance still cascades forward. So the ρε part of the residual should shrink with Re, and the scalar parts should not.
- **Experiment:** Rayleigh–Bénard convection swept from a steady roll at Ra = 2×10³ (laminar) to Ra = 10⁸ (turbulent), Pr = 1, both laws with consvar = θ, with the magnified fluid of F1. One configuration covers both limits, and it reaches statistically steady states, so the two laws are compared through time means rather than through two chaotic trajectories that would diverge anyway. A decaying KH layer at J ≈ 0.1, the canonical stratified-mixing flow, can be added for mixing efficiency, but only as an ensemble over seeds and through integrated quantities. Measure the difference of the mean θ profiles relative to ΔT, Nu, the time-mean dissipations, and mixing from the Lorenz reference state ([Winters et al. 1995](https://doi.org/10.1017/S002211209500125X)), the energy residual of the down-gradient runs divided by the dissipations, and the Cox number.
- **Expected:** with θ, profile and mixing differences fall as 1/Cox; the normalised residual does not.
- **Cost:** each Ra must be resolved, about 1024×512 at Ra = 10⁸ with Pr = 1, so run the sweep in AN2D and FC2D at one Ra only.
- **Pertinence:** medium. The outcome refines the hypothesis rather than confirming it, and ties directly to the diabatic terms of your GAFD paper ([Tailleux, Dubos & Hatton](https://doi.org/10.48550/arXiv.2512.18461); [Tailleux 2009](https://doi.org/10.1017/S002211200999111X) on internal-energy–BPE conversions; [Tailleux 2010](https://doi.org/10.48550/arXiv.1011.5071) on the non-conservation of θ with a ∇T heat flux; [Graham & McDougall 2013](https://doi.org/10.1175/JPO-D-11-0188.1)).

### F4 · Buoyancy schemes in AN2D

- **Setup:** I5 or the lock exchange, with density contrasts ρ′/ρ from 10⁻³ to 5×10⁻² (through α\_qΔq), run with `ThermodynamicForcing`, `ConsistentBuoyancy` and `LegacyBuoyancy`.
- **Expected:** the first two conserve energy to round-off. Legacy uses b = g(1 − 1/ρ\_Rv) instead of g(ρ\_Rv − 1); with x = ρ\_Rv − 1 the two differ by gx²/(1 + x), so its energy residual is a fraction ≈ ρ′/ρ of the conversion. The two consistent forms differ only at truncation level (the thermodynamic form has a small horizontal component).
- **Pertinence:** medium. It puts a number on the case made by [Tailleux & Dubos 2024](https://doi.org/10.1016/j.ocemod.2024.102339) inside your own code.

## Equation of state and available energy (E1–E4)

These four exercise the nonlinear parts of the equation of state and the available-energy diagnostics built on it. The fluid's specific volume is v = v0\[1 + α\_T(1 + γ(P − P0))(ϑ − T0) − α\_q(q − q0) − α\_P(P − P0) + ½α\_TT(ϑ − T0)²\], so cabbeling comes from α\_TT and thermobaricity from γ.

### E1 · Cabbeling of a compensated front

- **Setup:** a closed box, no motion, γ = 0, κ\_T = κ\_q = κ (equal, so differential diffusion makes no density anomaly). Two waters side by side, separated at x = L\_x/2 by a sharp tanh front: θ = T0 ± Δθ/2 and q = q0 ± Δq/2 with α\_qΔq = α\_TΔθ, which makes them exactly equal in density at every depth even with α\_TT > 0. Take Δθ = 4 K, Δq = 10⁻³ and a magnified α\_TT = 10⁻⁴ K⁻².
- **Expected, diffusive phase:** before any motion, θ − T0 = (Δθ/2) erf(ξ) with ξ = (x − L\_x/2)/(2√(κt)), and the mixed water is denser by δρ/ρ0 = (α\_TTΔθ²/8)(1 − erf²ξ), up to 2×10⁻⁴ here. Its integral grows as √(κt), with ∫(1 − erf²ξ) dξ = 2√(2/π).
- **Expected, then:** the dense water slumps into two gravity currents along the floor (g′ = gδρ/ρ0). The APE they carry has no adiabatic source, so the budget must attribute all of it to the diabatic term. With α\_TT = 0 the box stays at rest to round-off.
- **Spurious cabbeling:** periodic x, κ\_T = κ\_q = 0, the two fronts carried by a uniform current U. The exact solution is a translation, so any dense water formed is numerical mixing of θ and q at the fronts (Ilicak et al. 2012 find this in real ocean models).
- **Tests:** the cabbeling part of the energy and APE budgets, the consistency of mixing with a nonlinear EOS, and the numerical mixing of the tracer scheme.
- **Pertinence:** high. It is the only experiment where mixing alone creates APE.

*References:* [McDougall 1987](https://doi.org/10.1029/JC092iC05p05448); [Ilicak et al. 2012](https://doi.org/10.1016/j.ocemod.2011.10.003).

### E2 · Analytic checks of the APE diagnostics

- **Idea:** three flows whose available energy is known in closed form, so the reference-state and APE code is tested from outside its own unit tests.
- **(a) Lock:** AN2D, α\_TT = γ = 0, two waters differing in density by Δρ side by side over the full depth H, with a front of width δ. Expected APE = Δρ g H² L\_x/8 per unit width, minus an O(δ/L\_x) correction; during the B1 run, K + APE + the dissipated energy stays constant.
- **(b) Standing gravity wave:** the W1 mode θ′ = Θ cos(kx) sin(mz) over a uniform gradient θ\_z. The APE is ρ0N²⟨ζ²⟩/2 with ζ = θ′/θ\_z, i.e. ρ0 g α\_T Θ² L\_xL\_z/(8θ\_z) per unit width, and K and APE have equal time means.
- **(c) Acoustic energy:** the A1 heated layer in FC2D. The initial ACE is α\_PΔp²/2 per unit volume, with Δp ≈ α\_TΔθ/α\_P, i.e. α\_PΔp²(z₂ − z₁)L\_x/2 per unit width.
- **Expected:** each to the discretisation error of the reference state; (a) also checks the sorting, since the Lorenz state is exact for two waters.
- **Pertinence:** high. Every energetics result rests on these diagnostics.

*References:* [Winters et al. 1995](https://doi.org/10.1017/S002211209500125X); [Bannon 1995](https://doi.org/10.1175/1520-0469%281995%29052%3C1743:HALP%3E2.0.CO;2).

### E3 · Deep regime

- **Setup:** lower c until gL\_z/c² ≈ 0.3, i.e. α\_P = 6×10⁻⁵ Pa⁻¹ and c ≈ 4.1 m s⁻¹, so ρ\_R(z) varies by \~30 % across the box. Keep velocities below \~0.04 m s⁻¹ so Ma stays small. FC2D becomes \~8× cheaper than at the default c.
- **Runs:** (1) the W1 mode in AN2D and FC2D; (2) Rayleigh–Bénard onset with a magnified dissipation number α\_T gL\_z/c\_p0 (raise α\_T or lower c\_p0); (3) AN2D posed about each of the three T4 reference states.
- **Expected:** AN2D conserves the true total energy, not a stratification-weighted pseudo-energy, and its gravity-wave frequencies match FC2D's gravity branch, with the difference set by the stratification rather than by the depth. The critical superadiabatic Ra departs from 1707.76 / 657.51 as Alboussière & Ricard predict. The three reference states give different AN2D results, and route (c) of T4 should come closest to FC2D.
- **Pertinence:** medium. It shows where AN2D stays traceable to FC2D once the reference density is far from uniform.

*References:* [Brown, Vasil & Zweibel 2012](https://doi.org/10.1088/0004-637X/756/2/109); [Alboussière & Ricard 2017](https://doi.org/10.1017/jfm.2017.108); [Curbelo et al. 2019](https://doi.org/10.1017/jfm.2019.420).

### E4 · Thermobaric instability

- **Setup:** α\_TT = 0 and a magnified γ = 10⁻⁴ Pa⁻¹ (γΔP ≈ 0.5 over the box, against \~5×10⁻⁴ with the default). Cold, fresh water over warm, salty water, with an interface at mid-depth of thickness δ ≈ 0.01 m, Δθ = 2 K, and q chosen so the interface is stable with a small density step S = Δρ/ρ0 = 2×10⁻⁵. Displace the interface by a cos(kx), on 128 levels.
- **Theory:** a cold parcel pushed down by d gains a density excess α\_TγΔθρ0gd from thermobaricity, so the layer is stable to small displacements but unstable beyond

```latex
d_c = \frac{S}{\alpha_T\,\gamma\,\Delta\theta\,\rho_0\,g} \approx 0.05\ \text{m}
```

- **Expected:** for a < d\_c, interfacial waves and no overturning; for a > d\_c, the cold water plunges to the floor. The threshold matches d\_c to O(δ), and AN2D, which evaluates buoyancy at P\_R(z), agrees with FC2D to O(Ma²). With γ = 0 nothing plunges at any a.
- **Tests:** the pressure dependence of the EOS in the buoyancy, the ½γ term of h in the energy budget, and a Lorenz reference state whose sorting depends on pressure.
- **Pertinence:** medium. Thermobaricity is weak at this box depth, so it needs the magnified γ.

*References:* [McDougall 1987](https://doi.org/10.1029/JC092iC05p05448); [Akitomo 1999](https://doi.org/10.1029/1998JC900062).

## Standard 2D benchmarks (B1–B8)

The best literature benchmarks for this code are the ocean-model lock exchange, the cavity flows, the dipole–wall collision and horizontal convection; the classic atmospheric ones need an ideal-gas fluid the models cannot use yet (see Gaps).

### B1 · Lock exchange and spurious mixing

- **Setup:** a closed box with L\_x/L\_z ≈ 8, free-slip walls, unstratified, two waters differing by Δθ (g′ = gα\_TΔθ), Re = (g′H)^{1/2}H/ν ≈ 10³–10⁴. The repo's thermohaline lock exchange (stratified, 40 % compensated) stays as the energetics case; this is the classic one.
- **Expected:** front speed Fr = U/(g′H)^{1/2} = 1/2 for the energy-conserving current (Benjamin; [Shin, Dalziel & Linden 2004](https://doi.org/10.1017/S002211200400165X)), a little less at low Re or with no-slip walls.
- **Spurious mixing:** repeat with κ\_T = κ\_q = 0. Physical BPE change is then zero, so any change in the reference potential energy is numerical ([Ilicak et al. 2012](https://doi.org/10.1016/j.ocemod.2011.10.003), where it grows with the grid Reynolds number). Centred advection conserves scalar variance in AN2D but creates over- and undershoots, so the sorted distribution changes and BPE may even fall: a sharp test of what your reference-state diagnostics mean in a non-diffusive run.
- **Pertinence:** high. It is the ocean-modelling standard for exactly the diagnostics TraceableEnergetics computes.

### B2 · Lid-driven cavity

- **Setup:** a square (Bounded, Bounded) box, g = 0 or neutral, top `TangentialVelocity(U)`, other walls `NoSlip`, Re = UL/ν = 100, 400, 1000.
- **Expected:** centreline profiles from Ghia, Ghia & Shin (1982) and high-accuracy Re = 1000 data from [Botella & Peyret 1998](<https://doi.org/10.1016/s0045-7930(98)00002-4>). In the steady state the lid's work equals ∫ρε exactly.
- **Tests:** corner singularities (infinite vorticity where the lid meets the side walls), no-slip ghost closures, moving-wall work; local order below 2 near the corners.
- **Pertinence:** medium; cheap in AN2D, and a steady check of forced walls.

### B3 · Differentially heated cavity

- **Setup:** a square cavity, left `Value(T_h)`, right `Value(T_c)`, top and bottom `Flux(0)`, all no-slip, Pr = 0.71, Ra = gα\_TΔTL³/(νκ) = 10³–10⁶.
- **Expected:** mean Nusselt numbers of [de Vahl Davis (1983)](https://doi.org/10.1002/fld.1650030305): ≈ 1.118, 2.243, 4.519 and 8.80 for Ra = 10³–10⁶ (later benchmarks give 8.80–8.83 at 10⁶). In the steady state the heat entering through the hot wall leaves through the cold one.
- **Tests:** value walls, wall heat fluxes, thin boundary layers, steady energy balance.
- **Pertinence:** medium.

### B4 · Dipole–wall collision

- **Setup:** a square box with no-slip walls, g = 0, two shielded Gaussian monopoles forming a dipole aimed at a wall, Re = 625, 1250, 2500.
- **Expected:** E(t), enstrophy Ω(t) and the peak enstrophy at the first collision, tabulated by [Clercx & Bruneau 2006](https://doi.org/10.1016/j.compfluid.2004.11.009), who call it an extremely tough test because thin vorticity sheets form at the wall.
- **Tests:** vorticity generation at no-slip walls, enstrophy production, and the effect of the vortex scheme.
- **Pertinence:** medium. It is the best literature test of the no-slip walls, and a component of S1.

### B5 · Under-resolved shear layer in a channel

- **Setup:** (Periodic, Bounded) with free-slip lids, g = 0 or uniform θ and q, one shear layer u = U tanh((z − L\_z/2)/δ) with δ = L\_z/80, a perturbation w = 0.05U sin(2πx/L\_x), Re = Uδ/ν ≈ 10⁴, on 128×64 so that δ < Δz. This keeps the under-resolution question of the doubly periodic double shear layer ([Minion & Brown 1997](https://doi.org/10.1006/jcph.1997.5843)) in a domain closed in the vertical, so both models run it. Reference: the same run on 512×256.
- **Expected:** under-resolved schemes form spurious secondary vortices in the thinning braid, or blow up. EnergyCons, EnstrophyCons and EnergyEnstrophyCons should all stay bounded but differ in grid-scale noise and in when the spurious vortices appear; compare each with the resolved run through the vorticity field and the enstrophy history.
- **Pertinence:** low to medium. It isolates under-resolution of the vortex force, which B4 mixes with wall-generated vorticity.

### B6 · Skamarock–Klemp gravity-wave packet

- **Setup:** a periodic channel with free-slip lids, uniform N, uniform mean flow U, and θ′ = Δθ sin(πz/H)/(1 + (x − x\_c)²/a²). Keep the original ratios rather than the atmospheric sizes: NH/U = 5, a/H = 0.5, L/H = 30.
- **Expected:** the linear Boussinesq analytic solution of [Skamarock & Klemp (1994)](https://doi.org/10.1175/1520-0493%281994%29122%3C2623:EAAOTK%3E2.0.CO;2) (also used in the CM1 model's test suite); a compressible analytic solution exists for an ideal gas ([Baldauf & Brdar 2013](https://doi.org/10.1002/qj.2105)).
- **Tests:** a dispersive packet over many wavenumbers, plus advection by a mean flow.
- **Pertinence:** medium; it complements W1's single mode.

### B7 · Horizontal convection

- **Setup:** a closed box with a top temperature (or heat flux) varying in x, e.g. `Value(x -> T0 + ΔT cos(πx/L))`, other walls insulating; run to a statistically steady state.
- **Expected:** the dissipation obeys ⟨ε⟩ ≤ κgα\_TΔT/H, so it vanishes as κ → 0 at fixed Pr ([Paparella & Young 2002](https://doi.org/10.1017/S0022112002001313); bound restated in [Hazewinkel, Paparella & Young](https://doi.org/10.1017/jfm.2011.514)).
- **Tests:** non-uniform value or heat-flux walls, long equilibration, and APE generation by boundary buoyancy fluxes (the G term of your budget).
- **Pertinence:** medium to high for the APE work; [Tailleux 2009](https://doi.org/10.1017/S002211200999111X) discusses its energetics in a compressible framework.

### B8 · Rayleigh–Bénard Nu–Ra with exact identities

- **Setup:** rigid walls, Pr = 1 or 7, Ra = 10⁴–10⁸, aspect ratio 2, periodic in x, statistically steady.
- **Expected:** the Boussinesq identities of Shraiman & Siggia (1990), ⟨ε\_u⟩ = ν³(Nu − 1)Ra/(H⁴Pr²) and ⟨ε\_θ⟩ = κΔT²Nu/H², should hold at small dissipation number. They follow from the energy and θ² budgets, so a mismatch means under-resolution or a budget error.
- **Pertinence:** medium. It tests budget closure in a turbulent state.

*References:* exact dissipation–Nusselt identities, [Shraiman & Siggia 1990](https://doi.org/10.1103/PhysRevA.42.3650); Nu(Ra, Pr) scaling, [Grossmann & Lohse 2000](https://doi.org/10.1017/S0022112099007545).

## Stress test (S1): thermohaline Rayleigh–Taylor–Kelvin–Helmholtz in a forced box

One under-resolved run with every wall condition, a magnified nonlinear equation of state and fast acceleration from rest exercises all the pinch points at once; two of them (q → 0 and Δt fixed per slice) will probably fail as the code stands.

- **Domain:** (Bounded, Bounded), 1 × 0.5 m, 128×64 (64×32 for the precision ladder), walls = `forced_box` from test/config.jl: a moving wall, a no-slip wall with value tracers and a salt flux, a heat-flux bottom, a stressed top.
- **Fluid:** α\_TT = 10⁻⁴ K⁻² and γ = 10⁻⁵ Pa⁻¹ (strong cabbeling and thermobaricity); α\_p = 10⁻⁵ Pa⁻¹, so c ≈ 10 m s⁻¹, Ma ≈ 0.05 and gL\_z/c² ≈ 5 %; consistent fluxes; ν and κ small enough that the grid Reynolds number is 10–100.
- **ICs:** a cold, salty block in the upper left over warm, fresh water (RT); a shear layer across mid-depth with J ≈ 0.05 (KH); a density-compensated thermohaline front in the lower half (cabbeling sinks it as it mixes); q close to 0 beside the wall that receives fresh water; streamfunction noise.
- **Runs:** both models; the three vortex schemes; Float64, plus other float types only once N3 has shown they work; `slice_size` 0.05 s, then 2 s with a large `dt_max`.

| Pinch point | What exercises it | Pass criterion |
| --- | --- | --- |
| Semi-discrete energy | every term at once | `integrated_budget_residual` ≤ \~10³ eps(F) E/τ at each output |
| Time-discrete energy | RK4 at `cfl` 0.9, 0.5, 0.25 | E(t) − E(0) − ∫(wall input) falls as Δt⁴⁻⁵ (N3) |
| Mass and tracers | wall fluxes, cabbeling | ∫ρ exact (FC2D); ∫ρq, ∫ρϑ change only by wall fluxes (+ ∫σ\_ϑ); AN2D ‖∇·(ρ\_R u)‖ ≤ eps U/Δ |
| Dissipation and entropy | under-resolved gradients | ρε ≥ 0 in every cell (exact); report the fraction of cells with Tσ\_s < 0, which should fall with resolution |
| Enstrophy | KH roll-up, wall vorticity sheets | EnergyCons: vortex work 0, grid-scale enstrophy pile-up; EnstrophyCons: vortex work ≠ 0; EnergyEnstrophyCons: both conserved; compare spectral tails and time to failure |
| CFL | RT acceleration from rest | the realised Courant number stays ≤ 1 with short slices; with 2 s slices, Δt chosen at rest is too large once the flow accelerates, and AN2D (not limited by sound) is expected to blow up |
| Composition near 0 | fresh-water wall, undershoots of centred advection | `chemical_potential_derivatives` evaluates μ with log(q) on every call, so q < 0 in any cell is expected to raise a DomainError (or NaN in vectorised loops); with consvar = entropy, q = 0 already gives Inf × 0 |
| Walls and corners | moving wall meeting a no-slip wall; stress + heat-flux wall | dE/dt equals the diagnosed wall input to round-off; no corner blow-up |
| Precision | Float32 / Float64 / BigFloat | Float64 until N3 has established which float types work |
| Reference state | turbulent, partly unstable, two-component state | `ExactRearrangement` certificate holds; AE ≥ 0; AE + K budget closes with the R terms; time per output recorded |

- **How to use it:** a stress test shows that something broke, not where. Pair each failure with the unit test that isolates it (N1, N3, W4, F1, B4).
- **Likely outcomes to decide in advance:** for q → 0, clip, use positivity-preserving tracer advection, or regularise s\_mix; for Δt, recompute it more often than once per slice, or bound `dt_max` by an expected maximum velocity.
- **Pertinence:** high, provided it comes after the unit tests.

*References:* background potential energy and mixing, [Winters et al. 1995](https://doi.org/10.1017/S002211209500125X); spurious mixing, [Ilicak et al. 2012](https://doi.org/10.1016/j.ocemod.2011.10.003); energy and enstrophy conservation, [Arakawa & Lamb 1981](https://doi.org/10.1175/1520-0493%281981%29109%3C0018:APEAEC%3E2.0.CO;2); time reversal, [Duponcheel, Orlandi & Winckelmans 2008](https://doi.org/10.1016/j.jcp.2008.06.020).

## Gaps, findings and further experiments

Reading the code turned up six things worth settling before running the list, and one topic central to the project is still missing: an ideal-gas fluid. Cabbeling, external checks of the APE diagnostics, the deep regime and thermobaricity are now E1–E4.

| Finding | Where | Consequence | Suggested action |
| --- | --- | --- | --- |
| "0.18 Δu/δ ≈ 0.4 s⁻¹" and "bulk Ri" | examples/kelvin\_helmholtz\_budget/run.jl | growth rate twice Michalke's; Ri\_min is twice the quoted value | 0.095 Δu/δ; J = 2gα\_TΔθδ/Δu² |
| μ = 10⁻³ Pa s with k\_T = 10⁻⁴, k\_q = 10⁻⁵ m² s⁻¹ | examples, test/config.jl | Pr = 0.01 and Sc = 0.1, unlike water (Pr ≈ 7); layers diffuse faster than they move | choose Pr and Sc per experiment |
| log(q) in μ on every tendency call | ClimFluids `chemical_potential_derivatives` | q ≤ 0 in one cell stops the run | regularise s\_mix, clip, or positivity-preserving advection |
| Δt fixed for a whole output slice | `loop` in both packages | the CFL limit can be exceeded as the flow accelerates | shorter slices, or update Δt within a slice |
| Float types other than Float64 untested; FFTW limits AN2D to Float32 and Float64 | CFAnelastics Poisson solver (FFTW) | the precision rule in AGENTS.md is unverified; BigFloat AN2D impossible as it stands | test each float type (N3); a generic transform for AN2D |
| Only `NonlinearBinaryFluid` provides `chemical_potential_derivatives` and `dcons_dq` | ClimFluids | no ideal-gas runs, so none of the atmospheric benchmarks (Straka density current, Robert bubble, Baldauf–Brdar) | implement them for the ideal gases |

### Further experiments

- **X5 · Cost model (medium).** Wall time per simulated second for AN2D and FC2D against grid, c and threads; it decides which tests can run in CI.

### Where each experiment lives

All 40 experiments live in examples/, in 23 scripts. Experiments share a script when they use the same setup or one feeds the next. Each script opens with a parameter block (grid, model, flux law, duration, and a quick flag for a small, fast run), so it can be stepped through in the REPL or as a notebook.

| Script (examples/…) | Experiments | Why together |
| --- | --- | --- |
| tutorials/build\_models | T1 | the recipe every other script starts from |
| tutorials/diagnostics\_and\_output | T2, T3 | choose diagnostics, then save, read back, restart and plot the same run |
| tutorials/reference\_state | T4 | the FC rest state used as AN reference by A2, E3 and every FC/AN comparison |
| exact\_diffusion | N1, N7 | closed-form viscous and diffusive solutions in columns and channels |
| convergence | N2, N3, N4 | one refinement driver for space order, time order, RK4 energy drift and manufactured solutions |
| symmetry\_and\_reversibility | N5, N6 | the cellular flow is the reversible flow; mirror symmetry uses the I5 seed |
| linear\_modes | W1–W4, E2 (b) | the Jacobian (W4) gives the discrete relations that W1–W3 measure; wave APE comes from the same mode |
| shear\_instability | I1, I3, B5 | one tanh shear layer: stratified (KH, Holmboe) and unstratified and under-resolved |
| rayleigh\_benard | I2, B8 | the same box from onset to Nu–Ra |
| salt\_fingers | I4 |  |
| rayleigh\_taylor | I5, F4 | the buoyancy schemes compared on the single-mode RT |
| acoustic\_adjustment | A1, A3, E2 (c) | heated layer and localised heating, both measured through the ACE |
| sound\_speed\_sweep | A2, E3 | one sweep in c from 4 to 126 m s⁻¹, from the deep regime to the low-Mach limit |
| flux\_laws\_equilibrium | F1, F2 | columns relaxing to equilibrium, for each flux law and entropic variable |
| flux\_laws\_turbulence | F3 |  |
| lock\_exchange | B1, E1, E2 (a) | side-by-side waters: density-driven, compensated (cabbeling), and the APE of a lock |
| cavities | B2, B3 | square cavities: lid-driven and differentially heated |
| dipole\_wall | B4 |  |
| gravity\_wave\_packet | B6 |  |
| horizontal\_convection | B7 |  |
| thermobaricity | E4 |  |
| stress\_test | S1 |  |
| cost\_model | X5 | wall time against grid, c and threads |

test/ keeps the package's own unit tests; it can include any script with quick = true as a smoke test, so the examples stay runnable.

**Decision: examples are Literate.jl scripts.** Each tutorial and study is a plain `.jl` script with Literate comment blocks, so it still runs with `julia examples/…/run.jl` (through `tools/setup.jl`) and can be included by tests. Literate generates Jupyter notebooks and Documenter pages from the same file when needed.

## Implementation guide for agents

This section turns the plan into work in the TraceableEnergetics repository. AGENTS.md takes precedence wherever the two differ, and the code takes precedence over the names of functions and arguments quoted here: report any mismatch.

**Read first.**

- AGENTS.md: hard rules, conventions, when to check in.
- notes/models/traceable\_energetics.tex: the two models, the equation of state, barodiffusion.
- notes/energy\_budget/energy.tex: the energy budget and the available energy.
- notes/reference\_state/ape\_reference\_state.tex and notes/reference\_state/reference\_state\_plan.md: the Lorenz reference state and its budgets.
- src/setups.jl and test/setups.jl: model-independent setups (`setup(model; θ, q)`, `rising_bubble`, …) and how they are tested.
- test/config.jl: the shared fluid, grids and the `forced_box` walls used by S1.
- examples/thermohaline\_lock\_exchange: the most complete current example (models.jl shared by run.jl and figures.jl, `IntegralSeries`, `Thinned` data files, budget figures).

**Rules carried over from AGENTS.md.**

- Never commit or push; give the user the commands.
- Check in before design decisions (APIs, abstractions, numerical methods), after diagnosing a bug and before fixing it, at the end of each milestone below, and whenever unsure about physics (signs, units, reference states).
- Never change CFAnelastics, CFCompressible or another ClimFlows package without the user's permission for that package and that change. Several findings in Gaps (q → 0, Δt within a slice, a generic transform for AN2D, ideal-gas `chemical_potential_derivatives`) are package changes: report them, don't work around them in this repository.
- AN/FC parity: every experiment marked "both" runs both models from the same setup.
- New model-independent setups go in src/setups.jl, with tests in test/setups.jl. New diagnostics are `entry(long_name, units, location, recipe)` merged into the book of `diagnostics`. Experiment-specific I/O, analysis and plotting stay in examples/ or tools/.
- Comments write equations in full and don't cite the notes by file, section or equation number. The Literate prose of a script cites papers by DOI, as in References.

**Script layout.** Each script is examples/\<script>/run.jl in Literate.jl format (`# ` lines become prose, code stays code; section headings are `# ## …`, not the `#####` blocks of the older examples), in this order:

1. A header: what the script tests (experiment IDs), the reference values, the command to run it, and the wall time with quick = true and quick = false.
2. `include(joinpath(@__DIR__, "…", "tools", "setup.jl"))` and then `tools/experiments.jl` the same way, with as many `".."` as the depth (two for examples/\<script>/, three for examples/tutorials/\<script>/).
3. A parameter block, one named tuple: grid, models, flux law, vortex scheme, duration, `cfl`, output interval, and `quick = quick_run()`, which selects a small, fast variant.
4. The models, built with the T1 recipe once T1 exists.
5. The runs, writing only to `dir = output_dir(@__DIR__, quick)`: examples/\<script>/output/, or output/quick/ for the quick variant (both untracked).
6. The checks: a `@testset` comparing each measured value with its reference value (Overview table and the experiment's section) at a tolerance stated and justified in the script, and a printed table of measured against expected values.
7. The figures, in figures.jl when the run is long, reading the output files, so that figures can be redrawn without rerunning.

The experiments run in Float64, with the fluid's parameters built as Float64. Code added to src/ stays generic in the float type, as AGENTS.md requires; other float types enter the experiments only once N3 has shown which work.

**Shared helpers.** Several scripts need the same tools: the discrete wavenumber k̂, the Jacobian of `tendencies!` about a base state (W4; ForwardDiff is already a dependency), the amplitude of a horizontal Fourier mode and an exponential fit of its growth (I1, I2, I4, I5), a fit of the order of convergence (N2, N3, N4), the T1 model recipe and the T4 routes to an AN reference profile. Propose where each lives (tools/ for analysis code included by scripts, src/ for diagnostics and setups) before writing it. The worked diagnostics of T2 (Richardson number, mode amplitude, Courant number) are entries in src/.

**Existing examples.** Six example folders already exist; their content maps onto the plan as follows. Don't delete, rename or merge them without the user's agreement, and keep tools/budget\_figures.jl working for the scripts that use it.

| Existing | Becomes part of | Note |
| --- | --- | --- |
| examples/recording | tutorials/diagnostics\_and\_output (T3) | T3 adds restart, xarray reading and a fine `IntegralSeries` |
| examples/kelvin\_helmholtz, examples/kelvin\_helmholtz\_budget | shear\_instability (I1) | correct the growth rate to 0.095 Δu/δ and Ri to J = 2gα\_TΔθδ/Δu² (Gaps) |
| examples/thermohaline\_lock\_exchange | lock\_exchange (B1, E1, E2 (a)) | reuse models.jl and the budget figures |
| examples/energy\_budget | kept as the rising-bubble budget example | not a validation test |
| examples/vortex\_schemes | S1 (vortex schemes) | has output/ but no script in the repository: ask the user |

**Order of work.** Milestones, each ending with a check-in that summarises what is done and verified. Within a milestone, High priority first.

1. Milestone 0, conventions: done (2026-11-01), as agreed with the user:
   - the script template above;
   - `quick`: `quick_run()` (tools/experiments.jl) reads the environment variable `TE_QUICK`, `"true"` or `"false"` (the default, the full experiment); any other value is an error. Run a quick variant with `TE_QUICK=true julia -t 4 --project=. examples/<script>/run.jl`;
   - shared helpers: analysis code (k̂, fits of growth rates and orders of convergence, the Jacobian, the T1 model recipe) in tools/, included through tools/experiments.jl and tested in test/ as tools/scales.jl is; the T2 diagnostic entries and T4 route (a) in src/. Each is written when its first experiment needs it;
   - test hook: `TE_EXAMPLES=true julia -t 4 --project=. test/runtests.jl` includes each script of `EXAMPLES` (test/runtests.jl) with `TE_QUICK=true`, each in a module of its own (`test_examples`, test/examples.jl). A script joins `EXAMPLES` once it follows the template;
   - tools/setup.jl activates the environment only if it is not already active, and loads Revise only in an interactive session.
2. Milestone 1, foundations: T1, T4, N1, N3 (settles the float types), then T2, T3.
3. Milestone 2, discrete checks: N2, N4–N7, linear\_modes (W1–W4, E2 (b)).
4. Milestone 3, sound and flux laws: acoustic\_adjustment (A1, A3, E2 (c)), sound\_speed\_sweep (A2, E3), F1, F2, F4.
5. Milestone 4, instabilities: I1–I5, B5, F3.
6. Milestone 5, equation of state: lock\_exchange (B1, E1, E2 (a)), E4.
7. Milestone 6, benchmarks: B2–B4, B6–B8.
8. Milestone 7: S1, then X5.

**Done, for each script.**

- `julia -t 4 --project=. examples/<script>/run.jl` runs from a fresh session; with quick = true it finishes in a few minutes at most (X5 measures this).
- Its checks pass, or each failure is diagnosed and reported to the user before anything is changed. A failure the plan predicts (S1's q → 0 and Δt per slice, the float types) is a result: record it.
- Energy, mass and tracer budgets close to round-off wherever the experiment's section says they should.
- `julia -t 4 --project=. test/runtests.jl` still passes.
- The script writes only to its output/ folder.
- This plan (notes/experiments/experiment\_plan.md) records the measured values and any change of setup in the experiment's section.

## References

Every link is a DOI (arXiv DOIs for preprints). Entries without a link are books or papers with no DOI, or whose DOI could not be confirmed.

**Instabilities and convection**

- Alboussière & Ricard (2017), [Rayleigh–Bénard stability and the validity of quasi-Boussinesq or quasi-anelastic liquid approximations](https://doi.org/10.1017/jfm.2017.108), JFM 817.
- Alexakis (2005), [On Holmboe's instability for smooth shear and density profiles](https://doi.org/10.48550/arXiv.physics/0503241) (R = 1 neutral curve J = k(1 − k)).
- Chandrasekhar (1961), *Hydrodynamic and Hydromagnetic Stability*, OUP.
- Curbelo et al. (2019), [Compressible convection: anelastic and anelastic-liquid models vs the exact equations](https://doi.org/10.1017/jfm.2019.420), JFM.
- Duff, Harlow & Hirt (1962), Effects of diffusion on interface instability between gases, Phys. Fluids 5 (via [Morgan et al. 2016](https://doi.org/10.1017/jfm.2016.46)).
- Goncharov (2002), [Analytical model of nonlinear single-mode Rayleigh–Taylor instability at arbitrary Atwood numbers](https://doi.org/10.1103/PhysRevLett.88.134502), PRL 88; C = 3 in 2D per [Bian et al. (2020)](https://doi.org/10.1016/j.physd.2019.132250), Physica D 403.
- Hazel (1972), Numerical studies of the stability of inviscid stratified shear flows, JFM 51.
- [Efficient algorithm on a non-staggered mesh for simulating Rayleigh–Bénard convection](https://doi.org/10.1103/PhysRevE.68.026705) [in a box (arXiv:nlin/0302057)](https://doi.org/10.48550/arXiv.nlin/0302057): Ra\_c = 1707.76, k\_c = 3.117, growth-rate interpolation.
- Kunze (2003), [A review of oceanic salt-fingering theory](https://doi.org/10.1016/S0079-6611%2803%2900027-2), Prog. Oceanogr. 56; Radko (2013), *Double-Diffusive Convection*, CUP; [finger regimes (arXiv:2605.00335)](https://doi.org/10.48550/arXiv.2605.00335).
- Michalke (1964), On the inviscid instability of the hyperbolic-tangent velocity profile, JFM 19; values as reproduced in [arXiv:2206.13205](https://doi.org/10.48550/arXiv.2206.13205).
- Miles (1961), [On the stability of heterogeneous shear flows](https://doi.org/10.1017/S0022112061000305), JFM 10; Howard (1961), JFM 10.
- [Shraiman & Siggia (1990)](https://doi.org/10.1103/PhysRevA.42.3650), Heat transport in high-Rayleigh-number convection, Phys. Rev. A 42.
- Smyth & Carpenter (2019), [*Instability in Geophysical Flows*](https://doi.org/10.1017/9781108640084), CUP, open access.
- Thorpe (2013), [Effect of small viscosity and diffusivity on the marginal stability of stably stratified shear flows](https://doi.org/10.1017/jfm.2013.378), JFM.

**Waves, sound-proofing and adjustment**

- Baldauf & Brdar (2013), [An analytic solution for linear gravity waves in a channel](https://doi.org/10.1002/qj.2105), QJRMS.
- Bannon (1995), [Hydrostatic adjustment: Lamb's problem](https://doi.org/10.1175/1520-0469%281995%29052%3C1743:HALP%3E2.0.CO;2), JAS 52; Bannon (1996), [Nonlinear hydrostatic adjustment](https://doi.org/10.1175/1520-0469%281996%29053%3C3606:NHA%3E2.0.CO;2), JAS 53.
- Brown, Vasil & Zweibel (2012), [Energy conservation and gravity waves in sound-proof treatments of stellar interiors I](https://doi.org/10.1088/0004-637X/756/2/109), ApJ 756.
- Davies, Staniforth, Wood & Thuburn (2003), [Validity of anelastic and other equation sets as inferred from normal-mode analysis](https://doi.org/10.1256/qj.02.1951), QJRMS 129.
- Fanelli & Bannon (2005), [Nonlinear atmospheric adjustment to thermal forcing](https://doi.org/10.1175/JAS3517.1), JAS 62.
- Klainerman & Majda (1981), [Singular limits of quasilinear hyperbolic systems](https://doi.org/10.1002/cpa.3160340405), CPAM 34.
- Klein, Achatz, Bresch, Knio & Smolarkiewicz (2010), [Regime of validity of soundproof atmospheric flow models](https://doi.org/10.1175/2010JAS3490.1), JAS 67.
- Skamarock & Klemp (1994), [Efficiency and accuracy of the Klemp–Wilhelmson time-splitting technique](https://doi.org/10.1175/1520-0493%281994%29122%3C2623:EAAOTK%3E2.0.CO;2), MWR 122.
- Thuburn & Woollings (2005), [Vertical discretizations for compressible Euler equation atmospheric models](https://doi.org/10.1016/j.jcp.2004.08.018), JCP 203; Thuburn, [Vertical discretizations giving optimal representation of normal modes: general equations of state](https://doi.org/10.1002/qj.2971), QJRMS.

**Energetics, thermodynamics and mixing**

- Dubos (2026), [Thermodynamically consistent large-eddy simulation models](https://doi.org/10.48550/arXiv.2601.16957).
- Gassmann & Herzog (2015), [How is local material entropy production represented in a numerical model?](https://doi.org/10.1002/qj.2404), QJRMS 141.
- Graham & McDougall (2013), [Quantifying the nonconservative production of Conservative Temperature, potential temperature and entropy](https://doi.org/10.1175/JPO-D-11-0188.1), JPO.
- IOC, SCOR & IAPSO (2010), TEOS-10 manual, appendix B.
- Ilicak, Adcroft, Griffies & Hallberg (2012), [Spurious dianeutral mixing and the role of momentum closure](https://doi.org/10.1016/j.ocemod.2011.10.003), Ocean Modelling 45–46.
- [Kraichnan & Montgomery (1980)](https://doi.org/10.1088/0034-4885/43/5/001), Two-dimensional turbulence, Rep. Prog. Phys. 43.
- McDougall (2003), [Potential enthalpy: a conservative oceanic variable](https://doi.org/10.1175/1520-0485%282003%29033%3C0945:PEACOV%3E2.0.CO;2), JPO 33.
- Middleton & Taylor (2020), [A general criterion for the release of background potential energy through double diffusion](https://doi.org/10.1017/jfm.2020.259), JFM 893.
- Osborn & Cox (1972), Oceanic fine structure, Geophys. Fluid Dyn. 3.
- Paparella & Young (2002), [Horizontal convection is non-turbulent](https://doi.org/10.1017/S0022112002001313), JFM 466; Hazewinkel, Paparella & Young, [Stressed horizontal convection](https://doi.org/10.1017/jfm.2011.514).
- Tailleux (2009), [On the energetics of stratified turbulent mixing, irreversible thermodynamics, Boussinesq models and the ocean heat engine controversy](https://doi.org/10.1017/S002211200999111X), JFM 638.
- Tailleux (2010), [Thermodynamics/dynamics coupling and thermodynamic consistency of Boussinesq and anelastic binary fluids](https://doi.org/10.48550/arXiv.1011.5071).
- Tailleux (2024), [Negative APE dissipation as the fundamental criterion for double diffusive instabilities](https://doi.org/10.1017/jfm.2024.647), JFM 994.
- Tailleux & Dubos (2024), [Static Energy Asymptotics (SEA)](https://doi.org/10.1016/j.ocemod.2024.102339), Ocean Modelling 188.
- Tailleux, Dubos & Hatton (2026), [An energetically and thermodynamically consistent Boussinesq model](https://doi.org/10.48550/arXiv.2512.18461), GAFD.
- Winters, Lombard, Riley & D'Asaro (1995), [Available potential energy and mixing in density-stratified fluids](https://doi.org/10.1017/S002211209500125X), JFM 289.

**Benchmarks and verification**

- Botella & Peyret (1998), [Benchmark spectral results on the lid-driven cavity flow](<https://doi.org/10.1016/s0045-7930(98)00002-4>), Comput. Fluids 27; [Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991%2882%2990058-4), JCP 48.
- Clercx & Bruneau (2006), [The normal and oblique collision of a dipole with a no-slip boundary](https://doi.org/10.1016/j.compfluid.2004.11.009), Comput. Fluids 35.
- de Vahl Davis (1983), [Natural convection of air in a square cavity: a bench mark numerical solution](https://doi.org/10.1002/fld.1650030305), IJNMF 3.
- Minion & Brown (1997), [Performance of under-resolved two-dimensional incompressible flow simulations II](https://doi.org/10.1006/jcph.1997.5843), JCP 138.
- Roache (2002), [Code verification by the method of manufactured solutions](https://doi.org/10.1115/1.1436090), J. Fluids Eng. 124; [Roy, Nelson, Smith & Ober (2004)](https://doi.org/10.1002/fld.660); [Veluri, Roy & Luke 2012](https://doi.org/10.1016/j.compfluid.2012.04.028).
- Sadourny (1975), [The dynamics of finite-difference models of the shallow-water equations](https://doi.org/10.1175/1520-0469%281975%29032%3C0680:TDOFDM%3E2.0.CO;2), JAS 32; [Arakawa & Lamb (1981)](https://doi.org/10.1175/1520-0493%281981%29109%3C0018:APEAEC%3E2.0.CO;2), discussed by [Salmon 2004](https://doi.org/10.1175/1520-0469%282004%29061%3C2016:PATTCO%3E2.0.CO;2).
- Shin, Dalziel & Linden (2004), [Gravity currents produced by lock exchange](https://doi.org/10.1017/S002211200400165X), JFM 521.
- Ramadhan et al. (2020), [Oceananigans.jl: fast and friendly geophysical fluid dynamics on GPUs](https://doi.org/10.21105/joss.02018), JOSS 5; its documentation is built from Literate.jl tutorials.

Numerics, verification and further sources

- Bannon (1996), [On the anelastic approximation for a compressible atmosphere](https://doi.org/10.1175/1520-0469%281996%29053%3C3618:OTAAFA%3E2.0.CO;2), JAS 53.
- Chagnon & Bannon (2005), [Adjustment to injections of mass, momentum, and heat in a compressible atmosphere](https://doi.org/10.1175/JAS3503.1), JAS 62.
- Duponcheel, Orlandi & Winckelmans (2008), [Time-reversibility of the Euler equations as a benchmark for energy conserving schemes](https://doi.org/10.1016/j.jcp.2008.06.020), JCP 227.
- Grossmann & Lohse (2000), [Scaling in thermal convection: a unifying theory](https://doi.org/10.1017/S0022112099007545), JFM 407.
- Hairer, Lubich & Wanner (2006), [Geometric Numerical Integration](https://doi.org/10.1007/3-540-30666-8), 2nd ed., Springer.
- Oberkampf & Roy (2010), [Verification and Validation in Scientific Computing](https://doi.org/10.1017/CBO9780511760396), CUP.
- Ogura & Phillips (1962), [Scale analysis of deep and shallow convection in the atmosphere](https://doi.org/10.1175/1520-0469%281962%29019%3C0173:SAODAS%3E2.0.CO;2), JAS 19.
- Schlichting & Gersten (2017), [Boundary-Layer Theory](https://doi.org/10.1007/978-3-662-52919-5), 9th ed., Springer.
- Taylor & Green (1937), [Mechanism of the production of small eddies from large ones](https://doi.org/10.1098/rspa.1937.0036), Proc. R. Soc. A 158.
