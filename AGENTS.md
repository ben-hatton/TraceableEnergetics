## GUIDANCE FOR AGENTS

## CONTEXT
ClimFlows is an ongoing effort to develop an ecosystem of composable and extensible packages.
CFAnelastics is a package that models a fluid under the anelastic approximation.
CFCompressible is a package that models a compressible fluid.
The current project concerns 2D and 3D energy-conserving finite volume models in a box geometry.
The packages also contain code for other methods and geometries.

## GOAL
Show that anelastic (and hence Boussinesq) dynamics are energetically traceable to the fully compressible
equations, using energy-consistent finite-volume discretisations of both models in a box.

## LAYOUT
- CFAnelastics, CFCompressible and CFPlanets are ordinary registered dependencies (compat in Project.toml). They are
  developed in their own repositories, not here; while a change is pending, the dependency is the user's fork branch
  `traceable-energetics` through [sources] in Project.toml.
- examples/, tools/: the science that uses the packages. Experiment-specific code (I/O, plotting,
  setups) lives here or in src/, never in the packages.
- notes/: the specifications. traceable_energetics.tex specifies the models; ape_reference_state.tex specifies the
  algorithms for the reference state of available potential energy and records the decisions taken with the user.

## MANAGING THE PACKAGES
- CFAnelastics and CFCompressible are ClimFlows packages like the others: the rules for upstream ClimFlows packages
  apply to them (CFDiffusionSchemes, ClimFluids, CFBoxes, CFPlanets, CFDomains, ...).
- Never `Pkg.develop` or edit ~/.julia/dev, and never touch upstream repositories, branches other than
  `traceable-energetics`, or files outside this repository, except under the procedure below.
- Changes to a ClimFlows package may be *proposed* by developing it with `Pkg.develop` from ~/.julia/dev/<Package> on a
  local branch `traceable-energetics`, but only after the user has given explicit permission for that package and that
  change. The agent never commits or pushes. The user pushes the branch to their own fork first, then opens the pull
  request to the upstream repository themselves. Manifest.toml files are untracked, so `Pkg.develop` changes no tracked
  file. After the change is pushed to the fork, the dependency in this repository is the fork's branch.
- When changes are made to the finite volume dynamical code in CFCompressible, analogous changes must be made in
  CFAnelastics, and vice versa, in the same piece of work (or listed as pending). Shared code is mirrored by hand.
- Keep packages upstream-ready: follow ClimFlows conventions, nothing specific to this project.
- Structure code so that 3D can be added (e.g. dispatch on the box type), but do not design 3D operators until asked.

## GIT
Never commit or push. When work is ready, give the user the commands, with an absolute path for each `cd`. For a
package developed under the procedure above: commit in the package first, push its branch to the fork, then commit the
change in this repository.

## DEFINITION OF DONE (hard rules)
- `julia -t 4 --project=. test/runtests.jl` passes, apart from failures already known and listed. It covers the setups,
  the canonical setups and the diagnostics and outputs on both models; the model suites (conservation, walls,
  allocations) run in the CFAnelastics and CFCompressible repositories and must pass there.
- Energy-consistent schemes conserve energy semi-discretely to round-off; with forcing walls, dE/dt equals the
  wall input to round-off.
- `tendencies!` allocates 0 bytes per call on a serial manager (type-stable, concrete parametric structs).
- AN/FC parity, as above.
- Tendencies never read uninitialised memory or depend on halo values that boundary conditions don't define,
  and never modify the interior of the state.
- Tests of a new method cover its limiting cases and extreme cases, in more than one float type, and find where it
  breaks: e.g. the parameter at which an approximation fails, or the precision below which a result loses meaning. The
  suite uses small grids; larger studies go in scripts.

## CODE CONVENTIONS
- ClimFlows idioms: ManagedLoops `@with mgr, let ... @vec for ...`; CFBoxes `alloc!(scratch_, F, mgr, domain, location)`
  with MutatingOrNot `void` for scratch (not `similar!`); `tendencies!(dstate, scratch, model, state, t)` returning
  `(dstate, scratch)`.
- Grid locations (CFBoxes `CGrid`) are tuples of `Cell()`/`Face()`, one per dimension: `(Face(), Cell())` for u,
  `(Cell(), Face())` for w, `(Face(), Face())` for vertices. The old `:xz`/`:Xz` symbols are obsolete. Declare a
  location once (`diagnostic_info`), never infer it from another array. Operators are `ℑ(a, dims, to, I...)` and
  `δ(...)` with explicit dimensions and target staggering (read `CFBoxes/src/CGrid/operators.jl`); `I, J` index
  faces, `i, j` centres.
- Each kernel's comment states the continuous equation and its discrete form and staggering.
- Every exported type and function gets a docstring with units and sign conventions.
- Comments and docstrings describe what the code does and write out in full the equations it represents. Don't refer to
  the notes (file names, section or equation numbers) in place of an equation. Most of this code is nontrivial: comment
  individual lines so that it is clear what each one does.
- Names are descriptive, in code and notes: name methods and quantities by what they do (e.g. "from the diabatic
  tendencies", not "method B"). Avoid words with another meaning in a fluid model (e.g. "transport", which means
  advection; the reference state is a "rearrangement" of the parcels, with a "placement" of their mass).
- Precision: all code is generic in the float type `F` of the model, which the user chooses freely (Float32, Float64,
  Float128, BigFloat). Nothing is fixed to Float64: no Float64 literals in kernels (use `F(...)` or `oftype`),
  tolerances scale with `eps(F)`, and no step that determines a result may use a library that only works in Float64.
  ClimFluids promotes to the type of the fluid's parameters, so the fluid must be built with parameters of type `F`.
- Performance: optimise for the CPU. Use the model's manager (`@with mgr`, `@vec`) for threads and SIMD wherever a loop
  allows it.
- Allocations: allocate as little as possible and reuse storage. Workspaces are allocated once and kept (e.g. closed
  over by the recipes of a CookBook); never allocate state-sized arrays per call. If allocations become a problem,
  report it to the user rather than working around it.
- Dependencies: take routines from existing packages, unless writing our own removes a bottleneck or the package
  breaks the precision rule above.
- Thermodynamics: take the specific volume and its derivatives from ClimFluids, e.g.
  `fluid(:p, :consvar, :q).volume_functions(p, consvar, q)` returns (v, ∂v/∂p, ∂v/∂consvar, ∂v/∂q); don't rebuild them
  by finite differences or automatic differentiation.
- Diagnostics: define new diagnostics as `entry(long_name, units, location, recipe)` in their own function and merge
  them into the book of `diagnostics(model; ...)`, as the energetics entries are. During a run they are computed at the
  outputs of `loop` (t = 0 and the end of each slice, set by `Nslice` and `slice_size`). Vertical profiles are 1D
  variables (on the z dimension, or a dimension of their own such as reference-state slots), not full cell fields.

## CURRENT WORK
- The reference state of available potential energy (not started). Start from notes/reference_state_plan.md, which
  lists what to settle with the user before writing code.

## WORKING WITH THE USER
Check in with the user:
- before design decisions (APIs, abstractions, numerical methods), with options and a recommendation;
- after diagnosing bugs, before fixing them;
- at each milestone, with a summary of what is done and verified;
- whenever unsure about physics (signs, units, reference states). Never guess these.
