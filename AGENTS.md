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
- experiments/, tools/: the science that uses the packages. Experiment-specific code (I/O, plotting,
  setups) lives here or in src/, never in the packages.

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

## WORKING WITH THE USER
Check in with the user:
- before design decisions (APIs, abstractions, numerical methods), with options and a recommendation;
- after diagnosing bugs, before fixing them;
- at each milestone, with a summary of what is done and verified;
- whenever unsure about physics (signs, units, reference states). Never guess these.
