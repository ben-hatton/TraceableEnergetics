## GUIDANCE FOR AGENTS

## CONTEXT
ClimFlows is an ongoing effort to develop an ecosystem of composable and extensible packages.
CFAnelastics is a package that models a fluid under the anelastic approximation.
CFCompressible is a package that models a compressible fluid.
The current project concerns 2D and 3D energy-conserving finite volume models in a box geometry.
The packages also contain code for other methods and geometries.

## GOAL
Show that anelastic (and hence Boussinesq) dynamics are energetically traceable to the fully compressible
equations, using energy-consistent finite-volume discretisations (SEA framework) of both models in a box.

## LAYOUT
- packages/: work-in-progress packages. CFAnelastics is in-tree; CFCompressible, ClimFluids and CFBoxes are
  git submodules of the user's forks, each on the branch `traceable-energetics`.
- The environment uses packages/ through [sources] in Project.toml. Never `Pkg.develop` or edit ~/.julia/dev.
- experiments/, analysis/, tools/: the science that uses the packages. Experiment-specific code (I/O, plotting,
  setups) lives here, never in the packages.

## MANAGING THE PACKAGES
- Edit freely: CFAnelastics, the box code of CFCompressible (FC2D files), CFBoxes.
- Ask first: ClimFluids, the non-box code of CFCompressible, CFDomains, CFDiffusionSchemes, any new package.
- Never touch branches other than `traceable-energetics`, upstream ClimFlows repos, or files outside this repository.
- When changes are made to the finite volume dynamical code in CFCompressible, analogous changes must be made in
  CFAnelastics, and vice versa, in the same piece of work (or listed as pending). Shared code is mirrored by hand.
- Keep packages upstream-ready: follow ClimFlows conventions, nothing specific to this project.
- Structure code so that 3D can be added (e.g. dispatch on the box type), but do not design 3D operators until asked.

## GIT
Never commit or push. When work is ready, give the user the commands: commit inside each changed submodule
first, then record the submodule pointers in the parent repository.

## DEFINITION OF DONE (hard rules)
- `julia -t 4 --project=. test/runtests.jl` passes, apart from failures already known and listed.
- Energy-consistent schemes conserve energy semi-discretely to round-off; with forcing walls, dE/dt equals the
  wall input to round-off.
- `tendencies!` allocates 0 bytes per call on a serial manager (type-stable, concrete parametric structs).
- AN/FC parity, as above.
- Tendencies never read uninitialised memory or depend on halo values that boundary conditions don't define,
  and never modify the interior of the state.

## CODE CONVENTIONS
- ClimFlows idioms: ManagedLoops `@with mgr, let ... @vec for ...`; MutatingOrNot `similar!`/`void` for
  scratch; `tendencies!(dstate, scratch, model, state, t)` returning `(dstate, scratch)`.
- Each kernel's comment states the continuous equation and its discrete form and staggering.
- Every exported type and function gets a docstring with units and sign conventions.
- Do not test fluid thermodynamics here; ClimFluids tests itself.

## WORKING WITH THE USER
Check in with the user:
- before design decisions (APIs, abstractions, numerical methods), with options and a recommendation;
- after diagnosing bugs, before fixing them;
- at each milestone, with a summary of what is done and verified;
- whenever unsure about physics (signs, units, reference states). Never guess these.
