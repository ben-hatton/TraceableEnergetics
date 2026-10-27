# TraceableEnergetics

Demonstrates that anelastic (and hence Boussinesq) dynamics are energetically traceable to the fully
compressible equations, using models built in the [ClimFlows](https://github.com/ClimFlows) ecosystem.

## Layout

```
src/               the TraceableEnergetics package: model-independent setups (initial conditions) and their
                   lowering to each model's initialize, the diagnostics of either model, and their output during a
                   run (NetCDF or JLD2 data files, time series, videos)
  energetics/        the energy budget: kernels and diagnostics
  reference_state/   the Lorenz reference state of available energy: placement of the parcels, reference heights,
                     available energy, tendency of the reference state, budgets of the available energy
  numerics/          interpolation, quadrature, minimisation, placement of masses, the transportation simplex
examples/          scripts that run the models, each writing to its own output/ (untracked):
  kelvin_helmholtz/          a shear layer in both models, side by side
  recording/                 the outputs of a run: data files and videos, and reading them back
  energy_budget/             local and global energy budgets of a rising bubble in both models
  kelvin_helmholtz_budget/   the energy budget of a Kelvin-Helmholtz shear layer in both models
  thermohaline_lock_exchange/ the budgets of the available energy of a thermohaline lock exchange in both models
tools/             helpers shared by scripts (setup.jl, budget figures, scales, stratification) and studies
notes/             the specifications of the models, the energy budget and the APE reference state
test/              setups tests, canonical setups, diagnostics and outputs on both models; cross-model tests
```

The models are ordinary registered dependencies, developed in their own repositories:
[CFAnelastics](https://github.com/ClimFlows/CFAnelastics.jl) (AN2D) and
[CFCompressible](https://github.com/ClimFlows/CFCompressible.jl) (FC2D), with compat entries in `Project.toml`. Their
own test suites (conservation, walls, allocations) run in their repositories.

To record the energy budget during a run, pass the book of `TraceableEnergetics.diagnostics` to `loop`, which
otherwise hands its outputs the diagnostics of the model package:
`loop(state, model, time_scheme, params; outputs, book = diagnostics(model))`
(see examples/energy_budget/run.jl).

## Getting started

```bash
git clone https://github.com/ben-hatton/TraceableEnergetics.git
cd TraceableEnergetics
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 4 --project=. test/runtests.jl
```

Scripts start with `include("<path to>/tools/setup.jl")`, which activates this environment and loads Revise.

## Updating or changing the models

`Manifest.toml` is untracked: to pick up new releases of the models run `Pkg.update(["CFAnelastics", "CFCompressible"])`,
and raise their compat entries when this repository needs a new feature. To change a model, develop it from a clone
of your fork (`Pkg.develop(path = "<clone>")`, which changes only the Manifest), push the branch to the fork and open a
pull request upstream; once it is released, `Pkg.free` the package and raise its compat entry here.
