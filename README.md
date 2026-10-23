# TraceableEnergetics

Demonstrates that anelastic (and hence Boussinesq) dynamics are energetically traceable to the fully
compressible equations, using models built in the [ClimFlows](https://github.com/ClimFlows) ecosystem.

## Layout

```
src/               the TraceableEnergetics package: model-independent setups (initial conditions) and their
                   lowering to each model's initialize, the diagnostics of either model with the energy budget,
                   and their output during a run (NetCDF or JLD2 data files, videos)
examples/          scripts that run the models, each writing to its own output/ (untracked):
  kelvin_helmholtz/  a shear layer in both models, side by side
  recording/         the outputs of a run: data files and videos, and reading them back
  energy_budget/     local and global energy budgets of a rising bubble in both models
  vortex_schemes/    energy conservation and nonlinear behaviour of the vortex-force schemes
tools/             helpers shared by scripts (setup.jl, scales, stratification)
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
