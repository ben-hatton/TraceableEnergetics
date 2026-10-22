# TraceableEnergetics

Demonstrates that anelastic (and hence Boussinesq) dynamics are energetically traceable to the fully
compressible equations, using models built in the [ClimFlows](https://github.com/ClimFlows) ecosystem.

## Layout

```
src/               the TraceableEnergetics package: model-independent setups (initial conditions) and their
                   lowering to each model's initialize
experiments/       simulations that use the packages (e.g. shear/: Kelvin-Helmholtz)
tools/             helpers shared by scripts (setup.jl, output, NetCDF)
test/              setups tests, canonical setups and diagnostics on both models; cross-model tests
```

The models are ordinary dependencies, developed in their own repositories:
[CFAnelastics](https://github.com/ben-hatton/CFAnelastics.jl) (AN2D) and
[CFCompressible](https://github.com/ben-hatton/CFCompressible.jl) (FC2D), both forks of
[ClimFlows](https://github.com/ClimFlows) packages, on the branch `traceable-energetics`. `Project.toml` installs them
from there through `[sources]` (Julia ≥ 1.11). Their own test suites (conservation, walls, allocations) run in their
repositories.

## Getting started

```bash
git clone https://github.com/ben-hatton/TraceableEnergetics.git
cd TraceableEnergetics
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 4 --project=. test/runtests.jl
```

Scripts start with `include("<path to>/tools/setup.jl")`, which activates this environment and loads Revise.

## Updating or changing the models

`Manifest.toml` is untracked, so the models follow the tip of `traceable-energetics`: to pick up new commits run
`Pkg.update("CFAnelastics", "CFCompressible")`. To change a model, work in a clone of its fork (e.g. with
`Pkg.develop(path = "<clone>")` in a scratch environment), push the branch to the fork, then update here.
Changes to ClimFlows packages go upstream by pull request from the fork.
