# TraceableEnergetics

Demonstrates that anelastic (and hence Boussinesq) dynamics are energetically traceable to the fully
compressible equations, using models built in the [ClimFlows](https://github.com/ClimFlows) ecosystem.

## Layout

```
src/               the TraceableEnergetics package: model-independent setups (initial conditions) and their
                   lowering to each model's initialize
packages/          work-in-progress packages, each with its own Project.toml and tests
  CFAnelastics/    git submodule: fork of ClimFlows/CFAnelastics.jl (AN2D)
  CFCompressible/  git submodule: fork of ClimFlows/CFCompressible.jl (FC2D)
experiments/       simulations that use the packages (e.g. shear/: Kelvin-Helmholtz)
energetics/        staging area: energetics of AN2D (traditional and available energetics, NetCDF post-processing),
                   moved out of CFAnelastics
analysis/          post-processing
tools/             helpers shared by scripts (setup.jl, output, NetCDF)
test/              setups tests, the package model suites, canonical setups on both models; cross-model tests
```

`Project.toml` points to `packages/` through `[sources]` (Julia ≥ 1.11), so the environment always uses
the local versions of the packages.

## Getting started

```bash
git clone --recursive https://github.com/ben-hatton/TraceableEnergetics.git   # or: git submodule update --init
cd TraceableEnergetics
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 4 --project=. test/runtests.jl
```

Scripts start with `include("<path to>/tools/setup.jl")`, which activates this environment and loads Revise.

## Working on the forked packages

Each submodule is a full clone of a fork, checked out on the branch `traceable-energetics`. Changes are
committed in two steps:

```bash
cd packages/ClimFluids
git add -p && git commit -m "..."
git push -u origin traceable-energetics
cd ../..
git add packages/ClimFluids          # record the new submodule commit
git commit -m "Bump ClimFluids"
```

The parent repository only records which commit of each package to use. Push the package first, so that
the recorded commit exists on GitHub.
