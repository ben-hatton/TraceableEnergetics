# TraceableEnergetics

Demonstrates that anelastic (and hence Boussinesq) dynamics are energetically traceable to the fully
compressible equations, using models built in the [ClimFlows](https://github.com/ClimFlows) ecosystem.

## Layout

```
packages/          work-in-progress packages, each with its own Project.toml and tests
  CFAnelastics/    anelastic 2D box model (AN2D)
  CFCompressible/  git submodule: fork of ClimFlows/CFCompressible.jl (FC2D)
  ClimFluids/      git submodule: fork of ClimFlows/ClimFluids.jl (NonlinearBinaryFluid)
  CFBoxes/         git submodule: fork of ClimFlows/CFBoxes.jl
experiments/       simulations that use the packages (e.g. shear/: Kelvin-Helmholtz)
analysis/          post-processing
tools/             helpers shared by scripts (setup.jl, output, NetCDF)
test/              runs the package model suites; cross-model tests
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
