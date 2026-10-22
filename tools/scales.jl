# Model-independent helpers for scales, nondimensional numbers and stratification diagnostics. Plain functions of
# scalars (or arrays, by broadcasting) in SI units, for any fluid (Boussinesq or compressible, any equation of state):
# no physical constant is assumed, g, ν, κ, ... are always arguments.  Use with:  include("<relative path>/tools/scales.jl")
include("nondimensional.jl")
include("stratification.jl")
include("turbulence_scales.jl")
