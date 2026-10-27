module TraceableEnergetics
# Shows that anelastic dynamics are energetically traceable to the fully compressible equations: model-independent
# initial conditions (setups), shared by the anelastic (AN2D) and fully compressible (FC2D) box models, the diagnostics
# of either model, among them the reference state of available potential energy, and their output during a run (NetCDF
# or JLD2 data files, videos).

using CFBoxes: Periodic, Bounded, boundary_topology, cellpoint, ncells, nfaces, interior, coordinates, dims,
               halo_size, cellgrid, facegrid, Face, Cell, alloc, interior_cells, interior_faces, fill_halo!, ZeroGradient, periodize!,
               operators
using ManagedLoops: @vec, @with
using CookBooks: CookBooks, CookBook
using CFAnelastics: CFAnelastics, AN2D
using CFCompressible: CFCompressible, FC2D
using NCDatasets: NCDataset, defDim, defVar
using JLD2: JLD2, jldopen
using Roots: find_zero, A42
using ForwardDiff: Dual, partials
using CairoMakie: Makie, Figure, Axis, Colorbar, Observable, heatmap!, lift, record

export linear_profile, tanh_profile, shear, jet, couette, mode, noise, consvar_from_theta
export setup, kelvin_helmholtz, rayleigh_benard, rising_bubble, lock_exchange
export initialize
export diagnostics, diagnostic_info, interior_diagnostic, diagnostic_coordinates
export Outputs, record!, DataFile, read_diagnostic, DiagnosticVideo, IntegralSeries, read_series, Thinned
export AbstractRearrangement, SortedRearrangement, ExactRearrangement, AnelasticLorenzReference, CompressibleLorenzReference, reference_state!

# initial states
include("setups.jl")
include("initialize.jl")
# numerical methods of the reference state: interpolation, minimisation, placement of masses, the transportation
# simplex, quadrature
include("numerics/interpolation.jl")
include("numerics/minimisation.jl")
include("numerics/placement.jl")
include("numerics/transportation.jl")
include("numerics/quadrature.jl")
# the energy budget: kernels and diagnostics
include("energetics/kernels.jl")
include("energetics/entries.jl")
# the Lorenz reference state of each model: the state and its steps, the placement of the parcels, their reference
# heights and available energy, the tendency of the reference state, the budgets of the available energy, and their
# diagnostics
include("reference_state/lorenz_reference.jl")
include("reference_state/rearrangement.jl")
include("reference_state/reference_heights.jl")
include("reference_state/available_energy.jl")
include("reference_state/tendency.jl")
include("reference_state/budget.jl")
include("reference_state/entries.jl")
# the book of diagnostics, and its outputs during a run
include("diagnostics.jl")
include("output.jl")

end # module TraceableEnergetics
