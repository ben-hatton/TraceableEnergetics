module TraceableEnergetics
# Shows that anelastic dynamics are energetically traceable to the fully compressible equations: model-independent
# initial conditions (setups), shared by the anelastic (AN2D) and fully compressible (FC2D) box models, the diagnostics
# of either model, and their output during a run (NetCDF or JLD2 data files, videos).

using CFBoxes: Periodic, Bounded, boundary_topology, cellpoint, ncells, nfaces, interior, coordinates, dims,
               halo_size, cellgrid, facegrid, Face, Cell, alloc, interior_cells, interior_faces, fill_halo!, ZeroGradient, periodize!,
               operators
using ManagedLoops: @vec, @with
using CookBooks: CookBooks, CookBook
using CFAnelastics: CFAnelastics, AN2D
using CFCompressible: CFCompressible, FC2D
using NCDatasets: NCDataset, defDim, defVar
using JLD2: JLD2, jldopen
using CairoMakie: Makie, Figure, Axis, Colorbar, Observable, heatmap!, lift, record

export linear_profile, tanh_profile, shear, jet, couette, mode, noise, consvar_from_theta
export setup, kelvin_helmholtz, rayleigh_benard, rising_bubble, lock_exchange
export initialize
export diagnostics, diagnostic_info, interior_diagnostic, diagnostic_coordinates
export Outputs, record!, DataFile, read_diagnostic, DiagnosticVideo

include("setups.jl")
include("initialize.jl")
include("energetics.jl")
include("diagnostics.jl")
include("output.jl")

end # module TraceableEnergetics
