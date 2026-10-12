module TraceableEnergetics
# Shows that anelastic dynamics are energetically traceable to the fully compressible equations: model-independent
# initial conditions (setups), shared by the anelastic (AN2D) and fully compressible (FC2D) box models, and the
# diagnostics of either model, with their visualisation (Makie extension).

using CFBoxes: Periodic, Bounded, boundary_topology, zpoint, xdim, zdim, Xdim, Zdim, interior, coordinates
using CookBooks: CookBooks
using CFAnelastics: CFAnelastics, AN2D
using CFCompressible: CFCompressible, FC2D

export linear_profile, tanh_profile, shear, jet, couette, mode, noise, consvar_from_theta
export rest_state, kelvin_helmholtz, rayleigh_benard, rising_bubble, lock_exchange
export initialize
export diagnostics, diagnostic_info, interior_diagnostic, diagnostic_coordinates, DiagnosticVideo, record_frame!

include("setups.jl")
include("initialize.jl")
include("diagnostics.jl")

end # module TraceableEnergetics
