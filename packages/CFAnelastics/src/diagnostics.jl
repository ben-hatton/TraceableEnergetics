## DIAGNOSTICS
#
# Every diagnostic is the full array, halo included, with the halo defined (copied from the bounded state and the
# tendency scratch, or filled by the boundary conditions), so that post-processing can difference it at the
# boundaries; strip the halo for plotting (e.g. `interior(domain, field)` for cell fields). Names are shared with the
# FC2D diagnostics (CFCompressible); `diagnostic_info` gives the long name and units of each.

"""
    book = diagnostics(model::AN2D)

`CookBook` of the AN2D diagnostics, opened with `open(book; state, dstate, scratch, model)`. Names, long names and
units: see `diagnostic_info` (or `describe_diagnostics(model)`). Debugging intermediates: `internal_diagnostics`.
"""
diagnostics(::AN2D) = CookBook(;
    # state
    conservative_variable            = scratch -> scratch.thermo.consvar,
    composition                      = scratch -> scratch.thermo.q,
    u                                = state -> state.u,
    w                                = state -> state.w,
    density                          = (state, model) -> reference_field(model.reference.ρ_R, state.m),
    ρconsvar                         = state -> state.m[:, :, 1],
    ρq                               = state -> state.m[:, :, 2],
    # thermodynamics
    specific_entropy                 = (model, pressure, conservative_variable, composition) -> model.fluid(:p, :consvar, :q).specific_entropy.(pressure, conservative_variable, composition),
    potential_temperature            = (model, pressure, conservative_variable, composition) -> model.fluid(:p, :consvar, :q).potential_temperature.(pressure, conservative_variable, composition),
    pressure                         = (scratch, model) -> pressure_field(model.reference, scratch.momentum.φ),
    temperature                      = scratch -> scratch.thermo.T,
    conjugate_variable               = scratch -> scratch.thermo.conjvar,
    modified_chemical_potential      = scratch -> scratch.thermo.chempot,
    forcing_potential                = scratch -> scratch.momentum.B,
    buoyancy                         = scratch -> scratch.thermo.b,
    kinematic_pressure_anomaly       = scratch -> scratch.momentum.φ,
    # kinematics
    vorticity                        = scratch -> scratch.derivatives.∂u_z .- scratch.derivatives.∂w_x,
    divergence                       = scratch -> scratch.derivatives.∂u_x .+ scratch.derivatives.∂w_z,
    mass_divergence                  = (state, model) -> divergence_field(model, mass_fluxes(model, state)...),
    kinetic_energy                   = scratch -> scratch.advection.K,
    ∂u_x                             = scratch -> scratch.derivatives.∂u_x,
    ∂w_z                             = scratch -> scratch.derivatives.∂w_z,
    ∂u_z                             = scratch -> scratch.derivatives.∂u_z,
    ∂w_x                             = scratch -> scratch.derivatives.∂w_x,
    # irreversible processes
    Jcons_x                          = scratch -> scratch.irreversible.Jcons_x,
    Jcons_z                          = scratch -> scratch.irreversible.Jcons_z,
    Jq_x                             = scratch -> scratch.irreversible.Jq_x,
    Jq_z                             = scratch -> scratch.irreversible.Jq_z,
    viscous_dissipation              = scratch -> scratch.irreversible.ρε,
    conservative_variable_production = scratch -> scratch.irreversible.σ_cons,
    div_Jcons                        = (scratch, model) -> divergence_field(model, scratch.irreversible.Jcons_x, scratch.irreversible.Jcons_z),
    div_Jq                           = (scratch, model) -> divergence_field(model, scratch.irreversible.Jq_x, scratch.irreversible.Jq_z),
    # tendencies
    dρconsvar                        = dstate -> dstate.m[:, :, 1],
    dρq                              = dstate -> dstate.m[:, :, 2],
    du                               = dstate -> dstate.u,
    dw                               = dstate -> dstate.w,
)

"""
    book = internal_diagnostics(model::AN2D)

`CookBook` of intermediates of the AN2D tendencies, for debugging: the vorticity terms ωw, ωu of the advection, the
advective fluxes of the tracers mu, mw (third index: tracer), and the gradients of the kinetic energy ∂K_x, ∂K_z.
"""
internal_diagnostics(::AN2D) = CookBook(;
    ωw   = scratch -> scratch.advection.ωw,
    ωu   = scratch -> scratch.advection.ωu,
    mu   = scratch -> scratch.advection.mu,
    mw   = scratch -> scratch.advection.mw,
    ∂K_x = scratch -> scratch.advection.∂K_x,
    ∂K_z = scratch -> scratch.advection.∂K_z,
)

"""
    info = diagnostic_info(model::AN2D)

Long name and units of each diagnostic of `diagnostics(model)`, as a named tuple of `(; long_name, units)`.
"""
diagnostic_info(::AN2D) = (;
    conservative_variable            = (long_name = "conservative variable",                           units = "consvar"),
    composition                      = (long_name = "composition (solute mass fraction)",              units = "kg/kg"),
    u                                = (long_name = "horizontal velocity",                             units = "m s⁻¹"),
    w                                = (long_name = "vertical velocity",                               units = "m s⁻¹"),
    density                          = (long_name = "reference density ρ_R",                           units = "kg m⁻³"),
    ρconsvar                         = (long_name = "reference density × conservative variable",       units = "kg m⁻³ × consvar"),
    ρq                               = (long_name = "reference density × composition",                 units = "kg m⁻³"),
    specific_entropy                 = (long_name = "specific entropy",                                units = "J kg⁻¹ K⁻¹"),
    potential_temperature            = (long_name = "potential temperature",                           units = "K"),
    pressure                         = (long_name = "pressure, p_R + ρ_R φ",                           units = "Pa"),
    temperature                      = (long_name = "temperature (at the reference pressure)",         units = "K"),
    conjugate_variable               = (long_name = "conjugate variable of consvar, ∂h/∂consvar",      units = "J kg⁻¹ / consvar"),
    modified_chemical_potential      = (long_name = "modified chemical potential, ∂h/∂q",              units = "J kg⁻¹"),
    forcing_potential                = (long_name = "potential of the momentum forcing",               units = "J kg⁻¹"),
    buoyancy                         = (long_name = "buoyancy",                                        units = "m s⁻²"),
    kinematic_pressure_anomaly       = (long_name = "kinematic pressure anomaly (p - p_R)/ρ_R",        units = "J kg⁻¹"),
    vorticity                        = (long_name = "vorticity ∂u/∂z - ∂w/∂x",                         units = "s⁻¹"),
    divergence                       = (long_name = "velocity divergence ∂u/∂x + ∂w/∂z",               units = "s⁻¹"),
    mass_divergence                  = (long_name = "mass divergence ∇·(ρ_R u)",                       units = "kg m⁻³ s⁻¹"),
    kinetic_energy                   = (long_name = "kinetic energy per unit mass",                    units = "J kg⁻¹"),
    ∂u_x                             = (long_name = "∂u/∂x",                                           units = "s⁻¹"),
    ∂w_z                             = (long_name = "∂w/∂z",                                           units = "s⁻¹"),
    ∂u_z                             = (long_name = "∂u/∂z",                                           units = "s⁻¹"),
    ∂w_x                             = (long_name = "∂w/∂x",                                           units = "s⁻¹"),
    Jcons_x                          = (long_name = "diffusive flux of consvar along x",               units = "kg m⁻² s⁻¹ × consvar"),
    Jcons_z                          = (long_name = "diffusive flux of consvar along z",               units = "kg m⁻² s⁻¹ × consvar"),
    Jq_x                             = (long_name = "diffusive flux of composition along x",           units = "kg m⁻² s⁻¹"),
    Jq_z                             = (long_name = "diffusive flux of composition along z",           units = "kg m⁻² s⁻¹"),
    viscous_dissipation              = (long_name = "viscous dissipation",                             units = "W m⁻³"),
    conservative_variable_production = (long_name = "irreversible production of consvar",              units = "kg m⁻³ s⁻¹ × consvar"),
    div_Jcons                        = (long_name = "divergence of the diffusive flux of consvar",     units = "kg m⁻³ s⁻¹ × consvar"),
    div_Jq                           = (long_name = "divergence of the diffusive flux of composition", units = "kg m⁻³ s⁻¹"),
    dρconsvar                        = (long_name = "tendency of ρ_R consvar",                         units = "kg m⁻³ s⁻¹ × consvar"),
    dρq                              = (long_name = "tendency of ρ_R q",                               units = "kg m⁻³ s⁻¹"),
    du                               = (long_name = "tendency of u",                                   units = "m s⁻²"),
    dw                               = (long_name = "tendency of w",                                   units = "m s⁻²"),
)

"""
    describe_diagnostics(model::AN2D)

Print the name, long name and units of each diagnostic.
"""
describe_diagnostics(model::AN2D) = foreach(pairs(diagnostic_info(model))) do (name, (; long_name, units))
    println(rpad(name, 34), rpad(long_name, 52), units)
end

# a profile f(z) (e.g. ρ_R) as an array on the cells of `like`
reference_field(f, like) = [f[j] for i in axes(like, 1), j in axes(like, 2)]

# pressure p = p_R + ρ_R φ on the cells
pressure_field((; p_R, ρ_R), φ) = [p_R[j] + ρ_R[j] * φ[i, j] for i in axes(φ, 1), j in axes(φ, 2)]

# mass fluxes (ρ_R u, ρ_R w) on the faces, with ρ_R averaged to the w-faces
function mass_fluxes(model, (; u, w))
    (; ρ_R) = model.reference
    ρu = [ρ_R[j] * u[I, j] for I in axes(u, 1), j in axes(u, 2)]
    ρw = zero(w)
    for i in axes(w, 1), J in Zrange(model.domain)
        ρw[i, J] = avg_Z(ρ_R, J) * w[i, J]
    end
    return ρu, ρw
end

# divergence of a flux (Ax, Az) on the faces, at cell centres: δ_x Ax / Δx + δ_z Az / Δz on the interior cells, with
# the halo filled by mirroring (periodic copy in periodic directions)
function divergence_field((; domain, inv_dx, inv_dz), Ax, Az)
    div_A = zeros(eltype(Ax), xdim(domain), zdim(domain))
    for i in xrange_interior(domain), j in zrange_interior(domain)
        div_A[i, j] = inv_dx * dif_x(Ax, i, j) + inv_dz * dif_z(Az, i, j)
    end
    return mirror_halo!(domain, div_A)
end
