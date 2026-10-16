## DIAGNOSTICS
#
# Every diagnostic is a full array with a defined halo (from the state and the tendency scratch, or a zero-gradient
# halo), on the grid location given by `diagnostic_info` ((Cell(), Cell()) cells, (Face(), Cell()) u-faces, (Cell(), Face()) w-faces, (Face(), Face()) vertices). Names
# are shared with FC2D. Derived fields use the operators of `model.spatial_scheme`. For plotting,
# `interior(domain, field)` strips the halo and `coordinates(domain, (dx, dz), location)` gives the points.

"""
    book = diagnostics(model::AN2D)

`CookBook` of the AN2D diagnostics, opened with `open(book; state, dstate, scratch, model)`. Names, long names and
units: see `diagnostic_info` (or `describe_diagnostics(model)`). Debugging intermediates: `internal_diagnostics`.

Build the book once and reuse it: it owns the output array of each computed diagnostic, allocated on its grid location
on first use, which every session of the book overwrites. The other diagnostics are the arrays (or views) of the
state, tendency and scratch. Copy a result to keep it beyond the next session.
"""
diagnostics(model::AN2D{F}) where F = let (; mgr, domain) = model, info = diagnostic_info(model), cache = Dict{Symbol, Any}()
    # output array of the computed diagnostic `name`, on its grid location (see `diagnostic_info`), allocated on first use
    out(name) = get!(() -> alloc(F, mgr, domain, info[name].location), cache, name)::Matrix{F}
    CookBook(;
        # state
        conservative_variable            = scratch -> scratch.thermo.consvar,
        composition                      = scratch -> scratch.thermo.q,
        u                                = state -> state.u,
        w                                = state -> state.w,
        density                          = model -> density!(out(:density), model),
        ρconsvar                         = state -> view(state.m, :, :, 1),
        ρq                               = state -> view(state.m, :, :, 2),
        # thermodynamics
        specific_entropy                 = (model, pressure, conservative_variable, composition) -> pointwise!(out(:specific_entropy), mgr, model.fluid(:p, :consvar, :q).specific_entropy, pressure, conservative_variable, composition),
        potential_temperature            = (model, pressure, conservative_variable, composition) -> pointwise!(out(:potential_temperature), mgr, model.fluid(:p, :consvar, :q).potential_temperature, pressure, conservative_variable, composition),
        pressure                         = (scratch, model) -> total_pressure!(out(:pressure), model, scratch.φ),
        temperature                      = scratch -> scratch.thermo.T,
        conjugate_variable               = scratch -> scratch.thermo.conjvar,
        modified_chemical_potential      = scratch -> scratch.thermo.chempot,
        forcing_potential                = scratch -> scratch.thermo.B,
        buoyancy                         = (model, conservative_variable, composition) -> buoyancy_field!(out(:buoyancy), model, conservative_variable, composition),
        kinematic_pressure_anomaly       = scratch -> scratch.φ,
        # kinematics
        vorticity                        = scratch -> pointwise!(out(:vorticity), mgr, -, scratch.derivatives.∂u_z, scratch.derivatives.∂w_x),
        divergence                       = scratch -> pointwise!(out(:divergence), mgr, +, scratch.derivatives.∂u_x, scratch.derivatives.∂w_z),
        mass_flux_x                      = (state, model) -> mass_flux_x!(out(:mass_flux_x), model, state),
        mass_flux_z                      = (state, model) -> mass_flux_z!(out(:mass_flux_z), model, state),
        mass_divergence                  = (mass_flux_x, mass_flux_z, model) -> divergence_field!(out(:mass_divergence), model, mass_flux_x, mass_flux_z),
        kinetic_energy                   = scratch -> scratch.advection.K,
        ∂u_x                             = scratch -> scratch.derivatives.∂u_x,
        ∂w_z                             = scratch -> scratch.derivatives.∂w_z,
        ∂u_z                             = scratch -> scratch.derivatives.∂u_z,
        ∂w_x                             = scratch -> scratch.derivatives.∂w_x,
        # irreversible processes
        consflux_x                       = scratch -> scratch.irreversible.consflux_x,
        consflux_z                       = scratch -> scratch.irreversible.consflux_z,
        qflux_x                          = scratch -> scratch.irreversible.qflux_x,
        qflux_z                          = scratch -> scratch.irreversible.qflux_z,
        viscous_dissipation              = scratch -> scratch.irreversible.ρε,
        conservative_variable_production = scratch -> scratch.irreversible.σ_cons,
        div_consflux                     = (scratch, model) -> divergence_field!(out(:div_consflux), model, scratch.irreversible.consflux_x, scratch.irreversible.consflux_z),
        div_qflux                        = (scratch, model) -> divergence_field!(out(:div_qflux), model, scratch.irreversible.qflux_x, scratch.irreversible.qflux_z),
        # tendencies
        dρconsvar                        = dstate -> view(dstate.m, :, :, 1),
        dρq                              = dstate -> view(dstate.m, :, :, 2),
        du                               = dstate -> dstate.u,
        dw                               = dstate -> dstate.w,
    )
end

"""
    book = internal_diagnostics(model::AN2D)

`CookBook` of intermediates of the AN2D tendencies, for debugging: the vortex force ω×u = (vortex_x, vortex_z) of the
advection, the advective fluxes of the tracers mu, mw (third index: tracer).
"""
internal_diagnostics(::AN2D) = CookBook(;
    vortex_x = scratch -> scratch.advection.vortex_x,
    vortex_z = scratch -> scratch.advection.vortex_z,
    mu       = scratch -> scratch.advection.mu,
    mw       = scratch -> scratch.advection.mw,
)

"""
    info = diagnostic_info(model::AN2D)

Long name, units and grid location of each diagnostic of `diagnostics(model)`, as a named tuple of
`(; long_name, units, location)`.
"""
diagnostic_info(::AN2D) = (;
    conservative_variable            = (long_name = "conservative variable",                           units = "consvar",              location = (Cell(), Cell())),
    composition                      = (long_name = "composition (solute mass fraction)",              units = "kg/kg",                location = (Cell(), Cell())),
    u                                = (long_name = "horizontal velocity",                             units = "m s⁻¹",                location = (Face(), Cell())),
    w                                = (long_name = "vertical velocity",                               units = "m s⁻¹",                location = (Cell(), Face())),
    density                          = (long_name = "reference density ρ_R",                           units = "kg m⁻³",               location = (Cell(), Cell())),
    ρconsvar                         = (long_name = "reference density × conservative variable",       units = "kg m⁻³ × consvar",     location = (Cell(), Cell())),
    ρq                               = (long_name = "reference density × composition",                 units = "kg m⁻³",               location = (Cell(), Cell())),
    specific_entropy                 = (long_name = "specific entropy",                                units = "J kg⁻¹ K⁻¹",           location = (Cell(), Cell())),
    potential_temperature            = (long_name = "potential temperature",                           units = "K",                    location = (Cell(), Cell())),
    pressure                         = (long_name = "pressure, p_R + ρ_R φ",                           units = "Pa",                   location = (Cell(), Cell())),
    temperature                      = (long_name = "temperature (at the reference pressure)",         units = "K",                    location = (Cell(), Cell())),
    conjugate_variable               = (long_name = "conjugate variable of consvar, ∂h/∂consvar",      units = "J kg⁻¹ / consvar",     location = (Cell(), Cell())),
    modified_chemical_potential      = (long_name = "modified chemical potential, ∂h/∂q",              units = "J kg⁻¹",               location = (Cell(), Cell())),
    forcing_potential                = (long_name = "potential of the momentum forcing",               units = "J kg⁻¹",               location = (Cell(), Cell())),
    buoyancy                         = (long_name = "buoyancy",                                        units = "m s⁻²",                location = (Cell(), Cell())),
    kinematic_pressure_anomaly       = (long_name = "kinematic pressure anomaly (p - p_R)/ρ_R",        units = "J kg⁻¹",               location = (Cell(), Cell())),
    vorticity                        = (long_name = "vorticity ∂u/∂z - ∂w/∂x",                         units = "s⁻¹",                  location = (Face(), Face())),
    divergence                       = (long_name = "velocity divergence ∂u/∂x + ∂w/∂z",               units = "s⁻¹",                  location = (Cell(), Cell())),
    mass_flux_x                      = (long_name = "mass flux ρ_R u along x",                         units = "kg m⁻² s⁻¹",           location = (Face(), Cell())),
    mass_flux_z                      = (long_name = "mass flux ρ_R w along z",                         units = "kg m⁻² s⁻¹",           location = (Cell(), Face())),
    mass_divergence                  = (long_name = "mass divergence ∇·(ρ_R u)",                       units = "kg m⁻³ s⁻¹",           location = (Cell(), Cell())),
    kinetic_energy                   = (long_name = "kinetic energy per unit mass",                    units = "J kg⁻¹",               location = (Cell(), Cell())),
    ∂u_x                             = (long_name = "∂u/∂x",                                           units = "s⁻¹",                  location = (Cell(), Cell())),
    ∂w_z                             = (long_name = "∂w/∂z",                                           units = "s⁻¹",                  location = (Cell(), Cell())),
    ∂u_z                             = (long_name = "∂u/∂z",                                           units = "s⁻¹",                  location = (Face(), Face())),
    ∂w_x                             = (long_name = "∂w/∂x",                                           units = "s⁻¹",                  location = (Face(), Face())),
    consflux_x                       = (long_name = "diffusive flux of consvar along x",               units = "kg m⁻² s⁻¹ × consvar", location = (Face(), Cell())),
    consflux_z                       = (long_name = "diffusive flux of consvar along z",               units = "kg m⁻² s⁻¹ × consvar", location = (Cell(), Face())),
    qflux_x                          = (long_name = "diffusive flux of composition along x",           units = "kg m⁻² s⁻¹",           location = (Face(), Cell())),
    qflux_z                          = (long_name = "diffusive flux of composition along z",           units = "kg m⁻² s⁻¹",           location = (Cell(), Face())),
    viscous_dissipation              = (long_name = "viscous dissipation",                             units = "W m⁻³",                location = (Cell(), Cell())),
    conservative_variable_production = (long_name = "irreversible production of consvar",              units = "kg m⁻³ s⁻¹ × consvar", location = (Cell(), Cell())),
    div_consflux                     = (long_name = "divergence of the diffusive flux of consvar",     units = "kg m⁻³ s⁻¹ × consvar", location = (Cell(), Cell())),
    div_qflux                        = (long_name = "divergence of the diffusive flux of composition", units = "kg m⁻³ s⁻¹",           location = (Cell(), Cell())),
    dρconsvar                        = (long_name = "tendency of ρ_R consvar",                         units = "kg m⁻³ s⁻¹ × consvar", location = (Cell(), Cell())),
    dρq                              = (long_name = "tendency of ρ_R q",                               units = "kg m⁻³ s⁻¹",           location = (Cell(), Cell())),
    du                               = (long_name = "tendency of u",                                   units = "m s⁻²",                location = (Face(), Cell())),
    dw                               = (long_name = "tendency of w",                                   units = "m s⁻²",                location = (Cell(), Face())),
)

"""
    describe_diagnostics(model::AN2D)

Print the name, long name and units of each diagnostic.
"""
describe_diagnostics(model::AN2D) = foreach(pairs(diagnostic_info(model))) do (name, (; long_name, units))
    println(rpad(name, 34), rpad(long_name, 52), units)
end

# out = f(a, b[, c]) pointwise on arrays of the same grid, halo included
function pointwise!(out, mgr, f::Fun, a, b) where Fun
    @with mgr, let (irange, jrange) = (axes(out, 1), axes(out, 2))
        @inbounds @vec for i in irange, j in jrange
            out[i, j] = f(a[i, j], b[i, j])
        end
    end
    return out
end
function pointwise!(out, mgr, f::Fun, a, b, c) where Fun
    @with mgr, let (irange, jrange) = (axes(out, 1), axes(out, 2))
        @inbounds @vec for i in irange, j in jrange
            out[i, j] = f(a[i, j], b[i, j], c[i, j])
        end
    end
    return out
end

# reference density ρ_R(z) on the cells (a plain loop, i innermost: @vec cannot store a value independent of i)
function density!(ρ, (; mgr, reference))
    (; ρ_R) = reference
    @with mgr, let (irange, jrange) = (axes(ρ, 1), axes(ρ, 2))
        @inbounds for j in jrange, i in irange
            ρ[i, j] = ρ_R[j]
        end
    end
    return ρ
end

# buoyancy of the buoyancy scheme on the cells, from the specific volume v(p_R, consvar, q)
function buoyancy_field!(b, model, consvar, q)
    (; mgr, fluid, reference, space, buoyancy_scheme) = model
    (; ρ_R, p_R) = reference
    (; g) = space
    @with mgr, let (irange, jrange) = (axes(b, 1), axes(b, 2))
        @inbounds @vec for i in irange, j in jrange
            b[i, j] = buoyancy(buoyancy_scheme, g, ρ_R[j], fluid(:p, :consvar, :q).specific_volume(p_R[j], consvar[i, j], q[i, j]))
        end
    end
    return b
end

# pressure p = p_R + ρ_R φ on the cells
function total_pressure!(p, (; mgr, reference), φ)
    (; p_R, ρ_R) = reference
    @with mgr, let (irange, jrange) = (axes(p, 1), axes(p, 2))
        @inbounds @vec for i in irange, j in jrange
            p[i, j] = p_R[j] + ρ_R[j] * φ[i, j]
        end
    end
    return p
end

# mass flux ρ_R u on the u-faces (ρ_R depends on z only)
function mass_flux_x!(flux, (; mgr, reference), (; u))
    (; ρ_R) = reference
    @with mgr, let (irange, jrange) = (axes(flux, 1), axes(flux, 2))
        @inbounds @vec for I in irange, j in jrange
            flux[I, j] = ρ_R[j] * u[I, j]
        end
    end
    return flux
end
# mass flux ρ_R w on the w-faces, ρ_R averaged to the face (the lowest halo row of faces, without a cell below, is zero)
function mass_flux_z!(flux, model, (; w))
    (; mgr, domain, reference) = model
    (; ρ_R) = reference
    (; ℑ) = operators(model.spatial_scheme)
    @with mgr, let (irange, jrange) = (axes(flux, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            flux[i, J] = ℑ(ρ_R, (2,), (Face(),), J) * w[i, J]
        end
    end
    return flux
end

# divergence of a vector (Vx, Vz) on the faces, at the cells: δ_x Vx / Δx + δ_z Vz / Δz on the interior cells;
# zero-gradient halo
function divergence_field!(div, model, Vx, Vz)
    (; mgr, domain, inv_dx, inv_dz) = model
    (; δ) = operators(model.spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            div[i, j] = inv_dx * δ(Vx, (1,), (Cell(),), i, j) + inv_dz * δ(Vz, (2,), (Cell(),), i, j)
        end
    end
    return fill_halo!(domain, div, ZeroGradient())
end
