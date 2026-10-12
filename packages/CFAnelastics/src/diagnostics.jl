## DIAGNOSTICS
#
# Every diagnostic is the full array, halo included, with the halo defined (copied from the bounded state and the
# tendency scratch, or filled by the boundary conditions), so that post-processing can difference it at the boundaries.
# Names are shared with the FC2D diagnostics (CFCompressible); `diagnostic_info` gives the long name, units and grid
# location (:xz cells, :Xz u-faces, :xZ w-faces, :XZ vertices) of each. For plotting, `interior(domain, field)` strips
# the halo on any location and `coordinates(domain, (dx, dz), location)` gives the coordinates of the points.

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
    out(name) = get!(() -> alloc(F, domain, info[name].location), cache, name)::Matrix{F}
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
        pressure                         = (scratch, model) -> pressure!(out(:pressure), model, scratch.momentum.φ),
        temperature                      = scratch -> scratch.thermo.T,
        conjugate_variable               = scratch -> scratch.thermo.conjvar,
        modified_chemical_potential      = scratch -> scratch.thermo.chempot,
        forcing_potential                = scratch -> scratch.momentum.B,
        buoyancy                         = scratch -> scratch.thermo.b,
        kinematic_pressure_anomaly       = scratch -> scratch.momentum.φ,
        # kinematics
        vorticity                        = scratch -> pointwise!(out(:vorticity), mgr, -, scratch.derivatives.∂u_z, scratch.derivatives.∂w_x),
        divergence                       = scratch -> pointwise!(out(:divergence), mgr, +, scratch.derivatives.∂u_x, scratch.derivatives.∂w_z),
        mass_divergence                  = (state, model) -> mass_divergence!(out(:mass_divergence), model, state),
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
        div_Jcons                        = (scratch, model) -> divergence_field!(out(:div_Jcons), model, scratch.irreversible.Jcons_x, scratch.irreversible.Jcons_z),
        div_Jq                           = (scratch, model) -> divergence_field!(out(:div_Jq), model, scratch.irreversible.Jq_x, scratch.irreversible.Jq_z),
        # tendencies
        dρconsvar                        = dstate -> view(dstate.m, :, :, 1),
        dρq                              = dstate -> view(dstate.m, :, :, 2),
        du                               = dstate -> dstate.u,
        dw                               = dstate -> dstate.w,
    )
end

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
    conservative_variable            = (long_name = "conservative variable",                           units = "consvar",              location = :xz),
    composition                      = (long_name = "composition (solute mass fraction)",              units = "kg/kg",                location = :xz),
    u                                = (long_name = "horizontal velocity",                             units = "m s⁻¹",                location = :Xz),
    w                                = (long_name = "vertical velocity",                               units = "m s⁻¹",                location = :xZ),
    density                          = (long_name = "reference density ρ_R",                           units = "kg m⁻³",               location = :xz),
    ρconsvar                         = (long_name = "reference density × conservative variable",       units = "kg m⁻³ × consvar",     location = :xz),
    ρq                               = (long_name = "reference density × composition",                 units = "kg m⁻³",               location = :xz),
    specific_entropy                 = (long_name = "specific entropy",                                units = "J kg⁻¹ K⁻¹",           location = :xz),
    potential_temperature            = (long_name = "potential temperature",                           units = "K",                    location = :xz),
    pressure                         = (long_name = "pressure, p_R + ρ_R φ",                           units = "Pa",                   location = :xz),
    temperature                      = (long_name = "temperature (at the reference pressure)",         units = "K",                    location = :xz),
    conjugate_variable               = (long_name = "conjugate variable of consvar, ∂h/∂consvar",      units = "J kg⁻¹ / consvar",     location = :xz),
    modified_chemical_potential      = (long_name = "modified chemical potential, ∂h/∂q",              units = "J kg⁻¹",               location = :xz),
    forcing_potential                = (long_name = "potential of the momentum forcing",               units = "J kg⁻¹",               location = :xz),
    buoyancy                         = (long_name = "buoyancy",                                        units = "m s⁻²",                location = :xz),
    kinematic_pressure_anomaly       = (long_name = "kinematic pressure anomaly (p - p_R)/ρ_R",        units = "J kg⁻¹",               location = :xz),
    vorticity                        = (long_name = "vorticity ∂u/∂z - ∂w/∂x",                         units = "s⁻¹",                  location = :XZ),
    divergence                       = (long_name = "velocity divergence ∂u/∂x + ∂w/∂z",               units = "s⁻¹",                  location = :xz),
    mass_divergence                  = (long_name = "mass divergence ∇·(ρ_R u)",                       units = "kg m⁻³ s⁻¹",           location = :xz),
    kinetic_energy                   = (long_name = "kinetic energy per unit mass",                    units = "J kg⁻¹",               location = :xz),
    ∂u_x                             = (long_name = "∂u/∂x",                                           units = "s⁻¹",                  location = :xz),
    ∂w_z                             = (long_name = "∂w/∂z",                                           units = "s⁻¹",                  location = :xz),
    ∂u_z                             = (long_name = "∂u/∂z",                                           units = "s⁻¹",                  location = :XZ),
    ∂w_x                             = (long_name = "∂w/∂x",                                           units = "s⁻¹",                  location = :XZ),
    Jcons_x                          = (long_name = "diffusive flux of consvar along x",               units = "kg m⁻² s⁻¹ × consvar", location = :Xz),
    Jcons_z                          = (long_name = "diffusive flux of consvar along z",               units = "kg m⁻² s⁻¹ × consvar", location = :xZ),
    Jq_x                             = (long_name = "diffusive flux of composition along x",           units = "kg m⁻² s⁻¹",           location = :Xz),
    Jq_z                             = (long_name = "diffusive flux of composition along z",           units = "kg m⁻² s⁻¹",           location = :xZ),
    viscous_dissipation              = (long_name = "viscous dissipation",                             units = "W m⁻³",                location = :xz),
    conservative_variable_production = (long_name = "irreversible production of consvar",              units = "kg m⁻³ s⁻¹ × consvar", location = :xz),
    div_Jcons                        = (long_name = "divergence of the diffusive flux of consvar",     units = "kg m⁻³ s⁻¹ × consvar", location = :xz),
    div_Jq                           = (long_name = "divergence of the diffusive flux of composition", units = "kg m⁻³ s⁻¹",           location = :xz),
    dρconsvar                        = (long_name = "tendency of ρ_R consvar",                         units = "kg m⁻³ s⁻¹ × consvar", location = :xz),
    dρq                              = (long_name = "tendency of ρ_R q",                               units = "kg m⁻³ s⁻¹",           location = :xz),
    du                               = (long_name = "tendency of u",                                   units = "m s⁻²",                location = :Xz),
    dw                               = (long_name = "tendency of w",                                   units = "m s⁻²",                location = :xZ),
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
        @vec for i in irange, j in jrange
            out[i, j] = f(a[i, j], b[i, j])
        end
    end
    return out
end
function pointwise!(out, mgr, f::Fun, a, b, c) where Fun
    @with mgr, let (irange, jrange) = (axes(out, 1), axes(out, 2))
        @vec for i in irange, j in jrange
            out[i, j] = f(a[i, j], b[i, j], c[i, j])
        end
    end
    return out
end

# reference density ρ_R(z) on the cells (a plain loop: @vec cannot store a value independent of i)
function density!(ρ, (; mgr, reference))
    (; ρ_R) = reference
    @with mgr, let (irange, jrange) = (axes(ρ, 1), axes(ρ, 2))
        for i in irange, j in jrange
            ρ[i, j] = ρ_R[j]
        end
    end
    return ρ
end

# pressure p = p_R + ρ_R φ on the cells
function pressure!(p, (; mgr, reference), φ)
    (; p_R, ρ_R) = reference
    @with mgr, let (irange, jrange) = (axes(p, 1), axes(p, 2))
        @vec for i in irange, j in jrange
            p[i, j] = p_R[j] + ρ_R[j] * φ[i, j]
        end
    end
    return p
end

# mass divergence ∇·(ρ_R u) at the cells: δ_x(ρ_R u) / Δx + δ_z(ρ_R w) / Δz, with ρ_R averaged to the w-faces, on the
# interior cells; halo filled by mirroring (periodic copy in periodic directions)
function mass_divergence!(div, (; mgr, domain, reference, inv_dx, inv_dz), (; u, w))
    (; ρ_R) = reference
    @with mgr, let (irange, jrange) = (xrange_interior(domain), zrange_interior(domain))
        @vec for i in irange, j in jrange
            div[i, j] = inv_dx * ρ_R[j] * dif_x(u, i, j) + inv_dz * (avg_Z(ρ_R, j + 1) * w[i, j + 1] - avg_Z(ρ_R, j) * w[i, j])
        end
    end
    return mirror_halo!(domain, div)
end

# divergence of a vector (Vx, Vz) on the faces, at the cells: δ_x Vx / Δx + δ_z Vz / Δz on the interior cells; halo
# filled by mirroring (periodic copy in periodic directions)
function divergence_field!(div, (; mgr, domain, inv_dx, inv_dz), Vx, Vz)
    @with mgr, let (irange, jrange) = (xrange_interior(domain), zrange_interior(domain))
        @vec for i in irange, j in jrange
            div[i, j] = inv_dx * dif_x(Vx, i, j) + inv_dz * dif_z(Vz, i, j)
        end
    end
    return mirror_halo!(domain, div)
end
