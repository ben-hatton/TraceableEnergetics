## TENDENCIES
#
# ∂ₜm = -∇·(m u) - ∇·J + σ ẑ_consvar,   ∂ₜu = forcing - ∇K + ω×u + F/ρ_R - ∇φ,   ∇·(ρ_R ∂ₜu) = 0,
# with m = ρ_R (consvar, q) at cell centres and (u, w) on faces. Averages and differences come from
# `operators(model.spatial_scheme)`.

function CFTimeSchemes.tendencies!(dstate, scratch, model::AN2D{F}, state, t) where {F}
    # fill the halo of the state (never its interior) from the boundary conditions
    state = state_bc!(model, state)

    ## THERMODYNAMICS; consvar, q, T, conjvar, chempot, Cp, ∂consvar/∂q, Γ, B
    (; consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B) = scratch.thermo

    consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B =
        thermodynamics!((consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B), model, state)
    thermo = (; consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B)

    ## KINEMATICS; ∂u/∂x, ∂w/∂z, ∂u/∂z, ∂w/∂x, K = (u² + w²) / 2
    (; ∂u_x, ∂w_z, ∂u_z, ∂w_x) = scratch.derivatives
    (; K) = scratch.advection

    ∂u_x, ∂w_z, ∂u_z, ∂w_x, K = kinematics!((∂u_x, ∂w_z, ∂u_z, ∂w_x, K), model, state)

    ## ADVECTION; tracer fluxes m u, vortex force ω×u = (ωw, -ωu)
    (; mu, mw, ωw, ωu, vortex_scratch) = scratch.advection

    mu, mw                 = advective_flux!((mu, mw), model, state)
    ωw, ωu, vortex_scratch = vortex_force!((ωw, ωu, vortex_scratch), model, model.vortex_scheme, state, (∂u_z, ∂w_x))

    ## VISCOUS STRESS; u·τ and F = ∇·τ on faces
    (; uτ_x, uτ_z, F_x, F_z) = scratch.irreversible

    uτ_x, uτ_z, F_x, F_z = viscous_stress!((uτ_x, uτ_z, F_x, F_z), model, state, (∂u_x, ∂w_z, ∂u_z, ∂w_x))

    ## IRREVERSIBLE FLUXES of consvar and q, with the wall conditions, and the diffusive heating on faces
    (; consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z) = scratch.irreversible

    consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z =
        fluxes!((consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z), model, thermo)

    ## DISSIPATIVE HEATING; ρε, σ_cons
    (; ρε, σ_cons) = scratch.irreversible

    ρε, σ_cons = dissipative_heating!((ρε, σ_cons), model, model.thermoflux, state, (uτ_x, uτ_z, F_x, F_z), thermo,
                                      (diffusive_heating_x, diffusive_heating_z))

    ## MOMENTUM; velocity tendencies before the pressure gradient
    du, dw = velocity_tendencies!((dstate.u, dstate.w), model, model.buoyancy, state, thermo, (K, ωw, ωu, F_x, F_z))

    ## PRESSURE; φ = (p - p_R) / ρ_R such that ∇·(ρ_R (du - ∇φ)) = 0, with no flow through walls
    φ       = pressure!(alloc!(scratch.φ, F, model.domain, :xz), model.poisson, model.mgr, (du, dw))
    du, dw  = subtract_pressure_gradient!((du, dw), model, φ)

    ## TRACERS
    dm = dm!(dstate.m, model, state, (mu, mw, consflux_x, consflux_z, qflux_x, qflux_z, σ_cons))

    dstate = dstate_bc!(model, (m = dm, u = du, w = dw))

    scratch = (
        thermo          = thermo,
        derivatives     = (; ∂u_x, ∂w_z, ∂u_z, ∂w_x),
        advection       = (; K, mu, mw, ωw, ωu, vortex_scratch),
        irreversible    = (; uτ_x, uτ_z, F_x, F_z, ρε, consflux_x, consflux_z, qflux_x, qflux_z,
                             diffusive_heating_x, diffusive_heating_z, σ_cons),
        φ               = φ,
    )
    return dstate, scratch
end

## THERMODYNAMICS

# Pointwise thermodynamics at the reference pressure p_R, from m = ρ_R (consvar, q), on the interior cells: T, conjvar,
# chempot, Cp, ∂consvar/∂q, Γ = ρ_R ∂μ/∂p / ∂μ/∂q (barodiffusion) and B, the potential of the momentum forcing.
# Zero-gradient halo: every difference across a wall vanishes.
function thermodynamics!((consvar_, q_, T_, conjvar_, chempot_, Cp_, ∂consvar_∂q_, Γ_, B_), model::AN2D{F}, (; m)) where {F}
    (; mgr, domain, fluid, reference, space) = model
    (; ρ_R, p_R) = reference
    (; g) = space
    buoyancy_scheme = model.buoyancy

    consvar     = alloc!(consvar_, F, domain, :xz)
    q           = alloc!(q_, F, domain, :xz)
    T           = alloc!(T_, F, domain, :xz)
    conjvar     = alloc!(conjvar_, F, domain, :xz)
    chempot     = alloc!(chempot_, F, domain, :xz)
    Cp          = alloc!(Cp_, F, domain, :xz)
    ∂consvar_∂q = alloc!(∂consvar_∂q_, F, domain, :xz)
    Γ           = alloc!(Γ_, F, domain, :xz)
    B           = alloc!(B_, F, domain, :xz)

    @with mgr, let (irange, jrange) = (xrange_interior(domain), zrange_interior(domain))
        @inbounds @vec for i in irange, j in jrange
            v_R               = inv(ρ_R[j])
            cons, qij         = v_R * m[i, j, 1], v_R * m[i, j, 2]
            Tij               = fluid(:p, :consvar, :q).temperature(p_R[j], cons, qij)
            conj              = fluid(:p, :consvar, :q).conjugate_variable(p_R[j], cons, qij)
            chem              = fluid(:p, :consvar, :q).modified_chemical_potential(p_R[j], cons, qij)
            _, ∂μ_∂p, _, ∂μ_∂q = fluid(:p, :T, :q).chemical_potential_derivatives(p_R[j], Tij, qij)
            consvar[i, j], q[i, j], T[i, j] = cons, qij, Tij
            conjvar[i, j], chempot[i, j] = conj, chem
            Cp[i, j]          = fluid(:p, :consvar, :q).heat_capacity(p_R[j], cons, qij)
            ∂consvar_∂q[i, j] = fluid(:p, :consvar, :q).dcons_dq(p_R[j], cons, qij)
            Γ[i, j]           = ρ_R[j] * ∂μ_∂p / ∂μ_∂q
            B[i, j]           = forcing_potential(buoyancy_scheme, fluid, g, ρ_R[j], p_R[j], cons, qij, conj, chem)
        end
    end

    fill_halo!((consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B), domain, ZeroGradient())

    return consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B
end

# buoyancy b at a level of reference density ρ_R, from the specific volume v of the fluid
@inline buoyancy(::BuoyDynOld, g, ρ_R, v) = g * (1 - inv(ρ_R * v))
@inline buoyancy(::Union{BuoyDynNew, BuoyThermo}, g, ρ_R, v) = g * (ρ_R * v - 1)

# potential B of the momentum forcing at reference density ρ_R and pressure p: h - conjvar consvar - chempot q
# (thermodynamic form), or the buoyancy b (dynamic forms)
@inline forcing_potential(::BuoyThermo, fluid, g, ρ_R, p, cons, q, conj, chem) = fluid(:p, :consvar, :q).specific_enthalpy(p, cons, q) - conj * cons - chem * q
@inline forcing_potential(scheme::BuoyDyn, fluid, g, ρ_R, p, cons, q, conj, chem) = buoyancy(scheme, g, ρ_R, fluid(:p, :consvar, :q).specific_volume(p, cons, q))

## KINEMATICS

# cells: ∂u/∂x, ∂w/∂z and K = (avg_x u² + avg_z w²) / 2;  vertices: ∂u/∂z, ∂w/∂x
function kinematics!((∂u_x_, ∂w_z_, ∂u_z_, ∂w_x_, K_), model::AN2D{F}, (; u, w)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; avg_x, avg_z, dif_x, dif_z, dif_X, dif_Z) = operators(model.spatial_scheme)

    ∂u_x    = alloc!(∂u_x_, F, domain, :xz)
    ∂w_z    = alloc!(∂w_z_, F, domain, :xz)
    ∂u_z    = alloc!(∂u_z_, F, domain, :XZ)
    ∂w_x    = alloc!(∂w_x_, F, domain, :XZ)
    K       = alloc!(K_, F, domain, :xz)

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @inbounds @vec for i in irange, j in jrange
            ∂u_x[i, j] = inv_dx * dif_x(u, i, j)
            ∂w_z[i, j] = inv_dz * dif_z(w, i, j)
            K[i, j]    = 0.5 * ( avg_x(abs2, u, i, j) + avg_z(abs2, w, i, j) )
        end
    end
    # each vertex derivative over the full axis along which it is not differenced
    @with mgr, let (irange, jrange) = (axes(∂u_z, 1), Zrange(domain))
        @inbounds @vec for I in irange, J in jrange
            ∂u_z[I, J] = inv_dz * dif_Z(u, I, J)
        end
    end
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(∂w_x, 2))
        @inbounds @vec for I in irange, J in jrange
            ∂w_x[I, J] = inv_dx * dif_X(w, I, J)
        end
    end

    periodize!(model, (∂u_x, ∂w_z, ∂u_z, ∂w_x, K))

    return ∂u_x, ∂w_z, ∂u_z, ∂w_x, K
end

## ADVECTION

# tracer fluxes on faces: m u = u avg_x(m), m w = w avg_z(m) (third index: tracer)
function advective_flux!((mu_, mw_), model::AN2D{F}, (; m, u, w)) where {F}
    (; mgr, domain) = model
    (; avg_X, avg_Z) = operators(model.spatial_scheme)

    mu = alloc!(mu_, F, domain, :Xz, 2)
    mw = alloc!(mw_, F, domain, :xZ, 2)

    m1, m2      = eachslice(m; dims = 3)
    mu1, mu2    = eachslice(mu; dims = 3)
    mw1, mw2    = eachslice(mw; dims = 3)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(mu, 2))
        @inbounds @vec for I in irange, j in jrange
            mu1[I, j] = u[I, j] * avg_X(m1, I, j)
            mu2[I, j] = u[I, j] * avg_X(m2, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(mw, 1), Zrange(domain))
        @inbounds @vec for i in irange, J in jrange
            mw1[i, J] = w[i, J] * avg_Z(m1, i, J)
            mw2[i, J] = w[i, J] * avg_Z(m2, i, J)
        end
    end
    periodize!(model, (mu, mw))
    return mu, mw
end

# vortex force ω×u = (ωw, -ωu) on faces, ω = ∂u/∂z - ∂w/∂x at vertices; energy-conserving: products at vertices
function vortex_force!((ωw_, ωu_, (ωwᴵᴶ_, ωuᴵᴶ_)), model::AN2D{F}, ::VortexEnergyCons, (; u, w), (∂u_z, ∂w_x)) where {F}
    (; mgr, domain) = model
    (; ρ_R) = model.reference
    (; avg_x, avg_z, avg_X, avg_Z) = operators(model.spatial_scheme)

    ωw      = alloc!(ωw_, F, domain, :Xz)
    ωu      = alloc!(ωu_, F, domain, :xZ)
    ωwᴵᴶ    = alloc!(ωwᴵᴶ_, F, domain, :XZ)
    ωuᴵᴶ    = alloc!(ωuᴵᴶ_, F, domain, :XZ)

    # vertices: ω w and ω u (u weighted by ρ_R)
    @with mgr, let (irange, jrange) = (Xrange(domain), Zrange(domain))
        @inbounds @vec for I in irange, J in jrange
            ω = ∂u_z[I, J] - ∂w_x[I, J]
            ωwᴵᴶ[I, J] = ω * avg_X(w, I, J)
            ωuᴵᴶ[I, J] = ω * inv(avg_Z(ρ_R, J)) * avg_Z(ρ_R, u, I, J)
        end
    end
    periodize!(model, (ωwᴵᴶ, ωuᴵᴶ))

    # u-faces: ωw
    @with mgr, let (irange, jrange) = (axes(ωw, 1), zrange(domain))
        @inbounds @vec for I in irange, j in jrange
            ωw[I, j] = avg_z(ωwᴵᴶ, I, j)
        end
    end
    # w-faces: ωu
    @with mgr, let (irange, jrange) = (xrange(domain), axes(ωu, 2))
        @inbounds @vec for i in irange, J in jrange
            ωu[i, J] = avg_x(ωuᴵᴶ, i, J)
        end
    end

    periodize!(model, (ωw, ωu))

    return ωw, ωu, (ωwᴵᴶ, ωuᴵᴶ)
end
# enstrophy-conserving: ω/ρ_R and the mass flux averaged separately to the faces, then multiplied
function vortex_force!((ωw_, ωu_, (ωvᴵᴶ_, ρwᴵ_, ρuᴶ_)), model::AN2D{F}, ::VortexEnstrophyCons, (; u, w), (∂u_z, ∂w_x)) where {F}
    (; mgr, domain) = model
    (; ρ_R) = model.reference
    (; avg_x, avg_z, avg_X, avg_Z) = operators(model.spatial_scheme)

    ωw      = alloc!(ωw_, F, domain, :Xz)
    ωu      = alloc!(ωu_, F, domain, :xZ)
    ωvᴵᴶ    = alloc!(ωvᴵᴶ_, F, domain, :XZ)
    ρwᴵ     = alloc!(ρwᴵ_, F, domain, :XZ)
    ρuᴶ     = alloc!(ρuᴶ_, F, domain, :XZ)

    # vertices: ω / ρ_R and the mass fluxes ρ_R w, ρ_R u
    @with mgr, let (irange, jrange) = (Xrange(domain), Zrange(domain))
        @inbounds @vec for I in irange, J in jrange
            ω = ∂u_z[I, J] - ∂w_x[I, J]
            ωvᴵᴶ[I, J] = ω * inv(avg_Z(ρ_R, J))
            ρwᴵ[I, J] = avg_Z(ρ_R, J) * avg_X(w, I, J)
            ρuᴶ[I, J] = avg_Z(ρ_R, u, I, J)
        end
    end
    periodize!(model, (ωvᴵᴶ, ρuᴶ, ρwᴵ))

    # u-faces: ωw
    @with mgr, let (irange, jrange) = (axes(ωw, 1), zrange(domain))
        @inbounds @vec for I in irange, j in jrange
            ωw[I, j] = avg_z(ωvᴵᴶ, I, j) * avg_z(ρwᴵ, I, j)
        end
    end
    # w-faces: ωu
    @with mgr, let (irange, jrange) = (xrange(domain), axes(ωu, 2))
        @inbounds @vec for i in irange, J in jrange
            ωu[i, J] = avg_x(ωvᴵᴶ, i, J) * avg_x(ρuᴶ, i, J)
        end
    end
    periodize!(model, (ωw, ωu))
    return ωw, ωu, (ωvᴵᴶ, ρwᴵ, ρuᴶ)
end

## VISCOUS STRESS

# On faces: the viscous force F = ∇·τ (per unit volume) and the energy flux u·τ; ρε = ∇·(u·τ) - u·F is formed in
# `dissipative_heating!`
function viscous_stress!((uτ_x_, uτ_z_, F_x_, F_z_), model::AN2D{F}, (; u, w), (∂u_x, ∂w_z, ∂u_z, ∂w_x)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dyn_visc, bulk_visc) = model.viscosity
    (; avg_x, avg_z, avg_X, avg_Z, avg_xZ, avg_Xz, dif_x, dif_z, dif_X, dif_Z) = operators(model.spatial_scheme)

    uτ_x    = alloc!(uτ_x_, F, domain, :Xz)
    uτ_z    = alloc!(uτ_z_, F, domain, :xZ)
    F_x     = alloc!(F_x_, F, domain, :Xz)
    F_z     = alloc!(F_z_, F, domain, :xZ)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @inbounds @vec for I in irange, j in jrange
            divu_x  = avg_X(∂u_x, I, j) + avg_X(∂w_z, I, j)                   # ∇·u
            ∂divu_x = inv_dx * (dif_X(∂u_x, I, j) + dif_X(∂w_z, I, j))        # ∂ₓ(∇·u)
            ∂ω_z    = inv_dz * (dif_z(∂u_z, I, j) - dif_z(∂w_x, I, j))        # ∂_z ω
            # u·τ along x
            uτ_x[I, j] = ( u[I, j] * ( 2 * dyn_visc * avg_X(∂u_x, I, j) + ( bulk_visc - (2 / 3) * dyn_visc ) * divu_x )
                    + dyn_visc * avg_Xz(w, I, j) * ( avg_z(∂u_z, I, j) + avg_z(∂w_x, I, j) ) )
            # Fₓ = μ ∂_z ω + (μ_b + 4μ/3) ∂ₓ(∇·u)
            F_x[I, j] = dyn_visc * ∂ω_z + ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_x
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @inbounds @vec for i in irange, J in jrange
            divu_z  = avg_Z(∂u_x, i, J) + avg_Z(∂w_z, i, J)                   # ∇·u
            ∂divu_z = inv_dz * (dif_Z(∂u_x, i, J) + dif_Z(∂w_z, i, J))        # ∂_z(∇·u)
            ∂ω_x    = inv_dx * (dif_x(∂u_z, i, J) - dif_x(∂w_x, i, J))        # ∂ₓ ω
            # u·τ along z
            uτ_z[i, J] = ( w[i, J] * ( 2 * dyn_visc * avg_Z(∂w_z, i, J) + ( bulk_visc - (2 / 3) * dyn_visc ) * divu_z )
                    + dyn_visc * avg_xZ(u, i, J) * ( avg_x(∂u_z, i, J) + avg_x(∂w_x, i, J) ) )
            # F_z = -μ ∂ₓ ω + (μ_b + 4μ/3) ∂_z(∇·u)
            F_z[i, J] = - dyn_visc * ∂ω_x + ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_z
        end
    end

    periodize!(model, (uτ_x, uτ_z, F_x, F_z))

    return uτ_x, uτ_z, F_x, F_z
end

## IRREVERSIBLE FLUXES

# On faces: qflux = -ρ_R k_q (∂q - Γ ∂Φ) and consflux (carrying the heat flux -ρ_R Cp k_T ∂T, or down the gradient of
# consvar), coefficients averaged from the two cells, ∂Φ = g on w-faces; wall faces from `apply_tracer_bcs!`. Also the
# diffusive heating -(consflux ∂conjvar + qflux ∂chempot), zero on wall faces (zero-gradient halo of conjvar, chempot).
function fluxes!((consflux_x_, consflux_z_, qflux_x_, qflux_z_, diffusive_heating_x_, diffusive_heating_z_), model::AN2D{F}, thermo) where {F}
    (; mgr, domain, space, thermoflux, inv_dx, inv_dz) = model
    (; ρ_R) = model.reference
    (; g) = space
    (; consvar, q, conjvar, chempot, T, Cp, ∂consvar_∂q, Γ) = thermo
    (; avg_X, avg_Z, dif_X, dif_Z) = operators(model.spatial_scheme)

    consflux_x          = alloc!(consflux_x_, F, domain, :Xz)
    consflux_z          = alloc!(consflux_z_, F, domain, :xZ)
    qflux_x             = alloc!(qflux_x_, F, domain, :Xz)
    qflux_z             = alloc!(qflux_z_, F, domain, :xZ)
    diffusive_heating_x = alloc!(diffusive_heating_x_, F, domain, :Xz)
    diffusive_heating_z = alloc!(diffusive_heating_z_, F, domain, :xZ)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(consflux_x, 2))
        @inbounds @vec for I in irange, j in jrange
            qflux    = salt_flux(thermoflux, ρ_R[j], avg_X(Γ, I, j), inv_dx * dif_X(q, I, j), zero(g))
            consflux = consvar_flux(thermoflux, ρ_R[j], avg_X(Cp, I, j), inv_dx * dif_X(T, I, j), inv_dx * dif_X(consvar, I, j), qflux, avg_X(conjvar, I, j), avg_X(∂consvar_∂q, I, j))
            qflux_x[I, j], consflux_x[I, j] = qflux, consflux
            diffusive_heating_x[I, j] = - inv_dx * ( consflux * dif_X(conjvar, I, j) + qflux * dif_X(chempot, I, j) )
        end
    end
    @with mgr, let (irange, jrange) = (axes(consflux_z, 1), Zrange(domain))
        @inbounds @vec for i in irange, J in jrange
            qflux    = salt_flux(thermoflux, avg_Z(ρ_R, J), avg_Z(Γ, i, J), inv_dz * dif_Z(q, i, J), g)
            consflux = consvar_flux(thermoflux, avg_Z(ρ_R, J), avg_Z(Cp, i, J), inv_dz * dif_Z(T, i, J), inv_dz * dif_Z(consvar, i, J), qflux, avg_Z(conjvar, i, J), avg_Z(∂consvar_∂q, i, J))
            qflux_z[i, J], consflux_z[i, J] = qflux, consflux
            diffusive_heating_z[i, J] = - inv_dz * ( consflux * dif_Z(conjvar, i, J) + qflux * dif_Z(chempot, i, J) )
        end
    end

    apply_tracer_bcs!((consflux_x, qflux_x, consflux_z, qflux_z), model, thermo)
    periodize!(model, (consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z))

    return consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z
end

# composition flux -ρ k_q (∂q - Γ ∂Φ), with barodiffusion in the consistent scheme only
@inline salt_flux(hf::ThermoFluxConsistent, ρ, Γ, ∂q, ∂Φ) = - ρ * hf.k_q * (∂q - Γ * ∂Φ)
@inline salt_flux(hf::ThermoFluxDowngradient, ρ, Γ, ∂q, ∂Φ) = - ρ * hf.k_q * ∂q
# consvar flux carrying the heat flux JT: JT / conjvar (+ ∂consvar/∂q qflux in the consistent scheme)
@inline heat_to_consvar(::ThermoFluxConsistent, JT, qflux, conj, ∂consvar_∂q) = JT / conj + ∂consvar_∂q * qflux
@inline heat_to_consvar(::ThermoFluxDowngradient, JT, qflux, conj, ∂consvar_∂q) = JT / conj
# consvar flux: carrying -ρ Cp k_T ∂T (consistent), or -ρ k_T ∂consvar (downgradient)
@inline consvar_flux(hf::ThermoFluxConsistent, ρ, Cp, ∂T, ∂cons, qflux, conj, ∂consvar_∂q) =
    heat_to_consvar(hf, - ρ * Cp * hf.k_T * ∂T, qflux, conj, ∂consvar_∂q)
@inline consvar_flux(hf::ThermoFluxDowngradient, ρ, Cp, ∂T, ∂cons, qflux, conj, ∂consvar_∂q) = - ρ * hf.k_T * ∂cons

## TRACER BOUNDARY CONDITIONS

# On wall faces, the fluxes of consvar and q are set by the wall conditions, replacing those computed by `fluxes!`.
# Plain loops: about 2(Mx + Mz) faces, too few for vectorisation or threads to pay off.
function apply_tracer_bcs!((consflux_x, qflux_x, consflux_z, qflux_z), model::AN2D, thermo)
    left_right, bottom_top = model.boundary.walls
    apply_tracer_bcs!((consflux_x, qflux_x), left_right, Val(1), model, thermo)
    apply_tracer_bcs!((consflux_z, qflux_z), bottom_top, Val(2), model, thermo)
    return nothing
end
apply_tracer_bcs!(fluxes, ::Nothing, D, model, thermo) = nothing          # periodic direction: no walls
function apply_tracer_bcs!(fluxes, (lower, upper)::Tuple{Wall, Wall}, D, model, thermo)
    lo, hi = wall_sides(model.domain, D)                                   # wall face, adjacent cell, inward normal n
    apply_tracer_bcs!(fluxes, lower.tracers, lo, D, model, thermo)
    apply_tracer_bcs!(fluxes, upper.tracers, hi, D, model, thermo)
end
function apply_tracer_bcs!((consflux, qflux), (cons_bc, q_bc), side, D, model::AN2D, thermo)
    for I in slice(qflux, D)                                               # each position along the wall
        face, cell = across(I, D, side.face), across(I, D, side.cell)
        wall = wall_state(model, D, side, face, cell)                     # (; n, d, ρ, p, ∂Φ) on the wall face
        # q flux: Flux → n F;  Value → face flux with the wall value at distance d
        qflux[face] = q_flux_bc(q_bc, bc_value(q_bc, I), thermo, cell, wall, model)
        # consvar flux: Flux → n F;  HeatFlux → n F / conjvar (+ cross term);  Value → face flux with T_wall (uses qflux)
        consflux[face] = cons_flux_bc(cons_bc, bc_value(cons_bc, I), thermo, cell, wall, model, qflux[face])
    end
end

# wall face state: inward normal n, distance d to the cell centre, reference density ρ and pressure p on the face,
# and the normal gradient ∂Φ of the geopotential; on bottom/top walls, p extrapolated hydrostatically from the cell
@inline function wall_state(model::AN2D, ::Val{1}, side, face, cell)
    (; ρ_R, p_R) = model.reference
    j = cell[2]
    return (; n = side.n, d = model.dx / 2, ρ = ρ_R[j], p = p_R[j], ∂Φ = zero(model.dx))
end
@inline function wall_state(model::AN2D, ::Val{2}, side, face, cell)
    (; ρ_R, p_R) = model.reference
    (; g) = model.space
    (; avg_Z) = operators(model.spatial_scheme)
    j, d = cell[2], model.dz / 2
    return (; n = side.n, d, ρ = avg_Z(ρ_R, face[2]), p = p_R[j] + side.n * ρ_R[j] * g * d, ∂Φ = g)
end

# composition: prescribed flux F of ρq, or the face flux with the wall value q_w at distance d
@inline q_flux_bc(::Flux, F, thermo, cell, wall, model) = wall.n * F
@inline q_flux_bc(::Value, q_w, (; q, Γ), cell, wall, model) =
    salt_flux(model.thermoflux, wall.ρ, Γ[cell], wall.n * (q[cell] - q_w) / wall.d, wall.∂Φ)
# consvar: prescribed flux F of ρ consvar, prescribed heat flux F, or the face flux with the wall value cons_w at
# distance d and T_wall = T(p_wall, cons_w, q_cell). The wall delivers exactly the heat flux (HeatFlux, or Value with
# the consistent scheme): the production between the wall and the cell centre is inside consflux.
@inline cons_flux_bc(::Flux{V, :density}, F, thermo, cell, wall, model, qflux) where V = wall.n * F
@inline cons_flux_bc(::Flux{V, :heat}, F, (; conjvar, ∂consvar_∂q), cell, wall, model, qflux) where V =
    heat_to_consvar(model.thermoflux, wall.n * F, qflux, conjvar[cell], ∂consvar_∂q[cell])
@inline function cons_flux_bc(::Value, cons_w, (; consvar, q, T, conjvar, Cp, ∂consvar_∂q), cell, wall, model, qflux)
    T_w = model.fluid(:p, :consvar, :q).temperature(wall.p, cons_w, q[cell])
    return consvar_flux(model.thermoflux, wall.ρ, Cp[cell], wall.n * (T[cell] - T_w) / wall.d, wall.n * (consvar[cell] - cons_w) / wall.d,
                        qflux, conjvar[cell], ∂consvar_∂q[cell])
end

## DISSIPATIVE HEATING

# At interior cells: viscous heating ρε = ∇·(u·τ) - u·F, and the production of consvar
# σ = (ρε + avg(diffusive heating)) / conjvar. Wall faces carry no diffusive heating (zero-gradient halo): the
# energy a wall delivers is conjvar_cell consflux + chempot_cell qflux, see `cons_flux_bc`.
function dissipative_heating!((ρε_, σ_cons_), model::AN2D{F}, ::ThermoFluxConsistent, (; u, w), (uτ_x, uτ_z, F_x, F_z), (; conjvar),
                              (diffusive_heating_x, diffusive_heating_z)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; avg_x, avg_z, dif_x, dif_z) = operators(model.spatial_scheme)

    ρε      = alloc!(ρε_, F, domain, :xz)
    σ_cons  = alloc!(σ_cons_, F, domain, :xz)

    @with mgr, let (irange, jrange) = (xrange_interior(domain), zrange_interior(domain))
        @inbounds @vec for i in irange, j in jrange
            ρεij         = inv_dx * dif_x(uτ_x, i, j) + inv_dz * dif_z(uτ_z, i, j) - avg_x(u, F_x, i, j) - avg_z(w, F_z, i, j)
            ρε[i, j]     = ρεij
            σ_cons[i, j] = inv(conjvar[i, j]) * ( ρεij + avg_x(diffusive_heating_x, i, j) + avg_z(diffusive_heating_z, i, j) )
        end
    end

    fill_halo!((ρε, σ_cons), domain, ZeroGradient())

    return ρε, σ_cons
end
# downgradient scheme: no source of consvar (neither viscous nor diffusive heating)
function dissipative_heating!((ρε_, σ_cons_), model::AN2D{F}, ::ThermoFluxDowngradient, (; u, w), (uτ_x, uτ_z, F_x, F_z), _, _) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; avg_x, avg_z, dif_x, dif_z) = operators(model.spatial_scheme)

    ρε      = alloc!(ρε_, F, domain, :xz)
    σ_cons  = fill!(alloc!(σ_cons_, F, domain, :xz), 0)

    @with mgr, let (irange, jrange) = (xrange_interior(domain), zrange_interior(domain))
        @inbounds @vec for i in irange, j in jrange
            ρε[i, j] = inv_dx * dif_x(uτ_x, i, j) + inv_dz * dif_z(uτ_z, i, j) - avg_x(u, F_x, i, j) - avg_z(w, F_z, i, j)
        end
    end

    fill_halo!(ρε, domain, ZeroGradient())

    return ρε, σ_cons
end

## MOMENTUM

# Velocity tendencies on faces before the pressure gradient: forcing - ∇K + ω×u + F/ρ_R. Thermodynamic forcing
# -consvar ∇conjvar - q ∇chempot - ∇B - g ẑ, with consvar, q = avg(m)/ρ_R on the face. Periodized for `pressure!`.
function velocity_tendencies!((du_, dw_), model::AN2D{F}, ::BuoyThermo, (; m), (; conjvar, chempot, B), (K, ωw, ωu, F_x, F_z)) where {F}
    (; mgr, domain, space, inv_dx, inv_dz) = model
    (; ρ_R) = model.reference
    (; g) = space
    (; avg_X, avg_Z, dif_X, dif_Z) = operators(model.spatial_scheme)

    du = alloc!(du_, F, domain, :Xz)
    dw = alloc!(dw_, F, domain, :xZ)

    m1, m2 = eachslice(m; dims = 3)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @inbounds @vec for I in irange, j in jrange
            v_R      = inv(ρ_R[j])
            du[I, j] = - inv_dx * ( v_R * ( avg_X(m1, I, j) * dif_X(conjvar, I, j) + avg_X(m2, I, j) * dif_X(chempot, I, j) )
                                   + dif_X(B, I, j) + dif_X(K, I, j) ) - ωw[I, j] + v_R * F_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @inbounds @vec for i in irange, J in jrange
            v_R      = inv(avg_Z(ρ_R, J))
            dw[i, J] = - inv_dz * ( v_R * ( avg_Z(m1, i, J) * dif_Z(conjvar, i, J) + avg_Z(m2, i, J) * dif_Z(chempot, i, J) )
                                   + dif_Z(B, i, J) + dif_Z(K, i, J) ) - g + ωu[i, J] + v_R * F_z[i, J]
        end
    end

    periodize!(model, (du, dw))

    return du, dw
end
# dynamic forms: buoyancy b = B averaged to the w-faces, no horizontal forcing
function velocity_tendencies!((du_, dw_), model::AN2D{F}, ::BuoyDyn, _, (; B), (K, ωw, ωu, F_x, F_z)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; ρ_R) = model.reference
    (; avg_Z, dif_X, dif_Z) = operators(model.spatial_scheme)

    du = alloc!(du_, F, domain, :Xz)
    dw = alloc!(dw_, F, domain, :xZ)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @inbounds @vec for I in irange, j in jrange
            du[I, j] = - inv_dx * dif_X(K, I, j) - ωw[I, j] + inv(ρ_R[j]) * F_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @inbounds @vec for i in irange, J in jrange
            dw[i, J] = avg_Z(B, i, J) - inv_dz * dif_Z(K, i, J) + ωu[i, J] + inv(avg_Z(ρ_R, J)) * F_z[i, J]
        end
    end

    periodize!(model, (du, dw))

    return du, dw
end

# (du, dw) -= ∇φ, in place, with φ from `pressure!`
function subtract_pressure_gradient!((du, dw), model::AN2D, φ)
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dif_X, dif_Z) = operators(model.spatial_scheme)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @inbounds @vec for I in irange, j in jrange
            du[I, j] = du[I, j] - inv_dx * dif_X(φ, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @inbounds @vec for i in irange, J in jrange
            dw[i, J] = dw[i, J] - inv_dz * dif_Z(φ, i, J)
        end
    end
    return du, dw
end

## TRACERS

# at interior cells: dm = -∇·(m u) - ∇·(consflux, qflux) + (σ_cons, 0)
function dm!(dm_, model::AN2D{F}, (; m), (mu, mw, consflux_x, consflux_z, qflux_x, qflux_z, σ_cons)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dif_x, dif_z) = operators(model.spatial_scheme)

    dm = alloc!(dm_, F, domain, :xz, size(m, 3))

    dm1, dm2    = eachslice(dm; dims = 3)
    mu1, mu2    = eachslice(mu; dims = 3)
    mw1, mw2    = eachslice(mw; dims = 3)

    @with mgr, let (irange, jrange) = (xrange_interior(domain), zrange_interior(domain))
        @inbounds @vec for i in irange, j in jrange
            dm1[i, j] = - inv_dx * dif_x(mu1, i, j) - inv_dz * dif_z(mw1, i, j) - inv_dx * dif_x(consflux_x, i, j) - inv_dz * dif_z(consflux_z, i, j) + σ_cons[i, j]
            dm2[i, j] = - inv_dx * dif_x(mu2, i, j) - inv_dz * dif_z(mw2, i, j) - inv_dx * dif_x(qflux_x, i, j) - inv_dz * dif_z(qflux_z, i, j)
        end
    end
    return dm
end
