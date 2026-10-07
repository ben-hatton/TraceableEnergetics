## TENDENCIES

# assign CFTimeschemes.tendencies! for the AN2D type
CFTimeSchemes.tendencies!(dstate, scratch, model::AN2D, state, t) =
    tendencies_AN!(dstate, scratch, model, state, t)

function tendencies_AN!(dstate, scratch, model::AN2D, state, t)
    # compute conservative variable, composition, temperature
    (; consvar, comp, T) = scratch.thermo
    consvar, comp, T = cons_comp_T!((consvar, comp, T), model, state)

    # apply boundary conditions to state (and consvar, comp, T)
    state = state_bc!(model, state, (consvar, comp, T))

    ## VELOCITY DERIVATIVES; ∂u_x, ∂w_z, ∂u/∂z, ∂w/∂x
    (; ∂u_x, ∂w_z, ∂u_z, ∂w_x) = scratch.derivatives

    ∂u_x, ∂w_z  = grad_uw!((∂u_x, ∂w_z), model, state)
    ∂u_z, ∂w_x  = skew_uw!((∂u_z, ∂w_x), model, state)

    ## ADVECTION; ω×u = (ωw, -ωu), K = (u² + w²) /2
    (; K, ∂K_x, ∂K_z, ωw, ωu, adv_scratch, mu, mw) = scratch.advection

    K, ∂K_x, ∂K_z       = ke_adv!((K, ∂K_x, ∂K_z), model, state)
    mu, mw              = advective_flux!((mu, mw), model, state)
    ωw, ωu, adv_scratch = vort_adv!((ωw, ωu, adv_scratch), model, model.advection, state, (∂u_z, ∂w_x))

    ## VISCOUS DISSIPATION
    (; ρε, visc_scratch) = scratch.irreversible

    ρε, visc_scratch = viscous_dissipation!((ρε, visc_scratch), model, state, (∂u_x, ∂w_z, ∂u_z, ∂w_x))

    ## THERMODYNAMICS
    (; conjvar, chempot, Cp, ∂cons_∂q, Γ_qΦ) = scratch.thermo

    conjvar, chempot, Cp, ∂cons_∂q, Γ_qΦ = thermodynamics!((conjvar, chempot, Cp, ∂cons_∂q, Γ_qΦ), model, (consvar, comp, T))
    
    ## BUOYANCY
    (; b) = scratch.thermo
    (; B_x, B_z, B_scratch) = scratch.momentum

    B_x, B_z, B_scratch, b = buoyancy!((B_x, B_z, B_scratch, b), model, model.buoyancy, state, (consvar, comp, conjvar, chempot))

    ## HEAT & COMPOSITION FLUX
    (; Jcons_x, Jcons_z, Jq_x, Jq_z)  = scratch.irreversible

    Jcons_x, Jcons_z, Jq_x, Jq_z = fluxes!((Jcons_x, Jcons_z, Jq_x, Jq_z), model, model.heatflux, state, (consvar, comp, conjvar, T, Cp, ∂cons_∂q, chempot, Γ_qΦ))
    
    ## ENTROPY PRODUCTION
    (; σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z) = scratch.irreversible

    σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z = entropy_production!((σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z), model, model.heatflux, state, (conjvar, chempot, Jcons_x, Jcons_z, Jq_x, Jq_z, ρε))
    
    # MOMENTUM EQUATION
    (; dU, dW) = scratch.momentum

    dU, dW = dUW!((dU, dW), model, state, (B_x, B_z, ∂K_x, ∂K_z, ωw, ωu, ∂u_x, ∂w_z, ∂u_z, ∂w_x))

    ## PRESSURE SOLVER; φ = (p - p_R) / ρ_R
    (; φ) = scratch.momentum
    poisson_scratch = scratch.poisson_scratch 

    φ, poisson_scratch = pressure!((φ, poisson_scratch), model, state, (dU, dW))
    
    # UPDATE DSTATE
    dm, du, dw  = (dstate.m, dstate.u, dstate.w)

    dm      = dm!(dm, model, state, (mu, mw, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons))
    du, dw  = duw!((du, dw), model, state, (dU, dW, φ))

    dstate = (m = dm, u = du, w = dw)    

    scratch = (
        momentum        = (; dU, dW, φ, B_x, B_z, B_scratch),
        thermo          = (; consvar, comp, conjvar, chempot, T, Cp, ∂cons_∂q, Γ_qΦ, b),
        derivatives     = (; ∂u_x, ∂w_z, ∂u_z, ∂w_x),
        irreversible    = (; ρε, visc_scratch, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z),
        advection       = (; mu, mw, K, ∂K_x, ∂K_z, ωw, ωu, adv_scratch),
        poisson_scratch = poisson_scratch
    )
    return dstate, scratch
end

## CONSERVATIVE VARIABLE, COMPOSITION, TEMPERATURE
function cons_comp_T!((consvar_, comp_, T_), model::AN2D, (; m))
    (; mgr, fluid, reference) = model
    (; p_R, v_R) = reference

    consvar = similar!(consvar_, m, size(m)[1:2]...)
    comp    = similar!(comp_,    m, size(m)[1:2]...)
    T       = similar!(T_,       m, size(m)[1:2]...)

    @with mgr, let (irange, jrange) = (axes(consvar, 1), axes(consvar, 2))
        @vec for i in irange, j in jrange
            consvar[i, j] = v_R[j] * m[i, j, 1]
            comp[i, j]    = v_R[j] * m[i, j, 2]
            T[i, j]       = fluid(:p, :consvar, :q).temperature(p_R[j], consvar[i, j], comp[i, j])  
        end
    end
    return consvar, comp, T
end

## VELOCITY DERIVATIVES

# skew derivatives : (u, w) at edges ↦ (∂u/∂z, ∂w/∂x) at vertices
function skew_uw!((∂u_z_, ∂w_x_), model::AN2D{F}, (; u, w)) where F
    (; mgr, domain, inv_dx, inv_dz) = model

    ∂u_z = similar!(∂u_z_, Array{F}, Xdim(domain), Zdim(domain))
    ∂w_x = similar!(∂w_x_, Array{F}, Xdim(domain), Zdim(domain))
    
    @with mgr, let (irange, jrange) = (axes(∂u_z, 1), Zrange(domain))
        @vec for I in irange, J in jrange
            ∂u_z[I, J] = inv_dz * dif_Z(u, I, J)
        end
    end
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(∂w_x, 2))
        @vec for I in irange, J in jrange
            ∂w_x[I, J] = inv_dx * dif_X(w, I, J)
        end
    end

    periodize!(model, (∂u_z, ∂w_x))

    return ∂u_z, ∂w_x
end
# gradient of edge quantities : (u, w) at edges ↦ (∂u/∂x, ∂w/∂z) at centres

function grad_uw!((∂u_x_, ∂w_z_), model::AN2D, (; m, u, w))
    (; mgr, domain, inv_dx, inv_dz) = model

    ∂u_x = similar!(∂u_x_, m, size(m)[1:2]...)
    ∂w_z = similar!(∂w_z_, m, size(m)[1:2]...)

    @with mgr, let (irange, jrange) = (xrange(domain), axes(∂u_x, 2))
        @vec for i in irange, j in jrange
            ∂u_x[i, j] = inv_dx * dif_x(u, i, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(∂w_z, 1), zrange(domain))
        @vec for i in irange, j in jrange
            ∂w_z[i, j] = inv_dz * dif_z(w, i, j)
        end
    end

    periodize!(model, (∂u_x, ∂w_z))

    return ∂u_x, ∂w_z
end

# ADVECTION

# vorticity term for advection
function vort_adv!((ωw_, ωu_, (ωwᴵᴶ_, ωuᴵᴶ_)), model::AN2D, advection::AdvEnergyCons, (; u, w), (∂u_z, ∂w_x))
    (; mgr, domain) = model
    (; ρ_R, ρ_R_J) = model.reference

    ωw      = similar!(ωw_, u)
    ωu      = similar!(ωu_, w)
    ωwᴵᴶ    = similar!(ωwᴵᴶ_, ∂u_z)
    ωuᴵᴶ    = similar!(ωuᴵᴶ_, ∂u_z)

    @with mgr, let (irange, jrange) = (Xrange(domain), Zrange(domain))
        @vec for I in irange, J in jrange
            ω = ∂u_z[I, J] - ∂w_x[I, J]
            ωwᴵᴶ[I, J] = ω * avg_X(w, I, J)
            ωuᴵᴶ[I, J] = ω * inv(ρ_R_J[J]) * avg_Z(ρ_R, u, I, J)
        end
    end
    periodize!(model, (ωwᴵᴶ, ωuᴵᴶ))

    # ωwⁱᴶ
    @with mgr, let (irange, jrange) = (axes(ωw, 1), zrange(domain))
        @vec for I in irange, j in jrange
            ωw[I, j] = avg_z(ωwᴵᴶ, I, j)
        end
    end

    # ωuᴵʲ
    @with mgr, let (irange, jrange) = (xrange(domain), axes(ωu, 2))
        @vec for i in irange, J in jrange
            ωu[i, J] = avg_x(ωuᴵᴶ, i, J)
        end
    end

    periodize!(model, (ωw, ωu))

    return ωw, ωu, (ωwᴵᴶ, ωuᴵᴶ)
end
function vort_adv!((ωw_, ωu_, (ωvᴵᴶ_, ρwᴵ_, ρuᴶ_)), model::AN2D, advection::AdvEnstrophyCons, (; u, w), (∂u_z, ∂w_x))
    (; mgr, domain) = model
    (; ρ_R, ρ_R_J) = model.reference

    ωw      = similar!(ωw_, u)
    ωu      = similar!(ωu_, w)

    ωvᴵᴶ    = similar!(ωvᴵᴶ_, ∂u_z)
    ρwᴵ     = similar!(ρwᴵ_, ∂u_z)
    ρuᴶ     = similar!(ρuᴶ_, ∂u_z)

    @with mgr, let (irange, jrange) = (Xrange(domain), Zrange(domain))
        @vec for I in irange, J in jrange
            ω = ∂u_z[I, J] - ∂w_x[I, J]
            ωvᴵᴶ[I, J] = ω * inv(ρ_R_J[J])
            ρwᴵ[I, J] = ρ_R_J[J] * avg_X(w, I, J)
            ρuᴶ[I, J] = avg_Z(ρ_R, u, I, J)
        end
    end
    periodize!(model, (ωvᴵᴶ, ρuᴶ, ρwᴵ))

    # ωwⁱᴶ
    @with model.mgr, let (irange, jrange) = (axes(ωw, 1), zrange(domain))
        @vec for I in irange, j in jrange
            ωw[I, j] = avg_z(ωvᴵᴶ, I, j) * avg_z(ρwᴵ, I, j)
        end
    end

    # ωuᴵʲ
    @with mgr, let (irange, jrange) = (xrange(domain), axes(ωu, 2))
        @vec for i in irange, J in jrange
            ωu[i, J] = avg_x(ωvᴵᴶ, i, J) * avg_x(ρuᴶ, i, J)
        end
    end    
    periodize!(model, (ωw, ωu))
    return ωw, ωu, (ωvᴵᴶ, ρwᴵ, ρuᴶ)
end

# kinetic energy term for advection
function ke_adv!((K_, ∂K_x_, ∂K_z_), model::AN2D, (; m, u, w))
    (; mgr, domain, inv_dx, inv_dz) = model

    K       = similar!(K_, m, size(m)[1:2]...)
    ∂K_x    = similar!(∂K_x_, u)
    ∂K_z    = similar!(∂K_z_, w)

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            K[i, j] = 0.5 * ( avg_x(abs2, u, i, j) + avg_z(abs2, w, i, j) )
        end
    end
    periodize!(model, K)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(∂K_x, 2))
        @vec for I in irange, j in jrange
            ∂K_x[I, j] = inv_dx * dif_X(K, I, j)
        end
    end    
    @with mgr, let (irange, jrange) = (axes(∂K_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            ∂K_z[i, J] = inv_dz * dif_Z(K, i, J)
        end
    end
    periodize!(model, (∂K_x, ∂K_z))
    return K, ∂K_x, ∂K_z
end

# flux for tracer advection
function advective_flux!((mu_, mw_), model::AN2D, (; m, u, w))
    (; mgr, domain) = model

    mu = similar!(mu_, u, size(u)..., 2)
    mw = similar!(mw_, w, size(w)..., 2)
    
    m1 = @view m[:, :, 1]
    m2 = @view m[:, :, 2]
    mu1 = @view mu[:, :, 1]
    mu2 = @view mu[:, :, 2]
    mw1 = @view mw[:, :, 1]
    mw2 = @view mw[:, :, 2]

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(mu, 2))
        @vec for I in irange, j in jrange
            mu1[I, j] = u[I, j] * avg_X(m1, I, j)
            mu2[I, j] = u[I, j] * avg_X(m2, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(mw, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            mw1[i, J] = w[i, J] * avg_Z(m1, i, J)
            mw2[i, J] = w[i, J] * avg_Z(m2, i, J)
        end
    end
    periodize!(model, (mu1, mu2, mw1, mw2))
    return mu, mw
end

## VISCOUS DISSIPATION

function viscous_dissipation!((ρε_, (uτ_x_, uτ_z_, u∂τ_x_, u∂τ_z_)), model, (; m, u, w), (∂u_x, ∂w_z, ∂u_z, ∂w_x))
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dyn_visc, bulk_visc) = model.viscosity

    ρε      = similar!(ρε_, m, size(m)[1:2]...)
    uτ_x    = similar!(uτ_x_, u)
    uτ_z    = similar!(uτ_z_, w)
    u∂τ_x   = similar!(u∂τ_x_, u)
    u∂τ_z   = similar!(u∂τ_z_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange
            divu_x = avg_X(∂u_x, I, j) + avg_X(∂w_z, I, j)
            ∂divu_x = inv_dx * (dif_X(∂u_x, I, j) + dif_X(∂w_z, I, j))
            ∂ω_z  = inv_dz * (dif_z(∂u_z, I, j) - dif_z(∂w_x, I, j))
            # term inside gradient
            uτ_x[I, j] = ( u[I, j] * ( 2 * dyn_visc * avg_X(∂u_x, I, j) + ( bulk_visc - (2 / 3) * dyn_visc ) * divu_x )
                    + dyn_visc * avg_Xz(w, I, j) * ( avg_z(∂u_z, I, j) + avg_z(∂w_x, I, j) ) )
            # term inside average
            u∂τ_x[I, j] = ( u[I, j] * ( - dyn_visc * ∂ω_z - ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_x ) )
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @vec for i in irange, J in jrange
            divu_z = avg_Z(∂u_x, i, J) + avg_Z(∂w_z, i, J)
            ∂divu_z = inv_dz * (dif_Z(∂u_x, i, J) + dif_Z(∂w_z, i, J))
            ∂ω_x  = inv_dx * (dif_x(∂u_z, i, J) - dif_x(∂w_x, i, J))
            # term inside gradient
            uτ_z[i, J] = ( w[i, J] * ( 2 * dyn_visc * avg_Z(∂w_z, i, J) + ( bulk_visc - (2 / 3) * dyn_visc ) * divu_z )
                    + dyn_visc * avg_xZ(u, i, J) * ( avg_x(∂u_z, i, J) + avg_x(∂w_x, i, J) ) )
            # term inside average
            u∂τ_z[i, J] = ( w[i, J] * ( dyn_visc * ∂ω_x - ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_z ) )
        end
    end

    periodize!(model, (uτ_x, u∂τ_x, uτ_z, u∂τ_z))

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            ρε[i, j] = inv_dx * dif_x(uτ_x, i, j) + inv_dz * dif_z(uτ_z, i, j) + avg_x(u∂τ_x, i, j) + avg_z(u∂τ_z, i, j)
        end
    end

    periodize!(model, ρε)

    return ρε, (uτ_x, uτ_z, u∂τ_x, u∂τ_z)
end

## THERMODYNAMICS

function thermodynamics!((conjvar_, chempot_, Cp_, ∂cons_∂q_, Γ_qΦ_), model::AN2D, (consvar, comp, T))
    (; mgr, fluid, reference) = model
    (; ρ_R, p_R) = reference

    conjvar = similar!(conjvar_, consvar)
    chempot = similar!(chempot_, consvar)
    Cp      = similar!(Cp_, consvar)
    ∂cons_∂q = similar!(∂cons_∂q_, consvar)
    Γ_qΦ   = similar!(Γ_qΦ_, consvar)
    
    @with mgr, let (irange, jrange) = (axes(consvar, 1), axes(consvar, 2))
        @vec for i in irange, j in jrange
            # variable conjugate to conservative variable
            conjvar[i, j]   = fluid(:p, :consvar, :q).conjugate_variable(p_R[j], consvar[i, j], comp[i, j])

            # modified chemical pontential (relative to choice of consvar)
            chempot[i, j]   = fluid(:p, :consvar, :q).modified_chemical_potential(p_R[j], consvar[i, j], comp[i, j])

            # specific heat capacity at reference pressure
            Cp[i, j]        = fluid(:p, :consvar, :q).heat_capacity(p_R[j], consvar[i, j], comp[i, j])

            # partial derivative ∂cons/∂q(p, T, q)
            ∂cons_∂q[i, j]   = fluid(:p, :consvar, :q).dcons_dq(p_R[j], consvar[i, j], comp[i, j])

            # derivatives of μ(p, T, q)
            _, ∂μ_∂p, _, ∂μ_∂q  = fluid(:p, :T, :q).chemical_potential_derivatives(p_R[j], T[i, j], comp[i, j])
            Γ_qΦ[i, j]          = ρ_R[j] * ∂μ_∂p / ∂μ_∂q
        end
    end
    return conjvar, chempot, Cp, ∂cons_∂q, Γ_qΦ
end

## BOUYANCY

function buoyancy(model::AN2D, ::BuoyDynOld, s, q, i, j)
    (; fluid, reference, space) = model
    (; v_R, p_R) = reference
    (; g) = space
    return g * ( 1 - v_R[j] * inv(fluid(:p, :consvar, :q).specific_volume(p_R[j], s[i, j], q[i, j])))
end

function buoyancy(model::AN2D, ::Union{BuoyDynNew, BuoyThermo}, s, q, i, j)
    (; fluid, reference, space) = model
    (; ρ_R, p_R) = reference
    (; g) = space
    return g * ( ρ_R[j] * fluid(:p, :consvar, :q).specific_volume(p_R[j], s[i, j], q[i, j]) - 1)
end

function buoyancy!((B_x_, B_z_, B_, b_), model::AN2D, buoyancy_scheme::BuoyThermo, (; m, u, w), (consvar, comp, conjvar, chempot))
    (; mgr, domain, space, reference, fluid, inv_dx, inv_dz) = model
    (; g) = space
    (; p_R, ρ_R, ρ_R_J) = reference

    B_x = similar!(B_x_, u)
    B_z = similar!(B_z_, w)
    B   = similar!(B_, consvar)
    b   = similar!(b_, consvar)

    @with mgr, let (irange, jrange) = (axes(B, 1), axes(B, 2))
        @vec for i in irange, j in jrange
            h = fluid(:p, :consvar, :q).specific_enthalpy(p_R[j], consvar[i, j], comp[i, j])
            B[i, j] = h - conjvar[i, j] * consvar[i, j] - chempot[i, j] * comp[i, j]
            b[i, j] = buoyancy(model, buoyancy_scheme, consvar, comp, i, j)
        end
    end
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(B_x, 2))
        @vec for I in irange, j in jrange
            v_x         = inv(ρ_R[j])
            consvar_x   = @views avg_X(m[:, :, 1], I, j) * v_x
            comp_x      = @views avg_X(m[:, :, 2], I, j) * v_x
            cons∂conj_x = inv_dx * dif_X(conjvar, I, j) * consvar_x
            comp∂chem_x = inv_dx * dif_X(chempot, I, j) * comp_x
            ∂B_x        = inv_dx * dif_X(B, I, j)
            B_x[I, j]   = - cons∂conj_x  - comp∂chem_x - ∂B_x
        end
    end    
    @with mgr, let (irange, jrange) = (axes(B_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            v_z         = inv(ρ_R_J[J])
            consvar_z   = @views avg_Z(m[:, :, 1], i, J) * v_z
            comp_z      = @views avg_Z(m[:, :, 2], i, J) * v_z
            cons∂conj_z = inv_dz * dif_Z(conjvar, i, J) * consvar_z
            comp∂chem_z = inv_dz * dif_Z(chempot, i, J) * comp_z
            ∂B_z        = inv_dz * dif_Z(B, i, J)
            B_z[i, J]   = - cons∂conj_z - comp∂chem_z - ∂B_z - g
        end
    end

    periodize!(model, (B_x, B_z))

    return B_x, B_z, B, b
end
function buoyancy!((B_x_, B_z_, B_, b_), model::AN2D, buoyancy_scheme::BuoyDyn, (; u, w), (consvar, comp, conjvar, chempot))
    (; mgr, domain) = model

    B_x = similar!(B_x_, u)
    B_z = similar!(B_z_, w)
    B   = similar!(B_, consvar)
    b   = similar!(b_, consvar)

    @with mgr, let (irange, jrange) = (axes(b, 1), axes(b, 2))
        @vec for i in irange, j in jrange
            buoy = buoyancy(model, buoyancy_scheme, consvar, comp, i, j)
            B[i, j] = buoy
            b[i, j] = buoy
        end
    end
    @with mgr, let (irange, jrange) = (axes(B_x, 1), axes(B_x, 2))
        @vec for I in irange, j in jrange
            B_x[I, j] = 0. * B_x[I, j]
        end
    end    
    @with mgr, let (irange, jrange) = (axes(B_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            B_z[i, J]  = avg_Z(b, i, J)
        end
    end

    periodize!(model, (B_x, B_z))

    return B_x, B_z, B, b
end

## IRREVERSIBLE TERMS

function fluxes!((Jcons_x_, Jcons_z_, Jq_x_, Jq_z_), model::AN2D, heatflux::HeatFluxConsistent, (; u, w), (consvar, comp, conjvar, T, Cp, ∂cons_∂q, Γ_qΦ))
    (; mgr, domain, space, inv_dx, inv_dz) = model
    (; ρ_R, ρ_R_J) = model.reference
    (; k_T, k_q) = model.heatflux
    (; g) = space

    Jcons_x = similar!(Jcons_x_, u)
    Jcons_z = similar!(Jcons_z_, w)
    Jq_x    = similar!(Jq_x_, u)
    Jq_z    = similar!(Jq_z_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(Jcons_x, 2))
        @vec for I in irange, j in jrange
            # choice of horizontal (reduced) heat flux and salt flux
            JT_x            = - ρ_R[j] * avg_X(Cp, I, j) * k_T * inv_dx * dif_X(T, I, j)
            Jq_x[I, j]      = - ρ_R[j] * k_q * ( inv_dx * dif_X(comp, I, j) )
            
            # horizontal consvar flux
            conjvar_x       = avg_X(conjvar, I, j)
            ∂cons_∂q_x      = avg_X(∂cons_∂q, I, j)
            Jcons_x[I, j]   = JT_x / conjvar_x + ∂cons_∂q_x * Jq_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            # coefficient of -∇Φ in salt flux
            Γ_qΦ_z = avg_Z(Γ_qΦ, i, J)

            # choice of vertical (reduced) heat flux and salt flux
            JT_z            = - ρ_R_J[J] * avg_Z(Cp, i, J) * k_T * inv_dz * dif_Z(T, i, J)
            Jq_z[i, J]      = - ρ_R_J[J] * k_q * ( inv_dz * dif_Z(comp, i, J) - Γ_qΦ_z * g ) 

            # vertical consvar flux
            conjvar_z       = avg_Z(conjvar, i, J)
            ∂cons_∂q_z      = avg_Z(∂cons_∂q, i, J)
            Jcons_z[i, J]   = JT_z / conjvar_z + ∂cons_∂q_z * Jq_z[i, J]
        end
    end

    periodize!(model, (Jcons_x, Jcons_z, Jq_x, Jq_z))

    return Jcons_x, Jcons_z, Jq_x, Jq_z
end

function fluxes!((Jcons_x_, Jcons_z_, Jq_x_, Jq_z_), model::AN2D, heatflux::HeatFluxSimple, (; u, w), (consvar, comp, conjvar, T, Cp, ∂cons_∂q, Γ_qΦ_z))
    (; mgr, domain, inv_dx, inv_dz) = model
    (; ρ_R, ρ_R_J) = model.reference
    (; k_T, k_q) = model.heatflux

    Jcons_x = similar!(Jcons_x_, u)
    Jcons_z = similar!(Jcons_z_, w)
    Jq_x    = similar!(Jq_x_, u)
    Jq_z    = similar!(Jq_z_, w)

    # JH = conjvar * Jcons = J_heat + conjvar * (∂cons / ∂q) * J_q
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(Jcons_x, 2))
        @vec for I in irange, j in jrange
            Jcons_x[I, j]   = - ρ_R[j] * k_T * inv_dx * dif_X(consvar, I, j)
            Jq_x[I, j]      = - ρ_R[j] * k_q * inv_dx * dif_X(comp, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            Jcons_z[i, J]   = - ρ_R_J[J] * k_T * inv_dz * dif_Z(consvar, i, J)
            Jq_z[i, J]      = - ρ_R_J[J] * k_q * inv_dz * dif_Z(comp, i, J)
        end
    end

    periodize!(model, (Jcons_x, Jcons_z, Jq_x, Jq_z))

    return Jcons_x, Jcons_z, Jq_x, Jq_z
end

function entropy_production!((σ_cons_, Jcons∂conj_∂x_, Jcons∂conj_∂z_, Jq∂chem_∂x_, Jq∂chem_∂z_), model, heatflux::HeatFluxConsistent, (; u, w), (conjvar, chempot, Jcons_x, Jcons_z, Jq_x, Jq_z, ρε))
    (; mgr, domain, inv_dx, inv_dz) = model
        
    σ_cons = similar!(σ_cons_, conjvar)
    Jcons∂conj_∂x = similar!(Jcons∂conj_∂x_, u)
    Jcons∂conj_∂z = similar!(Jcons∂conj_∂z_, w)
    Jq∂chem_∂x = similar!(Jq∂chem_∂x_, u)
    Jq∂chem_∂z = similar!(Jq∂chem_∂z_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(Jcons∂conj_∂x, 2))
        @vec for I in irange, j in jrange
            Jcons∂conj_∂x[I, j] = Jcons_x[I, j] * inv_dx * dif_X(conjvar, I, j) 
            Jq∂chem_∂x[I, j]    = Jq_x[I, j] * inv_dx * dif_X(chempot, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons∂conj_∂z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            Jcons∂conj_∂z[i, J] = Jcons_z[i, J] * inv_dz * dif_Z(conjvar, i, J)
            Jq∂chem_∂z[i, J]    = Jq_z[i, J] * inv_dz * dif_Z(chempot, i, J)
        end
    end
    periodize!(model, (Jcons∂conj_∂x, Jq∂chem_∂x, Jcons∂conj_∂z, Jq∂chem_∂z))

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            σ_cons[i, j] = inv(conjvar[i, j]) * ( ρε[i, j] - avg_x(Jcons∂conj_∂x, i, j) - avg_z(Jcons∂conj_∂z, i, j) - avg_x(Jq∂chem_∂x, i, j) - avg_z(Jq∂chem_∂z, i, j) ) 
        end
    end

    periodize!(model, σ_cons)

    return σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z
end

function entropy_production!((σ_cons_, Jcons∂conj_∂x_, Jcons∂conj_∂z_, Jq∂chem_∂x_, Jq∂chem_∂z_), model, heatflux::HeatFluxSimple, (; u, w), (conjvar, chempot))
    (; mgr) = model
        
    σ_cons = similar!(σ_cons_, conjvar)
    Jcons∂conj_∂x = similar!(Jcons∂conj_∂x_, u)
    Jcons∂conj_∂z = similar!(Jcons∂conj_∂z_, w)
    Jq∂chem_∂x = similar!(Jq∂chem_∂x_, u)
    Jq∂chem_∂z = similar!(Jq∂chem_∂z_, w)

    @with mgr, let (irange, jrange) = (axes(σ_cons, 1), axes(σ_cons, 2))
        @vec for i in irange, j in jrange
            σ_cons[i, j] = 0. * σ_cons[i, j]
        end
    end
    return σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z
end

## MOMENTUM

function dUW!((dU_, dW_), model::AN2D, (; u, w), (B_x, B_z, ∂K_x, ∂K_z, ωw, ωu, ∂u_x, ∂w_z, ∂u_z, ∂w_x))
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dyn_visc, bulk_visc) = model.viscosity
    (; v_R, ρ_R_J) = model.reference

    dU = similar!(dU_, u)
    dW = similar!(dW_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange
            ∂ω_z        = inv_dz * (dif_z(∂u_z, I, j) - dif_z(∂w_x, I, j))
            ∂divu_x     = inv_dx * (dif_X(∂u_x, I, j) + dif_X(∂w_z, I, j))
            dU[I, j]    = B_x[I, j] - ∂K_x[I, j] - ωw[I, j] + v_R[j] * (dyn_visc * ∂ω_z + (bulk_visc + (4 / 3) * dyn_visc) * ∂divu_x)
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @vec for i in irange, J in jrange
            ∂ω_x        = inv_dx * (dif_x(∂u_z, i, J) - dif_x(∂w_x, i, J))
            ∂divu_z     = inv_dz * (dif_Z(∂u_x, i, J) + dif_Z(∂w_z, i, J))
            dW[i, J]    = B_z[i, J] - ∂K_z[i, J] + ωu[i, J] + inv(ρ_R_J[J]) * (-dyn_visc * ∂ω_x + (bulk_visc + (4 / 3) * dyn_visc) * ∂divu_z)
        end
    end

    periodize!(model, (dU, dW))

    return dU, dW
end

## UPDATE DSTATE

function dm!(dm_, model::AN2D, (; m), (mu, mw, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons))
    (; mgr, domain, inv_dx, inv_dz) = model
    
    dm = similar!(dm_, m)
    
    # create views of three dimensional arrays
    dm1 = @view dm[:, :, 1]
    dm2 = @view dm[:, :, 2]
    mu1 = @view mu[:, :, 1]
    mu2 = @view mu[:, :, 2]
    mw1 = @view mw[:, :, 1]
    mw2 = @view mw[:, :, 2]

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange                      
            dm1[i, j] = - inv_dx * dif_x(mu1, i, j) - inv_dz * dif_z(mw1, i, j) - inv_dx * dif_x(Jcons_x, i, j) - inv_dz * dif_z(Jcons_z, i, j) + σ_cons[i, j] 
            dm2[i, j] = - inv_dx * dif_x(mu2, i, j) - inv_dz * dif_z(mw2, i, j) - inv_dx * dif_x(Jq_x, i, j) - inv_dz * dif_z(Jq_z, i, j)
        end       
    end
    return dm
end

function duw!((du_, dw_), model::AN2D, (; u, w), (dU, dW, φ))
    (; mgr, domain, inv_dx, inv_dz) = model

    # initialise arrays
    du = similar!(du_, u)
    dw = similar!(dw_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange
            ∂φ_x = dif_X(φ, I, j) * inv_dx
            du[I, j] = dU[I, j] - ∂φ_x
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        for i in irange, J in jrange
            ∂φ_z = dif_Z(φ, i, J) * inv_dz
            dw[i, J] = dW[i, J] - ∂φ_z
        end
    end
    return du, dw
end