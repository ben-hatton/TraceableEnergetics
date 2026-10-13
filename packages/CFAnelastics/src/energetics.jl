## TRADITIONAL ENERGETICS


# energetics tendencies
function energetics_tendencies!(energetics_scratch, model::AN2D, state, dynamics_scratch)
    # energies
    (; ke, pe, te) = energetics_scratch.energies
    ke, pe, te = energies!((ke, pe, te), state, model)

    # Boussinesq GPE, IE
    (; gpe, ie, gpe_bz) = energetics_scratch.bq_energies
    gpe, ie, gpe_bz = bq_energies!((gpe, ie, gpe_bz), state, model, pe, dynamics_scratch)

    # viscous dissipation: ρε
    (; viscous_diss) = energetics_scratch.irreversible
    viscous_diss = @. viscous_diss = dynamics_scratch.irreversible.ρε

    # irreversible entropy production: πσ = ρε - Js.∇π - Jq.∇μ
    (; entropy_prod) = energetics_scratch.irreversible
    entropy_prod = @. entropy_prod = dynamics_scratch.thermo.conjvar * dynamics_scratch.irreversible.σ_cons
    
    # consvar fluxes averaged onto cell centres
    (; Jcons_x, Jcons_z) = energetics_scratch.irreversible
    Jcons_x, Jcons_z = Jcons!((Jcons_x, Jcons_z), state, model, dynamics_scratch)

    # terms relating to consvar/composition fluxes: π∇.Js, μ∇.Jq, Js.∇π, Jq.∇μ, [∇.(πJs), ∇.(μJq) not included]
    (; conjdivJcons, chemdivJq, Jconsgradconj, Jqgradchem) = energetics_scratch.irreversible
    conjdivJcons, chemdivJq, Jconsgradconj, Jqgradchem = irr_flux!((conjdivJcons, chemdivJq, Jconsgradconj, Jqgradchem), state, model, dynamics_scratch)

    # pressure work: p∇.u
    (; pdivu) = energetics_scratch.dynamic
    pdivu = pdivu!(pdivu, state, model, dynamics_scratch)

    # buoyancy flux : ρᵣgw[1 - B + δp.v'(ϕ)]
    (; buoy_flux_Σ_ϕ, buoy_flux_B) = energetics_scratch.dynamic
    buoy_flux_Σ_ϕ = buoy_flux_Σ_ϕ!(buoy_flux_Σ_ϕ, state, model, dynamics_scratch)
    buoy_flux_B   = buoy_flux_B!(buoy_flux_B, state, model, dynamics_scratch)

    # thermodynamic budget terms: ∇.(ρu), π∇.(ρsu), μ∇.(ρqu)
    (; divru, conjdivrsu, chemdivrqu) = energetics_scratch.thermodynamic
    divru, conjdivrsu, chemdivrqu = thermodynamic_budget_terms!((divru, conjdivrsu, chemdivrqu), state, model, dynamics_scratch)

    energetics_scratch = (;
        energies = (; ke, pe, te),
        bq_energies = (; gpe, ie, gpe_bz),
        irreversible = (; Jcons_x, Jcons_z, viscous_diss, entropy_prod, conjdivJcons, chemdivJq, Jconsgradconj, Jqgradchem),
        dynamic = (; pdivu, buoy_flux_Σ_ϕ, buoy_flux_B),
        thermodynamic = (; divru, conjdivrsu, chemdivrqu)
    )
    return energetics_scratch
end

function energies!((ke, pe, te_), (; m, u, w), model::AN2D{F}) where {F}
    ke = kinetic_energy!(ke, (; m, u, w), model)
    pe = potential_energy!(pe, (; m), model)
    te = alloc!(te_, F, model.domain, :xz)
    te = @. te = ke + pe
    periodize!(model, te)
    return ke, pe, te
end

function Jcons!((Jcons_x_, Jcons_z_), (; m), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, domain) = model
    Jcons_x = alloc!(Jcons_x_, F, domain, :xz)
    Jcons_z = alloc!(Jcons_z_, F, domain, :xz)
    @with mgr, let (irange, jrange) = (xrange(domain), axes(Jcons_x, 2))
        @vec for i in irange, j in jrange
            Jcons_x[i, j] = avg_x(dynamics_scratch.irreversible.Jcons_x, i, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons_z, 1), zrange(domain))
        @vec for i in irange, j in jrange
            Jcons_z[i, j] = avg_z(dynamics_scratch.irreversible.Jcons_z, i, j)
        end
    end
    return Jcons_z, Jcons_x
end

function bq_energies!((gpe_, ie_, gpe_bz_), (; m, u, w), model::AN2D{F}, pe, dynamics_scratch) where {F}
    (; mgr, reference, fluid, space) = model
    (; p_R, ρ_R) = reference
    (; g) = space

    gpe = alloc!(gpe_, F, model.domain, :xz)
    ie = alloc!(ie_, F, model.domain, :xz)
    gpe_bz = alloc!(gpe_bz_, F, model.domain, :xz)
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @vec for i in irange, j in jrange
            θ = fluid(:p, :consvar, :q).potential_temperature(p_R[j], dynamics_scratch.thermo.consvar[i, j], dynamics_scratch.thermo.q[i, j])
            cp = fluid(:p, :consvar, :q).heat_capacity(p_R[j], dynamics_scratch.thermo.consvar[i, j], dynamics_scratch.thermo.q[i, j])

            # ie = ρ₀(cₚθ - p₀v₀)
            ie[i, j] = ρ_R[j] * cp * (θ - fluid.T0) - fluid.p0 

            # gpe = pe - ie
            gpe[i, j] = pe[i, j] - ie[i, j]

            # ρ_R[j] * ( v(θ, q) * (p_R - p0) + Φij(model, i, j) ) + g * ρ0 * z 

            # gpe_bz = gz - bz
            b = buoyancy(model, model.buoyancy, dynamics_scratch.thermo.consvar, dynamics_scratch.thermo.q, i, j)
            gpe_bz[i, j] = ρ_R[j] * (g - b) * Φij(model, i, j) / g
        end
    end
    periodize!(model, (gpe, ie, gpe_bz))
    return gpe, ie, gpe_bz
end

# kinetic energy tendency: KE =  ρ_R .(⟨u²⟩ᵢ + ⟨w²⟩ⱼ)/2
function kinetic_energy!(ke_, (; m, u, w), model::AN2D{F}) where {F}
    (; mgr, domain, reference) = model
    (; ρ_R) = reference
    ke = alloc!(ke_, F, domain, :xz)
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            ke[i, j] = 0.5 * ρ_R[j] * (avg_x(abs2, u, i, j) + avg_z(abs2, w, i, j))
        end
    end
    periodize!(model, ke)
    return ke
end

# potential energy tendency: PE = ρ_R .[ h(p_R, s, q) + Φ - p_R.v_R ]
function potential_energy!(pe_, (; m), model::AN2D{F}) where {F}
    (; mgr, reference, fluid) = model
    (; p_R, ρ_R) = reference
    pe = alloc!(pe_, F, model.domain, :xz)
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @vec for i in irange, j in jrange
            s = m[i, j , 1] / ρ_R[j]
            q = m[i, j , 2] / ρ_R[j]
            pe[i, j] = ρ_R[j] * (fluid(:p, :consvar, :q).specific_enthalpy(p_R[j], s, q) + Φij(model, i, j) ) - p_R[j] 
        end
    end
    periodize!(model, pe)
    return pe
end

# terms relating to consvar/composition fluxes: π∇.Js, μ∇.Jq, Js.∇π, Jq.∇μ, ∇.(πJs), ∇.(μJq)
function irr_flux!((conjdivJcons_, chemdivJq_, Jconsgradconj_, Jqgradchem_), (; m), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    conjdivJcons    = alloc!(conjdivJcons_, F, domain, :xz)
    chemdivJq       = alloc!(chemdivJq_, F, domain, :xz)
    Jconsgradconj   = alloc!(Jconsgradconj_, F, domain, :xz)
    Jqgradchem      = alloc!(Jqgradchem_, F, domain, :xz)

    (; Jcons_x, Jcons_z, Jq_x, Jq_z) = dynamics_scratch.irreversible
    (; Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z) = dynamics_scratch.irreversible
    (; conjvar, chempot) = dynamics_scratch.thermo

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            # π∇.Js
            conjdivJcons[i, j]   = conjvar[i, j] * ( dif_x(Jcons_x, i, j) * inv_dx + dif_z(Jcons_z, i, j) * inv_dz )
            # μ∇.Jq
            chemdivJq[i, j]      = chempot[i, j] * ( dif_x(Jq_x, i, j) * inv_dx + dif_z(Jq_z, i, j) * inv_dz )
            # Js.∇π = Js_x*∂π/∂x + Js_z*∂π/∂z
            Jconsgradconj[i, j]  = avg_x(Jcons∂conj_∂x, i, j) + avg_z(Jcons∂conj_∂z, i, j)
            # Jq.∇μ
            Jqgradchem[i, j]     = avg_x(Jq∂chem_∂x, i, j) + avg_z(Jq∂chem_∂z, i, j)
        end
    end
    periodize!(model, (conjdivJcons, chemdivJq, Jconsgradconj, Jqgradchem))
    return conjdivJcons, chemdivJq, Jconsgradconj, Jqgradchem
end

# pressure work tendency: p∇.u
function pdivu!(pdivu_, (; m), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, reference, domain) = model
    (; ρ_R, p_R) = reference
    pdivu = alloc!(pdivu_, F, domain, :xz)
    @with mgr, let (irange, jrange) = (axes(pdivu, 1), axes(pdivu, 2))
        @vec for i in irange, j in jrange
            p = p_R[j] + ρ_R[j] * dynamics_scratch.momentum.φ[i, j]
            pdivu[i, j] = p * (dynamics_scratch.derivatives.∂u_x[i, j] + dynamics_scratch.derivatives.∂w_z[i, j])
        end
    end
    periodize!(model, pdivu)
    return pdivu
end

# buoyancy flux tendency: ρ_R . Σ_ϕ . u.∇Φ
function buoy_flux_Σ_ϕ!(buoy_flux_Σ_ϕ_, (; m, w), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, reference, domain, fluid, space, inv_dz) = model
    (; ρ_R, p_R) = reference
    (; g ) = space
    buoy_flux_Σ_ϕ = fill!(alloc!(buoy_flux_Σ_ϕ_, F, domain, :xz), 0)    # interior cells only (∂ρ_R/∂z)
    @with mgr, let (irange, jrange) = (axes(buoy_flux_Σ_ϕ, 1), zrange_interior(domain))
        @vec for i in irange, j in jrange
            s = m[i, j , 1] / ρ_R[j]
            q = m[i, j , 2] / ρ_R[j]
            δp = dynamics_scratch.momentum.φ[i, j] * ρ_R[j]
            v = fluid(:p, :consvar, :q).specific_volume(p_R[j], s, q)
            ∂v_R∂Φ = - inv(ρ_R[j])^2 * inv_dz * (avg_Z(ρ_R, j+1) - avg_Z(ρ_R, j)) / g
            buoy_flux_Σ_ϕ[i, j] = ρ_R[j] * g * avg_z(w, i, j) * (2 - ρ_R[j] * v + δp * ∂v_R∂Φ)
        end
    end
    periodize!(model, buoy_flux_Σ_ϕ)
    return buoy_flux_Σ_ϕ
end

# buoyancy flux tendency: ρ_R . B . u.∇Φ
function buoy_flux_B!(buoy_flux_B_, (; m, w), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, reference, domain, fluid, space) = model
    (; ρ_R, p_R) = reference
    (; g ) = space
    buoy_flux_B = alloc!(buoy_flux_B_, F, domain, :xz)
    @with mgr, let (irange, jrange) = (axes(buoy_flux_B, 1), zrange(domain))
        @vec for i in irange, j in jrange
            s = m[i, j , 1] / ρ_R[j]
            q = m[i, j , 2] / ρ_R[j]
            v = fluid(:p, :consvar, :q).specific_volume(p_R[j], s, q)
            buoy_flux_B[i, j] = ρ_R[j] * g * avg_z(w, i, j) * (ρ_R[j] * v - 1)
        end
    end
    periodize!(model, buoy_flux_B)
    return buoy_flux_B
end

# thermodynamic budget terms: ∇.(ρu), π∇.(ρsu), μ∇.(ρqu)
function thermodynamic_budget_terms!((divru_, conjdivrsu_, chemdivrqu_), (; m, u, w), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, domain, reference, inv_dx, inv_dz) = model
    (; ρ_R) = reference
    divru    = fill!(alloc!(divru_, F, domain, :xz), 0)        # interior cells only (ρ_R on w-faces)
    conjdivrsu  = alloc!(conjdivrsu_, F, domain, :xz)
    chemdivrqu  = alloc!(chemdivrqu_, F, domain, :xz)

    (; mu, mw) = dynamics_scratch.advection

    rsu = @view mu[:, :, 1]
    rqu = @view mu[:, :, 2]
    rsw = @view mw[:, :, 1]
    rqw = @view mw[:, :, 2]

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            conj = dynamics_scratch.thermo.conjvar[i, j]
            chem = dynamics_scratch.thermo.chempot[i, j]
            conjdivrsu[i, j] = conj * ( dif_x(rsu, i, j) * inv_dx + dif_z(rsw, i, j) * inv_dz )
            chemdivrqu[i, j] = chem * ( dif_x(rqu, i, j) * inv_dx + dif_z(rqw, i, j) * inv_dz )
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), zrange_interior(domain))
        @vec for i in irange, j in jrange
            divru[i, j] = ρ_R[j] * dif_x(u, i, j) * inv_dx + (avg_Z(ρ_R, j+1) * w[i, j+1] - avg_Z(ρ_R, j) * w[i, j]) * inv_dz
        end
    end
    periodize!(model, (divru, conjdivrsu, chemdivrqu))
    return divru, conjdivrsu, chemdivrqu
end

## LOOP

# energetics loop
function Φij(model::AN2D, i, j)
    return model.space.g * (CFBoxes.zpoint(model.domain, model.dz, j) - model.space.Lz)
end

# initialisation
