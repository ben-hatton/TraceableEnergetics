## TRADITIONAL ENERGETICS

using NCDatasets

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

    # terms relating to heat/salt fluxes: π∇.Js, μ∇.Jq, Js.∇π, Jq.∇μ, [∇.(πJs), ∇.(μJq) not included]
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

function energies!((ke, pe, te_), (; m, u, w), model)
    ke = kinetic_energy!(ke, (; m, u, w), model)
    pe = potential_energy!(pe, (; m), model)
    te = similar!(te_, m, size(m)[1:2]...)
    te = @. te = ke + pe
    periodize!(model, te)
    return ke, pe, te
end

function Jcons!((Jcons_x_, Jcons_z_), (; m), model, dynamics_scratch)
    (; mgr, domain) = model
    Jcons_x = similar!(Jcons_x_, m, size(m)[1:2]...)
    Jcons_z = similar!(Jcons_z_, m, size(m)[1:2]...)
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

function bq_energies!((gpe_, ie_, gpe_bz_), (; m, u, w), model, pe, dynamics_scratch)
    (; mgr, reference, fluid, space) = model
    (; p_R, ρ_R, v_R) = reference
    (; g) = space

    gpe = similar!(gpe_, m, size(m)[1:2]...)
    ie = similar!(ie_, m, size(m)[1:2]...)
    gpe_bz = similar!(gpe_bz_, m, size(m)[1:2]...)
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @vec for i in irange, j in jrange
            θ = fluid(:p, :consvar, :q).potential_temperature(p_R[j], dynamics_scratch.thermo.consvar[i, j], dynamics_scratch.thermo.comp[i, j])
            cp = fluid(:p, :consvar, :q).heat_capacity(p_R[j], dynamics_scratch.thermo.consvar[i, j], dynamics_scratch.thermo.comp[i, j])

            # ie = ρ₀(cₚθ - p₀v₀)
            ie[i, j] = ρ_R[j] * cp * (θ - fluid.T0) - fluid.p0 

            # gpe = pe - ie
            gpe[i, j] = pe[i, j] - ie[i, j]

            # ρ_R[j] * ( v(θ, q) * (p_R - p0) + Φij(model, i, j) ) + g * ρ0 * z 

            # gpe_bz = gz - bz
            b = buoyancy(model, model.buoyancy, dynamics_scratch.thermo.consvar, dynamics_scratch.thermo.comp, i, j)
            gpe_bz[i, j] = ρ_R[j] * (g - b) * Φij(model, i, j) / g
        end
    end
    periodize!(model, (gpe, ie, gpe_bz))
    return gpe, ie, gpe_bz
end

# kinetic energy tendency: KE =  ρ_R .(⟨u²⟩ᵢ + ⟨w²⟩ⱼ)/2
function kinetic_energy!(ke_, (; m, u, w), model)
    (; mgr, domain, reference) = model
    (; ρ_R) = reference
    ke = similar!(ke_, m, size(m)[1:2]...)
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            ke[i, j] = 0.5 * ρ_R[j] * (avg_x(abs2, u, i, j) + avg_z(abs2, w, i, j))
        end
    end
    periodize!(model, ke)
    return ke
end

# potential energy tendency: PE = ρ_R .[ h(p_R, s, q) + Φ - p_R.v_R ]
function potential_energy!(pe_, (; m), model)
    (; mgr, reference, fluid) = model
    (; p_R, ρ_R, v_R) = reference
    pe = similar!(pe_, m, size(m)[1:2]...)
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @vec for i in irange, j in jrange
            s = m[i, j , 1] * v_R[j]
            q = m[i, j , 2] * v_R[j]
            pe[i, j] = ρ_R[j] * (fluid(:p, :consvar, :q).specific_enthalpy(p_R[j], s, q) + Φij(model, i, j) ) - p_R[j] 
        end
    end
    periodize!(model, pe)
    return pe
end

# terms relating to heat/salt fluxes: π∇.Js, μ∇.Jq, Js.∇π, Jq.∇μ, ∇.(πJs), ∇.(μJq)
function irr_flux!((conjdivJcons_, chemdivJq_, Jconsgradconj_, Jqgradchem_), (; m), model, dynamics_scratch)
    (; mgr, domain, inv_dx, inv_dz) = model
    conjdivJcons    = similar!(conjdivJcons_, m, size(m)[1:2]...)
    chemdivJq       = similar!(chemdivJq_, m, size(m)[1:2]...)
    Jconsgradconj   = similar!(Jconsgradconj_, m, size(m)[1:2]...)
    Jqgradchem      = similar!(Jqgradchem_, m, size(m)[1:2]...)

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
function pdivu!(pdivu_, (; m), model, dynamics_scratch)
    (; mgr, reference, domain) = model
    (; ρ_R, p_R) = reference
    pdivu = similar!(pdivu_, m, size(m)[1:2]...)
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
function buoy_flux_Σ_ϕ!(buoy_flux_Σ_ϕ_, (; m, w), model, dynamics_scratch)
    (; mgr, reference, domain, fluid, space, inv_dz) = model
    (; ρ_R, p_R, ρ_R_J, v_R) = reference
    (; g ) = space
    buoy_flux_Σ_ϕ = similar!(buoy_flux_Σ_ϕ_, m, size(m)[1:2]...)
    @with mgr, let (irange, jrange) = (axes(buoy_flux_Σ_ϕ, 1), zrange(domain))
        @vec for i in irange, j in jrange
            s = m[i, j , 1] * v_R[j]
            q = m[i, j , 2] * v_R[j]
            δp = dynamics_scratch.momentum.φ[i, j] * ρ_R[j]
            v = fluid(:p, :consvar, :q).specific_volume(p_R[j], s, q)
            ∂v_R∂Φ = - v_R[j]^2 * inv_dz * dif_z(ρ_R_J, j) / g
            buoy_flux_Σ_ϕ[i, j] = ρ_R[j] * g * avg_z(w, i, j) * (2 - ρ_R[j] * v + δp * ∂v_R∂Φ)
        end
    end
    periodize!(model, buoy_flux_Σ_ϕ)
    return buoy_flux_Σ_ϕ
end

# buoyancy flux tendency: ρ_R . B . u.∇Φ
function buoy_flux_B!(buoy_flux_B_, (; m, w), model, dynamics_scratch)
    (; mgr, reference, domain, fluid, space) = model
    (; ρ_R, p_R, v_R) = reference
    (; g ) = space
    buoy_flux_B = similar!(buoy_flux_B_, m, size(m)[1:2]...)
    @with mgr, let (irange, jrange) = (axes(buoy_flux_B, 1), zrange(domain))
        @vec for i in irange, j in jrange
            s = m[i, j , 1] * v_R[j]
            q = m[i, j , 2] * v_R[j]
            v = fluid(:p, :consvar, :q).specific_volume(p_R[j], s, q)
            buoy_flux_B[i, j] = ρ_R[j] * g * avg_z(w, i, j) * (ρ_R[j] * v - 1)
        end
    end
    periodize!(model, buoy_flux_B)
    return buoy_flux_B
end

# thermodynamic budget terms: ∇.(ρu), π∇.(ρsu), μ∇.(ρqu)
function thermodynamic_budget_terms!((divru_, conjdivrsu_, chemdivrqu_), (; m, u, w), model, dynamics_scratch)
    (; mgr, domain, reference, inv_dx, inv_dz) = model
    (; ρ_R, ρ_R_J) = reference
    divru    = similar!(divru_, m, size(m)[1:2]...)
    conjdivrsu  = similar!(conjdivrsu_, m, size(m)[1:2]...)
    chemdivrqu  = similar!(chemdivrqu_, m, size(m)[1:2]...)

    (; mu, mw) = dynamics_scratch.advection

    rsu = @view mu[:, :, 1]
    rqu = @view mu[:, :, 2]
    rsw = @view mw[:, :, 1]
    rqw = @view mw[:, :, 2]

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            conj = dynamics_scratch.thermo.conjvar[i, j]
            chem = dynamics_scratch.thermo.chempot[i, j]
            divru[i, j] = ρ_R[j] * dif_x(u, i, j) * inv_dx + dif_z(ρ_R_J, w, i, j) * inv_dz
            conjdivrsu[i, j] = conj * ( dif_x(rsu, i, j) * inv_dx + dif_z(rsw, i, j) * inv_dz )
            chemdivrqu[i, j] = chem * ( dif_x(rqu, i, j) * inv_dx + dif_z(rqw, i, j) * inv_dz )
        end
    end
    periodize!(model, (divru, conjdivrsu, chemdivrqu))
    return divru, conjdivrsu, chemdivrqu
end

## LOOP

# energetics loop
function energetics(ds, keys, model::AN2D{F}, params) where {F}
    (; Nslice, slice_size, energetics_variables) = params

    # define variables from energetics_variables
    # assume coordinates x, z, t already defined
    for sym in energetics_variables
        ~(string(sym) in NCDatasets.keys(ds)) && defVar(ds, string(sym), F, ("x", "z", "t"))
    end

    # initialise energetics state (; s, q, u, w ) from simulation dataset
    energetics_state0 = initialise_energetics_state(ds, keys)
    energetics_state = deepcopy(energetics_state0)

    # initialise dynamics scratch space
    dynamics_dstate, dynamics_scratch = CFTimeSchemes.tendencies!(void, void, model, energetics_state0, nothing)

    # initialise energetics scratch space
    energetics_scratch = energetics_tendencies!(void, model, energetics_state0, dynamics_scratch)

    # run loop
    for t_iter = 1:Nslice
        # update energetics state from dataset
        update_energetics_state!(energetics_state, ds, keys, t_iter)

        # compute dynamics scratch space
        dynamics_dstate, dynamics_scratch = CFTimeSchemes.tendencies!(dynamics_dstate, dynamics_scratch, model, energetics_state, t_iter * slice_size)

        # compute energetics
        energetics_tendencies!(energetics_scratch, model, energetics_state, dynamics_scratch)

        # write data from energetics diagnostics
        session = open(energetics_diagnostics(); energetics_scratch, model)
        let data = CookBooks.get(session, energetics_variables)
            for sym in energetics_variables
                ds[sym][:, :, t_iter] = data[sym]
            end
        end
        close(session);

        # advance progress meter
        progress(t_iter, slice_size, Nslice * slice_size)
    end
end

## TOOLS

# geopotential at cell centre
function Φij(model::AN2D, i, j)
    return model.space.g * (CFBoxes.zpoint(model.domain, model.dz, j) - model.space.Lz)
end

# initialisation
function initialise_energetics_state(ds, keys)
    energetics_state = (;
        m = ds[keys.m][:, :, :, 1],
        u = ds[keys.u][:, :, 1],
        w = ds[keys.w][:, :, 1]
        )
    return energetics_state
end

# update
function update_energetics_state!(energetics_state, ds, keys, t_iter)
    (; m, u, w) = energetics_state
    m[:, :, :] .= ds[keys.m][:, :, :, t_iter]
    u[:, :] .= ds[keys.u][:, :, t_iter]
    w[:, :] .= ds[keys.w][:, :, t_iter]
end

# construct state from (s, q, u, w)
function construct_state!(ds, keys, model::AN2D{F}) where {F}
    ~("mi" in NCDatasets.keys(ds.dim)) && defDim(ds, "mi", 2)
    ~("m" in NCDatasets.keys(ds)) && defVar(ds, "m", F, ("x", "z", "mi", "t"))
    
    (; mgr, domain, reference) = model
    (; ρ_R) = reference

    Nslice = ds.attrib["Nslice"]

    dsm1 = @views ds["m"][:, :, 1, :]
    dsm2 = @views ds["m"][:, :, 2, :]

    let jrange = zrange(domain)
        for j in jrange
            dsm1[:, j, :] = ds[keys.s][:, j, :] .* ρ_R[j]
            dsm2[:, j, :] = ds[keys.q][:, j, :] .* ρ_R[j]
        end
    end
    keys = (; m = :m, keys...)
    return ds, keys
end

## DIAGNOSTICS

# diagnostics for energetics
energetics_diagnostics() = CookBook(;
    ke = energetics_scratch -> energetics_scratch.energies.ke,
    pe = energetics_scratch -> energetics_scratch.energies.pe,
    te = energetics_scratch -> energetics_scratch.energies.te,
    gpe = energetics_scratch -> energetics_scratch.bq_energies.gpe,
    ie = energetics_scratch -> energetics_scratch.bq_energies.ie,
    gpe_bz = energetics_scratch -> energetics_scratch.bq_energies.gpe_bz,
    viscous_diss = energetics_scratch -> energetics_scratch.irreversible.viscous_diss,
    entropy_prod = energetics_scratch -> energetics_scratch.irreversible.entropy_prod,
    pdivu = energetics_scratch -> energetics_scratch.dynamic.pdivu,
    Jcons_x = energetics_scratch -> energetics_scratch.irreversible.Jcons_x,
    Jcons_z = energetics_scratch -> energetics_scratch.irreversible.Jcons_z,
    conjdivJcons = energetics_scratch -> energetics_scratch.irreversible.conjdivJcons,
    chemdivJq = energetics_scratch -> energetics_scratch.irreversible.chemdivJq,
    Jconsgradconj = energetics_scratch -> energetics_scratch.irreversible.Jconsgradconj,
    Jqgradchem = energetics_scratch -> energetics_scratch.irreversible.Jqgradchem,
    buoy_flux_Σ_ϕ = energetics_scratch -> energetics_scratch.dynamic.buoy_flux_Σ_ϕ,
    buoy_flux_B = energetics_scratch -> energetics_scratch.dynamic.buoy_flux_B,
    divru = energetics_scratch -> energetics_scratch.thermodynamic.divru,
    conjdivrsu = energetics_scratch -> energetics_scratch.thermodynamic.conjdivrsu,
    chemdivrqu = energetics_scratch -> energetics_scratch.thermodynamic.chemdivrqu
)