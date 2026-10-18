# Traditional energetics of AN2D. Staging area: moved out of CFAnelastics (project-specific, not model code), to be
# reworked here. Load after the model packages:  include("<path to>/energetics/energetics.jl")
using CFAnelastics: CFAnelastics, AN2D, buoyancy      # buoyancy: internal to CFAnelastics
using CFBoxes
using CFBoxes: slice, across
using ManagedLoops: @vec, @with
using MutatingOrNot: void

"""
    energetics_scratch = energetics_tendencies!(energetics_scratch, model::AN2D, state, dynamics_scratch)

Terms of the traditional energy budget of AN2D at `state`, from the scratch `dynamics_scratch` of `tendencies!`:
energies, Boussinesq energies, irreversible terms, pressure work and buoyancy fluxes, and thermodynamic budget
terms, in the named tuples of `energetics_scratch` (allocated when `void`).
"""
function energetics_tendencies!(energetics_scratch, model::AN2D, state, dynamics_scratch)
    # energies
    (; ke, pe, te) = energetics_scratch.energies
    ke, pe, te = energies!((ke, pe, te), model, state)

    # Boussinesq GPE, IE
    (; gpe, ie, gpe_bz) = energetics_scratch.bq_energies
    gpe, ie, gpe_bz = bq_energies!((gpe, ie, gpe_bz), state, model, pe, dynamics_scratch)

    # viscous dissipation: ρε
    (; viscous_diss) = energetics_scratch.irreversible
    viscous_diss = @. viscous_diss = dynamics_scratch.irreversible.ρε

    # irreversible entropy production: πσ = ρε - consflux·∇π - qflux·∇μ
    (; entropy_prod) = energetics_scratch.irreversible
    entropy_prod = @. entropy_prod = dynamics_scratch.thermo.conjvar * dynamics_scratch.irreversible.σ_cons

    # consvar fluxes averaged onto cell centers
    (; consflux_x, consflux_z) = energetics_scratch.irreversible
    consflux_x, consflux_z = consflux!((consflux_x, consflux_z), state, model, dynamics_scratch)

    # terms relating to consvar/composition fluxes: π∇·consflux, μ∇·qflux, consflux·∇π, qflux·∇μ
    (; conjdivconsflux, chemdivqflux, consfluxgradconj, qfluxgradchem, consfluxgradconj_x, consfluxgradconj_z, qfluxgradchem_x, qfluxgradchem_z) = energetics_scratch.irreversible
    conjdivconsflux, chemdivqflux, consfluxgradconj, qfluxgradchem, consfluxgradconj_x, consfluxgradconj_z, qfluxgradchem_x, qfluxgradchem_z =
        irr_flux!((conjdivconsflux, chemdivqflux, consfluxgradconj, qfluxgradchem, consfluxgradconj_x, consfluxgradconj_z, qfluxgradchem_x, qfluxgradchem_z), state, model, dynamics_scratch)

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
        irreversible = (; consflux_x, consflux_z, viscous_diss, entropy_prod, conjdivconsflux, chemdivqflux, consfluxgradconj, qfluxgradchem,
                        consfluxgradconj_x, consfluxgradconj_z, qfluxgradchem_x, qfluxgradchem_z),
        dynamic = (; pdivu, buoy_flux_Σ_ϕ, buoy_flux_B),
        thermodynamic = (; divru, conjdivrsu, chemdivrqu)
    )
    return energetics_scratch
end

"""
    ke, pe, te = energies!((ke, pe, te), model::AN2D, state)

Energies per unit volume [J m⁻³] at the cell centers: kinetic ke (`kinetic_energy!`), potential and internal pe
(`potential_energy!`), and total te = ke + pe. Periodized.
"""
function energies!((ke, pe, te_), model::AN2D{F}, (; m, u, w)) where {F}
    ke = kinetic_energy!(ke, model, (; m, u, w))
    pe = potential_energy!(pe, model, (; m))
    te = alloc!(te_, F, model.mgr, model.domain, (Cell(), Cell()))
    te = @. te = ke + pe
    periodize!(model.domain, te)
    return ke, pe, te
end

"""
    consflux_z, consflux_x = consflux!((consflux_x, consflux_z), state, model::AN2D, dynamics_scratch)

Fluxes of consvar of `dynamics_scratch` averaged from the faces to the cells.
"""
function consflux!((consflux_x_, consflux_z_), (; m), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, domain) = model
    (; ℑ) = operators(model.spatial_scheme)
    consflux_x = alloc!(consflux_x_, F, mgr, domain, (Cell(), Cell()))
    consflux_z = alloc!(consflux_z_, F, mgr, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (cells(domain, 1), axes(consflux_x, 2))
        @inbounds @vec for i in irange, j in jrange
            consflux_x[i, j] = ℑ(dynamics_scratch.irreversible.consflux_x, (1,), (Cell(),), i, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(consflux_z, 1), cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            consflux_z[i, j] = ℑ(dynamics_scratch.irreversible.consflux_z, (2,), (Cell(),), i, j)
        end
    end
    return consflux_z, consflux_x
end

"""
    gpe, ie, gpe_bz = bq_energies!((gpe, ie, gpe_bz), state, model::AN2D, pe, dynamics_scratch)

Boussinesq energies at the cells: the internal energy ie = ρ_R Cp (θ - T₀) - p₀, the gravitational potential energy
gpe = pe - ie, and gpe_bz = ρ_R (g - b) Φ / g, with the buoyancy b of the model's scheme. Periodized.
"""
function bq_energies!((gpe_, ie_, gpe_bz_), (; m, u, w), model::AN2D{F}, pe, dynamics_scratch) where {F}
    (; mgr, reference, fluid, space) = model
    (; p_R, ρ_R) = reference
    (; g) = space

    gpe = alloc!(gpe_, F, model.mgr, model.domain, (Cell(), Cell()))
    ie = alloc!(ie_, F, model.mgr, model.domain, (Cell(), Cell()))
    gpe_bz = alloc!(gpe_bz_, F, model.mgr, model.domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @inbounds @vec for i in irange, j in jrange
            θ = fluid(:p, :consvar, :q).potential_temperature(p_R[j], dynamics_scratch.thermo.consvar[i, j], dynamics_scratch.thermo.q[i, j])
            cp = fluid(:p, :consvar, :q).heat_capacity(p_R[j], dynamics_scratch.thermo.consvar[i, j], dynamics_scratch.thermo.q[i, j])

            # ie = ρ₀(cₚθ - p₀v₀)
            ie[i, j] = ρ_R[j] * cp * (θ - fluid.T0) - fluid.p0

            # gpe = pe - ie
            gpe[i, j] = pe[i, j] - ie[i, j]

            # gpe_bz = gz - bz
            b = buoyancy(model.buoyancy_scheme, g, ρ_R[j], fluid(:p, :consvar, :q).specific_volume(p_R[j], dynamics_scratch.thermo.consvar[i, j], dynamics_scratch.thermo.q[i, j]))
            gpe_bz[i, j] = ρ_R[j] * (g - b) * Φij(model, i, j) / g
        end
    end
    periodize!(model.domain, (gpe, ie, gpe_bz))
    return gpe, ie, gpe_bz
end

"""
    ke = kinetic_energy!(ke, model::AN2D, state)

Kinetic energy per unit volume ke = ρ_R (ℑx(u²) + ℑz(w²)) / 2 [J m⁻³] at the cells. Periodized.
"""
function kinetic_energy!(ke_, model::AN2D{F}, (; m, u, w)) where {F}
    (; mgr, domain, reference) = model
    (; ρ_R) = reference
    ke = alloc!(ke_, F, mgr, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (cells(domain, 1), cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            ke[i, j] = ρ_R[j] * (ℑ((u, u), (1,), (Cell(),), i, j) + ℑ((w, w), (2,), (Cell(),), i, j)) / 2
        end
    end
    periodize!(domain, ke)
    return ke
end

"""
    pe = potential_energy!(pe, model::AN2D, state)

Potential and internal energy per unit volume pe = ρ_R (h(p_R, consvar, q) + Φ) - p_R [J m⁻³] at the cells.
Periodized.
"""
function potential_energy!(pe_, model::AN2D{F}, (; m)) where {F}
    (; mgr, reference, fluid) = model
    (; p_R, ρ_R) = reference
    pe = alloc!(pe_, F, model.mgr, model.domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @inbounds @vec for i in irange, j in jrange
            s = m[i, j , 1] / ρ_R[j]
            q = m[i, j , 2] / ρ_R[j]
            pe[i, j] = ρ_R[j] * (fluid(:p, :consvar, :q).specific_enthalpy(p_R[j], s, q) + Φij(model, i, j) ) - p_R[j]
        end
    end
    periodize!(model.domain, pe)
    return pe
end

"""
    terms = irr_flux!(terms, state, model::AN2D, dynamics_scratch)

Terms of the consvar and composition fluxes at the cells: π ∇·consflux, μ ∇·qflux, and consflux·∇π, qflux·∇μ
averaged from their face values (zero on wall faces: zero-gradient halo of conjvar, chempot); also the face values.
"""
function irr_flux!((conjdivconsflux_, chemdivqflux_, consfluxgradconj_, qfluxgradchem_, consfluxgradconj_x_, consfluxgradconj_z_, qfluxgradchem_x_, qfluxgradchem_z_),
                   (; m), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; ℑ, δ) = operators(model.spatial_scheme)
    conjdivconsflux     = alloc!(conjdivconsflux_, F, mgr, domain, (Cell(), Cell()))
    chemdivqflux        = alloc!(chemdivqflux_, F, mgr, domain, (Cell(), Cell()))
    consfluxgradconj    = alloc!(consfluxgradconj_, F, mgr, domain, (Cell(), Cell()))
    qfluxgradchem       = alloc!(qfluxgradchem_, F, mgr, domain, (Cell(), Cell()))
    consfluxgradconj_x  = alloc!(consfluxgradconj_x_, F, mgr, domain, (Face(), Cell()))
    consfluxgradconj_z  = alloc!(consfluxgradconj_z_, F, mgr, domain, (Cell(), Face()))
    qfluxgradchem_x     = alloc!(qfluxgradchem_x_, F, mgr, domain, (Face(), Cell()))
    qfluxgradchem_z     = alloc!(qfluxgradchem_z_, F, mgr, domain, (Cell(), Face()))

    (; consflux_x, consflux_z, qflux_x, qflux_z) = dynamics_scratch.irreversible
    (; conjvar, chempot) = dynamics_scratch.thermo

    # faces: consflux ∂π and qflux ∂μ
    @with mgr, let (irange, jrange) = (faces(domain, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            consfluxgradconj_x[I, j] = inv_dx * consflux_x[I, j] * δ(conjvar, (1,), (Face(),), I, j)
            qfluxgradchem_x[I, j]    = inv_dx * qflux_x[I, j] * δ(chempot, (1,), (Face(),), I, j)
        end
    end
    @with mgr, let (irange, jrange) = (cells(domain, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            consfluxgradconj_z[i, J] = inv_dz * consflux_z[i, J] * δ(conjvar, (2,), (Face(),), i, J)
            qfluxgradchem_z[i, J]    = inv_dz * qflux_z[i, J] * δ(chempot, (2,), (Face(),), i, J)
        end
    end
    # cells
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            conjdivconsflux[i, j]  = conjvar[i, j] * ( δ(consflux_x, (1,), (Cell(),), i, j) * inv_dx + δ(consflux_z, (2,), (Cell(),), i, j) * inv_dz )
            chemdivqflux[i, j]     = chempot[i, j] * ( δ(qflux_x, (1,), (Cell(),), i, j) * inv_dx + δ(qflux_z, (2,), (Cell(),), i, j) * inv_dz )
            consfluxgradconj[i, j] = ℑ(consfluxgradconj_x, (1,), (Cell(),), i, j) + ℑ(consfluxgradconj_z, (2,), (Cell(),), i, j)
            qfluxgradchem[i, j]    = ℑ(qfluxgradchem_x, (1,), (Cell(),), i, j) + ℑ(qfluxgradchem_z, (2,), (Cell(),), i, j)
        end
    end
    periodize!(domain, (conjdivconsflux, chemdivqflux, consfluxgradconj, qfluxgradchem))
    return conjdivconsflux, chemdivqflux, consfluxgradconj, qfluxgradchem, consfluxgradconj_x, consfluxgradconj_z, qfluxgradchem_x, qfluxgradchem_z
end

"""
    pdivu = pdivu!(pdivu, state, model::AN2D, dynamics_scratch)

Pressure work p ∇·u at the cells, with p = p_R + ρ_R φ. Periodized.
"""
function pdivu!(pdivu_, (; m), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, reference, domain) = model
    (; ρ_R, p_R) = reference
    pdivu = alloc!(pdivu_, F, mgr, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (axes(pdivu, 1), axes(pdivu, 2))
        @inbounds @vec for i in irange, j in jrange
            p = p_R[j] + ρ_R[j] * dynamics_scratch.φ[i, j]
            pdivu[i, j] = p * (dynamics_scratch.derivatives.∂u_x[i, j] + dynamics_scratch.derivatives.∂w_z[i, j])
        end
    end
    periodize!(domain, pdivu)
    return pdivu
end

"""
    flux = buoy_flux_Σ_ϕ!(flux, state, model::AN2D, dynamics_scratch)

Buoyancy flux ρ_R Σ_ϕ u·∇Φ at the interior cells (zero elsewhere). Periodized.
"""
function buoy_flux_Σ_ϕ!(buoy_flux_Σ_ϕ_, (; m, w), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, reference, domain, fluid, space, inv_dz) = model
    (; ρ_R, p_R) = reference
    (; g ) = space
    buoy_flux_Σ_ϕ = fill!(alloc!(buoy_flux_Σ_ϕ_, F, mgr, domain, (Cell(), Cell())), 0)    # interior cells only (∂ρ_R/∂z)
    @with mgr, let (irange, jrange) = (axes(buoy_flux_Σ_ϕ, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            s = m[i, j , 1] / ρ_R[j]
            q = m[i, j , 2] / ρ_R[j]
            δp = dynamics_scratch.φ[i, j] * ρ_R[j]
            v = fluid(:p, :consvar, :q).specific_volume(p_R[j], s, q)
            ∂v_R∂Φ = - inv(ρ_R[j])^2 * inv_dz * (ℑ(ρ_R, (2,), (Face(),), j+1) - ℑ(ρ_R, (2,), (Face(),), j)) / g
            buoy_flux_Σ_ϕ[i, j] = ρ_R[j] * g * ℑ(w, (2,), (Cell(),), i, j) * (2 - ρ_R[j] * v + δp * ∂v_R∂Φ)
        end
    end
    periodize!(domain, buoy_flux_Σ_ϕ)
    return buoy_flux_Σ_ϕ
end

"""
    flux = buoy_flux_B!(flux, state, model::AN2D, dynamics_scratch)

Buoyancy flux ρ_R g ℑz(w) (ρ_R v - 1) at the cells. Periodized.
"""
function buoy_flux_B!(buoy_flux_B_, (; m, w), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, reference, domain, fluid, space) = model
    (; ρ_R, p_R) = reference
    (; g ) = space
    buoy_flux_B = alloc!(buoy_flux_B_, F, mgr, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (axes(buoy_flux_B, 1), cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            s = m[i, j , 1] / ρ_R[j]
            q = m[i, j , 2] / ρ_R[j]
            v = fluid(:p, :consvar, :q).specific_volume(p_R[j], s, q)
            buoy_flux_B[i, j] = ρ_R[j] * g * ℑ(w, (2,), (Cell(),), i, j) * (ρ_R[j] * v - 1)
        end
    end
    periodize!(domain, buoy_flux_B)
    return buoy_flux_B
end

"""
    divru, conjdivrsu, chemdivrqu = thermodynamic_budget_terms!(terms, state, model::AN2D, dynamics_scratch)

Thermodynamic budget terms at the cells: ∇·(ρ_R u) (interior cells only), π ∇·(ρ_R consvar u) and
μ ∇·(ρ_R q u). Periodized.
"""
function thermodynamic_budget_terms!((divru_, conjdivrsu_, chemdivrqu_), (; m, u, w), model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, domain, reference, inv_dx, inv_dz) = model
    (; ρ_R) = reference
    divru    = fill!(alloc!(divru_, F, mgr, domain, (Cell(), Cell())), 0)        # interior cells only (ρ_R on w-faces)
    conjdivrsu  = alloc!(conjdivrsu_, F, mgr, domain, (Cell(), Cell()))
    chemdivrqu  = alloc!(chemdivrqu_, F, mgr, domain, (Cell(), Cell()))

    (; mu, mw) = dynamics_scratch.advection

    rsu = @view mu[:, :, 1]
    rqu = @view mu[:, :, 2]
    rsw = @view mw[:, :, 1]
    rqw = @view mw[:, :, 2]

    @with mgr, let (irange, jrange) = (cells(domain, 1), cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            conj = dynamics_scratch.thermo.conjvar[i, j]
            chem = dynamics_scratch.thermo.chempot[i, j]
            conjdivrsu[i, j] = conj * ( δ(rsu, (1,), (Cell(),), i, j) * inv_dx + δ(rsw, (2,), (Cell(),), i, j) * inv_dz )
            chemdivrqu[i, j] = chem * ( δ(rqu, (1,), (Cell(),), i, j) * inv_dx + δ(rqw, (2,), (Cell(),), i, j) * inv_dz )
        end
    end
    @with mgr, let (irange, jrange) = (cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            divru[i, j] = ρ_R[j] * δ(u, (1,), (Cell(),), i, j) * inv_dx + (ℑ(ρ_R, (2,), (Face(),), j+1) * w[i, j+1] - ℑ(ρ_R, (2,), (Face(),), j) * w[i, j]) * inv_dz
        end
    end
    periodize!(domain, (divru, conjdivrsu, chemdivrqu))
    return divru, conjdivrsu, chemdivrqu
end

"""
    Φ = Φij(model::AN2D, i, j)

Geopotential Φ = g (z - Lz) [J kg⁻¹] at the cells of level j.
"""
function Φij(model::AN2D, i, j)
    return model.space.g * (CFBoxes.cellpoint(model.domain, model.dz, 2, j) - model.space.Lz)
end

