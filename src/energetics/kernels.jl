## KERNELS OF THE ENERGETICS DIAGNOSTICS
#
# The energy budget of energy.tex as diagnostics of AN2D and FC2D: the recipes are in entries.jl, the kernels here.
# K and P are specific energies [J kg⁻¹], the energy densities ρK, ρP [J m⁻³] and every term of the budget are per unit
# volume [W m⁻³], with ρ = ρ_R for AN2D. Fields on the cells carry no suffix; their components on the x and z faces end
# in `_x` and `_z`.
#
# Every term is built from the model's own operators (`ℑ`, `δ` of the spatial scheme): a face product -mass_flux ·
# ∇(potential) is averaged to the cells, as the model does for ρε. With the product rule of these operators,
#     δ(ℑ(a) b) = a δb + ℑ(b δa),
# the budget of ρP closes at each cell, and that of ρK in the volume integral (`integrated_budget_residual`).
#
# Cell fields are computed at the interior cells and face fields at the interior faces (both walls of a bounded
# dimension), then their halo is filled: zero-gradient for cells, periodic for faces (the wall faces are values, not
# halo). The mass flux, the tracer fluxes and the vortex force are those of the tendencies, so every term has the
# discretization of the model. The terms on the cells that are built from face products take `faces = (Fx, Fz)`, scratch
# arrays on the x and z faces that the diagnostics share.

"""
    (mu, mw) = tracer_flux((; mu, mw), k)

Advective flux of the k-th advected density (ρ_R ϑ u, ρ_R ϑ w), as views on the x and z faces, from the `advection`
scratch of the tendencies. The densities are (ρ_R ϑ, ρ_R q) for AN2D and (ρ, ρ ϑ, ρ q) for FC2D.
"""
tracer_flux((; mu, mw), k) = (view(mu, :, :, k), view(mw, :, :, k))

"""
    out = pointwise!(out, mgr, f, a...)

out = f(a...) pointwise on arrays of the same grid, halo included, by the manager `mgr`.
"""
function pointwise!(out, mgr, f::Fun, a::Vararg{Any, N}) where {Fun, N}
    @with mgr, let (irange, jrange) = (axes(out, 1), axes(out, 2))
        @inbounds @vec for i in irange, j in jrange
            out[i, j] = f(map(x -> @inbounds(x[i, j]), a)...)
        end
    end
    return out
end

"""
    out = cellwise!(out, model, f, a...)

out = f(a...) pointwise on cell arrays at the interior cells, by the manager of `model`, with a zero-gradient halo: the
halo of the arrays `a` is never read.
"""
function cellwise!(out, model, f::Fun, a::Vararg{Any, N}) where {Fun, N}
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            out[i, j] = f(map(x -> @inbounds(x[i, j]), a)...)
        end
    end
    return fill_halo!(domain, out, ZeroGradient())
end

"""
    integral = domain_integral(model, field)

Integral of a cell `field` over the interior cells, per unit length along y: sum(field) Δx Δz.
"""
domain_integral(model, field) = sum(interior(model.domain, field)) * model.dx * model.dz

# Energies

"""
    h = specific_enthalpy!(h, model, p, (; consvar, q))

Specific enthalpy h(p, consvar, q) [J kg⁻¹] at the interior cells, at the pressure `p`: an array, or the profile p_R(z).
"""
function specific_enthalpy!(h, model, p, (; consvar, q))
    (; mgr, domain, fluid, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            # ℑ with no direction reads an array at (i, j), a profile at j
            h[i, j] = fluid(:p, :consvar, :q).specific_enthalpy(ℑ(p, (), (), i, j), consvar[i, j], q[i, j])
        end
    end
    return fill_halo!(domain, h, ZeroGradient())
end

"""
    P = potential_energy!(P, model, h, ρ, p)

Potential energy P = h - p/ρ + g (z - z0) [J kg⁻¹] at the interior cells, with the density and pressure of the energy:
those of the state for FC2D, the profiles ρ_R(z), p_R(z) for AN2D.
"""
function potential_energy!(P, model, h, ρ, p)
    (; mgr, domain, space, dz, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    (; g, z0) = space
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            P[i, j] = h[i, j] - ℑ(p, (), (), i, j) / ℑ(ρ, (), (), i, j) + g * (cellpoint(domain, dz, 2, j) - z0)
        end
    end
    return fill_halo!(domain, P, ZeroGradient())
end

"""
    Σ_B = bernoulli_potential!(Σ_B, model, h, scratch)

Σ - ϑ Σ_ϑ - q Σ_q = h - consvar conjvar - q chempot + g (z - z0) [J kg⁻¹] at the interior cells, plus the kinematic
pressure anomaly φ = (p - p_R)/ρ_R for AN2D. With the thermodynamic forcing of the model it is B + g (z - z0) (+ φ).
"""
function bernoulli_potential!(Σ_B, model::FC2D, h, (; thermo))
    (; mgr, domain, space, dz) = model
    (; g, z0) = space
    (; consvar, q, conjvar, chempot) = thermo
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            Σ_B[i, j] = h[i, j] - consvar[i, j] * conjvar[i, j] - q[i, j] * chempot[i, j] + g * (cellpoint(domain, dz, 2, j) - z0)
        end
    end
    return fill_halo!(domain, Σ_B, ZeroGradient())
end
function bernoulli_potential!(Σ_B, model::AN2D, h, (; thermo, φ))
    (; mgr, domain, space, dz) = model
    (; g, z0) = space
    (; consvar, q, conjvar, chempot) = thermo
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            Σ_B[i, j] = h[i, j] - consvar[i, j] * conjvar[i, j] - q[i, j] * chempot[i, j] + φ[i, j] + g * (cellpoint(domain, dz, 2, j) - z0)
        end
    end
    return fill_halo!(domain, Σ_B, ZeroGradient())
end

# Fluxes of K and P on the faces. Each is a product of a flux of the tendencies and a cell quantity averaged to the face.
# A kernel on the faces takes the direction `dir`, Val(1) for the x-faces (u) and Val(2) for the z-faces (w), and is
# computed at the interior faces of that direction, both walls of a bounded dimension included.

"""
    (irange, jrange) = face_ranges(domain, dir)
    dims = along(dir)
    inv_Δ = inv_spacing(model, dir)

The index ranges of the interior faces of direction `dir` (Val(1): x-faces, `I, j`; Val(2): z-faces, `i, J`), the
dimensions `(d,)` of the operators `ℑ`, `δ` along it, and the inverse grid spacing 1/Δx or 1/Δz [m⁻¹] along it.
"""
face_ranges(domain, ::Val{1}) = (interior_faces(domain, 1), interior_cells(domain, 2))
face_ranges(domain, ::Val{2}) = (interior_cells(domain, 1), interior_faces(domain, 2))
along(::Val{d}) where d = (d,)
inv_spacing(model, ::Val{1}) = model.inv_dx
inv_spacing(model, ::Val{2}) = model.inv_dz

"""
    out = facewise!(out, model, dir, f, a...)

out = f(a...) pointwise on face arrays of direction `dir` at its interior faces, by the manager of `model`, then
periodized: the halo of the arrays `a` is never read.
"""
function facewise!(out, model, dir, f::Fun, a::Vararg{Any, N}) where {Fun, N}
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = face_ranges(domain, dir)
        @inbounds @vec for i in irange, j in jrange
            out[i, j] = f(map(x -> @inbounds(x[i, j]), a)...)
        end
    end
    return periodize!(domain, out)
end

"""
    J = advective_flux!(J, model, dir, flux, a)

Flux of the cell quantity `a` carried by a flux on the faces of direction `dir`, J = flux ℑ(a). With the mass flux ρu:
the reversible flux of K, ρ K u (`a` = K); with the velocity: J_C = (p - p_R) u (`a` = p - p_R).
"""
function advective_flux!(J, model, dir, flux, a)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = face_ranges(domain, dir)
        @inbounds @vec for i in irange, j in jrange
            J[i, j] = flux[i, j] * ℑ(a, along(dir), (Face(),), i, j)
        end
    end
    return periodize!(domain, J)
end

"""
    J = stress_work_flux!(J, model, dir, uτ)

Irreversible flux of K on the faces of direction `dir`, -u·τ, from the work `uτ` of the viscous stress of the
tendencies.
"""
stress_work_flux!(J, model, dir, uτ) = facewise!(J, model, dir, uτ -> -uτ, uτ)

"""
    J = potential_flux_rev!(J, model, dir, mass_flux, tracer_flux, k, Σ_B, conjvar, chempot)

Reversible flux of P on the faces of direction `dir`, ρ Σ u = ρu ℑ(Σ_B) + ρϑu ℑ(Σ_ϑ) + ρqu ℑ(Σ_q), from the mass flux
and the tracer fluxes `tracer_flux[:, :, k]` (ϑ) and `tracer_flux[:, :, k + 1]` (q) of the tendencies (`mu` or `mw`).
"""
function potential_flux_rev!(J, model, dir, mass_flux, tracer_flux, k, Σ_B, conjvar, chempot)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = face_ranges(domain, dir)
        @inbounds @vec for i in irange, j in jrange
            J[i, j] = mass_flux[i, j] * ℑ(Σ_B, along(dir), (Face(),), i, j) +
                      tracer_flux[i, j, k] * ℑ(conjvar, along(dir), (Face(),), i, j) +
                      tracer_flux[i, j, k+1] * ℑ(chempot, along(dir), (Face(),), i, j)
        end
    end
    return periodize!(domain, J)
end

"""
    J = potential_flux_irr!(J, model, dir, consflux, qflux, conjvar, chempot)

Irreversible flux of P on the faces of direction `dir`, Σ_ϑ J_ϑ + Σ_q J_q = ℑ(conjvar) consflux + ℑ(chempot) qflux. On a
wall face, ℑ is the value of the cell (zero-gradient halo), as for the energy the wall delivers in the tendencies.
"""
function potential_flux_irr!(J, model, dir, consflux, qflux, conjvar, chempot)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = face_ranges(domain, dir)
        @inbounds @vec for i in irange, j in jrange
            J[i, j] = consflux[i, j] * ℑ(conjvar, along(dir), (Face(),), i, j) + qflux[i, j] * ℑ(chempot, along(dir), (Face(),), i, j)
        end
    end
    return periodize!(domain, J)
end

"""
    div = divergence!(div, model, Fx, Fz)

Divergence δx Fx / Δx + δz Fz / Δz of a vector on the faces, at the interior cells, with a zero-gradient halo.
"""
function divergence!(div, model, Fx, Fz)
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; δ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            div[i, j] = inv_dx * δ(Fx, (1,), (Cell(),), i, j) + inv_dz * δ(Fz, (2,), (Cell(),), i, j)
        end
    end
    return fill_halo!(domain, div, ZeroGradient())
end

# Terms on the cells built from face products

"""
    C = faces_to_cells!(C, model, Fx, Fz)

C = ℑx(Fx) + ℑz(Fz): the terms on the faces averaged to the interior cells, as the model does for ρε.
"""
function faces_to_cells!(C, model, Fx, Fz)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            C[i, j] = ℑ(Fx, (1,), (Cell(),), i, j) + ℑ(Fz, (2,), (Cell(),), i, j)
        end
    end
    return fill_halo!(domain, C, ZeroGradient())
end

"""
    C = gradient_product!(C, model, faces, (flux_x, flux_z), Σ)
    F = face_gradient_product!(F, model, dir, flux, Σ)

-flux·∇Σ [W m⁻³] at the interior cells: the face product F = -flux δΣ / Δ along each direction, for a flux on the faces
and a potential Σ at the cells (`face_gradient_product!`, on the faces of direction `dir`), averaged to the cells. It is
a conversion C_ϑ, C_q, C_Σ (mass or tracer flux of the tendencies, potential Σ_ϑ, Σ_q or Σ_B) or a heating Q_ϑ, Q_q
(diffusive flux).
"""
function gradient_product!(C, model, (Fx, Fz), (flux_x, flux_z), Σ)
    face_gradient_product!(Fx, model, Val(1), flux_x, Σ)
    face_gradient_product!(Fz, model, Val(2), flux_z, Σ)
    return faces_to_cells!(C, model, Fx, Fz)
end
function face_gradient_product!(F, model, dir, flux, Σ)
    (; mgr, domain, spatial_scheme) = model
    (; δ) = operators(spatial_scheme)
    inv_Δ = inv_spacing(model, dir)
    @with mgr, let (irange, jrange) = face_ranges(domain, dir)
        @inbounds @vec for i in irange, j in jrange
            F[i, j] = - flux[i, j] * inv_Δ * δ(Σ, along(dir), (Face(),), i, j)
        end
    end
    return periodize!(domain, F)
end

"""
    VF = vortex_work!(VF, model, faces, (; vortex_x, vortex_z), mass_flux_x, mass_flux_z)

Work of the vortex force, VF = -ρu·(ω×u) [W m⁻³] at the interior cells: minus the mass flux times the vortex force
(vortex_x, vortex_z) of the tendencies, on the faces, averaged to the cells.
"""
function vortex_work!(VF, model, (Fx, Fz), (; vortex_x, vortex_z), mass_flux_x, mass_flux_z)
    facewise!(Fx, model, Val(1), (ρu, ω) -> -ρu * ω, mass_flux_x, vortex_x)
    facewise!(Fz, model, Val(2), (ρw, ω) -> -ρw * ω, mass_flux_z, vortex_z)
    return faces_to_cells!(VF, model, Fx, Fz)
end

"""
    dK = kinetic_energy_rate!(dK, model, faces, (u, w), (du, dw))

Tendency of K = (ℑx(u²) + ℑz(w²))/2, ℑx(u ∂ₜu) + ℑz(w ∂ₜw) [W kg⁻¹] at the interior cells. The tendencies du, dw have no
halo faces in a periodic dimension, so the products on the faces are periodized rather than read there.
"""
function kinetic_energy_rate!(dK, model, (Fx, Fz), (u, w), (du, dw))
    facewise!(Fx, model, Val(1), *, u, du)
    facewise!(Fz, model, Val(2), *, w, dw)
    return faces_to_cells!(dK, model, Fx, Fz)
end

# Tendencies of ρK and ρP

"""
    dρK = kinetic_energy_tendency!(dρK, model::FC2D, ρ, K, dK, dρ)

Tendency of ρ K, ρ dK + K dρ [W m⁻³] at the interior cells (for AN2D, ρ_R dK).
"""
kinetic_energy_tendency!(dρK, model, ρ, K, dK, dρ) = cellwise!(dρK, model, (ρ, K, dK, dρ) -> ρ * dK + K * dρ, ρ, K, dK, dρ)

"""
    dρP = potential_energy_tendency!(dρP, model::AN2D, conjvar, chempot, dρconsvar, dρq)
    dρP = potential_energy_tendency!(dρP, model::FC2D, Σ_B, conjvar, chempot, dρ, dρconsvar, dρq)

Tendency of ρ P [W m⁻³] at the interior cells from those of the densities, with d(ρ P) = Σ_B dρ + Σ_ϑ d(ρϑ) + Σ_q d(ρq)
(from d(ρ h - p) = h dρ + ρ (v dp + h_ϑ dϑ + h_q dq) - dp); dρ = 0 for AN2D, where ρ_R P depends on ϑ, q only.
"""
potential_energy_tendency!(dρP, model::AN2D, conjvar, chempot, dρconsvar, dρq) =
    cellwise!(dρP, model, (Σ_ϑ, Σ_q, dρϑ, dρq) -> Σ_ϑ * dρϑ + Σ_q * dρq, conjvar, chempot, dρconsvar, dρq)
potential_energy_tendency!(dρP, model::FC2D, Σ_B, conjvar, chempot, dρ, dρconsvar, dρq) =
    cellwise!(dρP, model, (Σ_B, Σ_ϑ, Σ_q, dρ, dρϑ, dρq) -> Σ_B * dρ + Σ_ϑ * dρϑ + Σ_q * dρq, Σ_B, conjvar, chempot, dρ, dρconsvar, dρq)

# Residuals

"""
    res = heating_residual!(res, model, Q, Q_ϑ, Q_q, ρε)

Closure of the energy budget, Q - Q_ϑ - Q_q - ρ ε_K [W m⁻³] at the interior cells.
"""
heating_residual!(res, model, Q, Q_ϑ, Q_q, ρε) = cellwise!(res, model, (Q, Q_ϑ, Q_q, ρε) -> Q - Q_ϑ - Q_q - ρε, Q, Q_ϑ, Q_q, ρε)

"""
    res = kinetic_budget_residual!(res, model, dρK, ∇J_rev, ∇J_irr, C, ρε, VF)

∂ₜ(ρK) + ∇·(J_K^rev + J_K^irr) - C + ρ ε_K - VF [W m⁻³] at the interior cells: not zero at a cell, but its sum over the
box is (see `integrated_budget_residual`).
"""
kinetic_budget_residual!(res, model, dρK, ∇J_rev, ∇J_irr, C, ρε, VF) =
    cellwise!(res, model, (dρK, ∇J_rev, ∇J_irr, C, ρε, VF) -> dρK + ∇J_rev + ∇J_irr - C + ρε - VF, dρK, ∇J_rev, ∇J_irr, C, ρε, VF)

"""
    res = potential_budget_residual!(res, model, dρP, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q)

∂ₜ(ρP) + ∇·(J_P^rev + J_P^irr) + C + Q_ϑ + Q_q - Q [W m⁻³] at the interior cells: zero to round-off, by the product rule
of the operators (for AN2D, up to the divergence of ρ_R u).
"""
potential_budget_residual!(res, model, dρP, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q) =
    cellwise!(res, model, (dρP, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q) -> dρP + ∇J_rev + ∇J_irr + C + Q_ϑ + Q_q - Q, dρP, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q)

# Manipulated conversion, C = -∇·J_C + C_b + C_D, which is exact in the continuous equations only

"""
    reference = require_reference(reference)

The reference state (`AnelasticReference`) needed for the manipulated conversion, or an error if it is `nothing`: FC2D
has none of its own.
"""
require_reference(::Nothing) = error("the manipulated conversion needs a reference state: pass `reference` (e.g. that of the matching AN2D) to `diagnostics`")
require_reference(reference) = reference

"""
    pa = pressure_anomaly!(pa, model, reference, pressure)

p - p_R(z) [Pa] at the interior cells.
"""
function pressure_anomaly!(pa, model, (; p_R), pressure)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            pa[i, j] = pressure[i, j] - p_R[j]
        end
    end
    return fill_halo!(domain, pa, ZeroGradient())
end

"""
    div = pressure_flux_divergence!(div, model, faces, u, w, pa)

∇·J_C [W m⁻³] at the interior cells, with J_C = (p - p_R) u = pa ℑ(u) on the faces (`advective_flux!`).
"""
function pressure_flux_divergence!(div, model, (Fx, Fz), u, w, pa)
    advective_flux!(Fx, model, Val(1), u, pa)
    advective_flux!(Fz, model, Val(2), w, pa)
    return divergence!(div, model, Fx, Fz)
end

"""
    b = reference_buoyancy!(b, model, reference, state, scratch)

g (ρ_R v - 1) [m s⁻²] at the interior cells, with the specific volume v = 1/ρ of the state (FC2D) or v(ϑ, q, p_R) at the
reference pressure (AN2D).
"""
function reference_buoyancy!(b, model::AN2D, (; ρ_R, p_R), state, (; thermo))
    (; mgr, domain, space, fluid) = model
    (; g) = space
    (; consvar, q) = thermo
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            b[i, j] = g * (ρ_R[j] * fluid(:p, :consvar, :q).specific_volume(p_R[j], consvar[i, j], q[i, j]) - 1)
        end
    end
    return fill_halo!(domain, b, ZeroGradient())
end
function reference_buoyancy!(b, model::FC2D, (; ρ_R), (; m), scratch)
    (; mgr, domain, space) = model
    (; g) = space
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            b[i, j] = g * (ρ_R[j] / m[i, j, 1] - 1)
        end
    end
    return fill_halo!(domain, b, ZeroGradient())
end

"""
    Cb = buoyancy_conversion!(Cb, model, (Fx, Fz), mass_flux_z, b)

C_b = ρ b w [W m⁻³] at the interior cells: the mass flux times the buoyancy averaged to the w-faces (`Fz`,
`advective_flux!`), averaged to the cells.
"""
function buoyancy_conversion!(Cb, model, (Fx, Fz), mass_flux_z, b)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    advective_flux!(Fz, model, Val(2), mass_flux_z, b)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            Cb[i, j] = ℑ(Fz, (2,), (Cell(),), i, j)
        end
    end
    return fill_halo!(domain, Cb, ZeroGradient())
end

"""
    C_D = pressure_dilatation!(C_D, model, pressure_anomaly, divergence)

C_D = (p - p_R) ∇·u [W m⁻³] for FC2D; for AN2D it is zero (left as the zero array: the dilatation is part of C_b, see
energy.tex).
"""
pressure_dilatation!(C_D, ::AN2D, pressure_anomaly, divergence) = C_D
pressure_dilatation!(C_D, model::FC2D, pressure_anomaly, divergence) = pointwise!(C_D, model.mgr, *, pressure_anomaly, divergence)

"""
    res = conversion_residual!(res, model, C, ∇J_C, C_b, C_D)

C + ∇·J_C - C_b - C_D [W m⁻³] at the interior cells: the error of the manipulations beyond discrete exactness.
"""
conversion_residual!(res, model, C, ∇J_C, C_b, C_D) = cellwise!(res, model, (C, ∇J_C, C_b, C_D) -> C + ∇J_C - C_b - C_D, C, ∇J_C, C_b, C_D)
