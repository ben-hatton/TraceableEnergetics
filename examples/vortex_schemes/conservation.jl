# Semi-discrete conservation properties of the vortex-force schemes of CFBoxes, on random periodic fields. For each
# scheme, the relative size of (exact conservation gives round-off, ~1e-15):
#
#   C1  the work of the vortex force, Σ U vortex_x + Σ W vortex_z, with the mass fluxes U = ρᴵ u, W = ρᴶ w
#   C2  dZ/dt for a non-divergent mass flux and a fixed density (the AN2D case), with the potential enstrophy
#       Z = ½ Σ ℑXZ(ρ) η² at the vertices, η = ω / ℑXZ(ρ): dZ/dt = Σ η ∂ₜω
#   C3  dZ/dt for a divergent mass flux, with ∂ₜρ = -∇·(ρu) (the FC2D case): dZ/dt = Σ η ∂ₜω - ½ Σ η² ℑXZ(∂ₜρ)
#   C4  momentum consistency of the linearisation about a uniform flow ū (Hollingsworth et al. 1983): for an
#       x-independent perturbation u′(z), w′(z), the advection ω×u + ∇K linearises to ū ∂ₓ(u′, w′) = 0, so the
#       linearised -vortex - ∇K must vanish. A residual is a spurious momentum source in FC2D; in AN2D it depends on z
#       only, hence it is a gradient, removed by the pressure projection.
#
# with ∂ₜu = -vortex_x, ∂ₜw = -vortex_z (∇K and the pressure gradient have no curl) and ∂ₜω = δZ ∂ₜu / Δz - δX ∂ₜw / Δx.
#
#   julia examples/vortex_schemes/conservation.jl
include("../../tools/setup.jl")

using CFBoxes
using MutatingOrNot: void
using Printf, Random

const schemes = (VortexEnergyCons(), VortexEnstrophyCons(), VortexEnergyEnstrophyCons())

Random.seed!(1)
dx, dz = 0.3, 0.2
box = Box2D(Mx = 16, Mz = 12, boundary = (Periodic, Periodic))
model = (mgr = plain(), domain = box, spatial_scheme = Centred2(), inv_dx = 1 / dx, inv_dz = 1 / dz)
(; ℑ, δ) = operators(model.spatial_scheme)
X, Z, x, z = interior_faces(box, 1), interior_faces(box, 2), interior_cells(box, 1), interior_cells(box, 2)

random(location) = (a = rand(size(alloc(Float64, nothing, box, location))...) .- 0.5; periodize!(model.domain, (a,)); a)
positive(location) = (a = 1 .+ rand(size(alloc(Float64, nothing, box, location))...); periodize!(model.domain, (a,)); a)

# vorticity, potential vorticity and the tendency of the vorticity at the vertices
vorticity(u, w) = [inv(dz) * δ(u, (2,), (Face(),), I, J) - inv(dx) * δ(w, (1,), (Face(),), I, J) for I in X, J in Z]
vertex_density(ρ) = [ℑ(ρ, (1, 2), (Face(), Face()), I, J) for I in X, J in Z]
vorticity_tendency(vortex_x, vortex_z) = vorticity(-vortex_x, -vortex_z)

# relative size of a sum of terms: |Σ t| / Σ |t|
relative(terms...) = abs(sum(sum, terms)) / sum(t -> sum(abs, t), terms)

function C1(scheme, (u, w), ρ)
    vortex_x, vortex_z, (ρᴵ, ρᴶ) = vortex_force!((void, void, void), model, scheme, (u, w), ρ)
    return relative(ρᴵ[X, z] .* u[X, z] .* vortex_x[X, z], ρᴶ[x, Z] .* w[x, Z] .* vortex_z[x, Z])
end

function C2(scheme, ρ)
    # non-divergent mass flux from a vertex streamfunction ψ: U = δz ψ / Δz, W = -δx ψ / Δx
    ψ = random((Face(), Face()))
    ρᴵ, ρᴶ = CFBoxes.face_densities!((void, void), model, ρ)
    u, w = alloc(Float64, nothing, box, (Face(), Cell())), alloc(Float64, nothing, box, (Cell(), Face()))
    for I in X, j in z
        u[I, j] = inv(dz) * δ(ψ, (2,), (Cell(),), I, j) / ρᴵ[I, j]
    end
    for i in x, J in Z
        w[i, J] = -inv(dx) * δ(ψ, (1,), (Cell(),), i, J) / ρᴶ[i, J]
    end
    periodize!(model.domain, (u, w))
    vortex_x, vortex_z, _ = vortex_force!((void, void, void), model, scheme, (u, w), ρ)
    η = vorticity(u, w) ./ vertex_density(ρ)
    return relative(η .* vorticity_tendency(vortex_x, vortex_z))
end

function C3(scheme, (u, w), ρ)
    vortex_x, vortex_z, (ρᴵ, ρᴶ) = vortex_force!((void, void, void), model, scheme, (u, w), ρ)
    # ∂ₜρ = -(δx U / Δx + δz W / Δz) at the cells
    dρ = alloc(Float64, nothing, box, (Cell(), Cell()))
    for i in x, j in z
        dρ[i, j] = -inv(dx) * δ((ρᴵ, u), (1,), (Cell(),), i, j) - inv(dz) * δ((ρᴶ, w), (2,), (Cell(),), i, j)
    end
    periodize!(model.domain, (dρ,))
    η = vorticity(u, w) ./ vertex_density(ρ)
    return relative(η .* vorticity_tendency(vortex_x, vortex_z), -0.5 * η .^ 2 .* vertex_density(dρ))
end

# -vortex - ∇K on the u- and w-faces, with K = ½ (ℑx u² + ℑz w²) at the cells
function advection((u, w), ρ, scheme)
    vortex_x, vortex_z, _ = vortex_force!((void, void, void), model, scheme, (u, w), ρ)
    K = alloc(Float64, nothing, box, (Cell(), Cell()))
    for i in x, j in z
        K[i, j] = 0.5 * (ℑ((u, u), (1,), (Cell(),), i, j) + ℑ((w, w), (2,), (Cell(),), i, j))
    end
    periodize!(model.domain, (K,))
    adv_x = [-vortex_x[I, j] - inv(dx) * δ(K, (1,), (Face(),), I, j) for I in X, j in z]
    adv_z = [-vortex_z[i, J] - inv(dz) * δ(K, (2,), (Face(),), i, J) for i in x, J in Z]
    return adv_x, adv_z
end

function C4(scheme, ρ; ū = 1.0, ε = 1e-3)
    # x-independent perturbation (u′(z), w′(z)) about (ū, 0); the advection is quadratic, so the centred difference is
    # its exact linearisation
    u′, w′ = alloc(Float64, nothing, box, (Face(), Cell())), alloc(Float64, nothing, box, (Cell(), Face()))
    u′ .= rand(1, size(u′, 2)) .- 0.5
    w′ .= rand(1, size(w′, 2)) .- 0.5
    periodize!(model.domain, (u′, w′))
    plus, minus = advection((ū .+ ε .* u′, ε .* w′), ρ, scheme), advection((ū .- ε .* u′, -ε .* w′), ρ, scheme)
    residual = map((p, m) -> (p - m) / 2ε, plus, minus)
    # relative to the size of the terms that should cancel, ū ω′
    scale = ū * maximum(abs, vorticity(u′, w′))
    return maximum(r -> maximum(abs, r), residual) / scale
end

ρ = positive((Cell(), Cell()))
uniform = fill(1.0, size(ρ))
u, w = random((Face(), Cell())), random((Cell(), Face()))

println("relative residuals (round-off ≈ 1e-15 means conserved)\n")
@printf("%-26s %10s %10s %10s %10s\n", "scheme", "C1 energy", "C2 Z, AN", "C3 Z, FC", "C4 Holl.")
for scheme in schemes
    @printf("%-26s %10.1e %10.1e %10.1e %10.1e\n", nameof(typeof(scheme)), C1(scheme, (u, w), ρ), C2(scheme, ρ),
            C3(scheme, (u, w), ρ), C4(scheme, uniform))
end
