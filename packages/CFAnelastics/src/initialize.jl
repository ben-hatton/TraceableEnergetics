## INITIAL STATE

"""
    state = initialize(model::AN2D; consvar, q, velocity = nothing)

State (m, u, w) of the AN2D model from the tracers consvar(x, z) and q(x, z), functions of position with z = 0 at the
bottom and x = 0 at the left, and the initial velocity `velocity = (u, w)`, face arrays shaped like u and w (e.g.
from `divergence_free_velocity`, plus a mean flow); `nothing` is the fluid at rest. The tracers are weighted by the
reference density, m = ρ_R (consvar, q); the halo is set by the boundary conditions.
"""
function initialize(model::AN2D{F}; consvar, q, velocity = nothing) where F
    (; mgr, domain, dx, dz, reference) = model
    (; ρ_R) = reference
    x(i) = cellpoint(domain, dx, 1, i); z(j) = cellpoint(domain, dz, 2, j)
    m = alloc(F, mgr, domain, (Cell(), Cell()), 2)
    for i in axes(m, 1), j in axes(m, 2)
        m[i, j, 1] = ρ_R[j] * consvar(x(i), z(j))
        m[i, j, 2] = ρ_R[j] * q(x(i), z(j))
    end
    u, w = initial_velocity(velocity, F, mgr, domain)
    return state_bc!(model, (; m, u, w))
end

# initial velocity: a copy of the given face arrays, or zero (at rest), in arrays placed by the manager (see `alloc`)
initial_velocity((u, w)::Tuple, F, mgr, domain) =
    (copyto!(alloc(F, mgr, domain, (Face(), Cell())), u), copyto!(alloc(F, mgr, domain, (Cell(), Face())), w))
initial_velocity(::Nothing, F, mgr, domain) = (alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face())))

"""
    (u, w) = divergence_free_velocity(model::AN2D, ψ)

Velocity on the faces with discretely divergence-free mass flux, ∇·(ρ_R u) = 0 in every cell, from a mass
streamfunction ψ(X, Z) [kg m⁻¹ s⁻¹] evaluated at the cell vertices: ρ_R u = δ_z ψ / Δz on u-faces and
ρ_R w = -δ_x ψ / Δx on w-faces (ρ_R averaged from the two cells). The velocity through a wall vanishes where ψ is
constant along it. Faces without two neighbouring cells (halo edges) are zero.
"""
function divergence_free_velocity(model::AN2D{F}, ψ) where F
    (; mgr, domain, dx, dz, reference) = model
    (; ρ_R) = reference
    X(I) = facepoint(domain, dx, 1, I); Z(J) = facepoint(domain, dz, 2, J)
    u, w = alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face()))
    for I in axes(u, 1), j in axes(u, 2)
        u[I, j] = (ψ(X(I), Z(j+1)) - ψ(X(I), Z(j))) / (dz * ρ_R[j])
    end
    for i in axes(w, 1), J in faces(domain, 2)
        w[i, J] = -(ψ(X(i+1), Z(J)) - ψ(X(i), Z(J))) / (dx * ℑ(ρ_R, (2,), (Face(),), J))
    end
    return u, w
end
