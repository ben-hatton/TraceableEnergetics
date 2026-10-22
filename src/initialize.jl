## INITIAL STATES OF THE MODELS FROM THE FIELDS OF A SETUP

"""
    state = initialize(model::AN2D, (; consvar, q, u, ψ))
    state = initialize(model::FC2D, (; consvar, q, u, ψ); pref)

State of `model` from the fields of a setup (tracers consvar(x, z) and q(x, z), mean flow u(z), mass streamfunction
ψ(X, Z)), in three steps: the rest profiles (AN2D: the reference density of the model; FC2D: the balanced density of
each column, with pressure `pref` at the height z0 of the geopotential origin), the discretely divergence-free perturbation from ψ for them, then
the state, with the velocity the mean flow plus the perturbation.
"""
function initialize(model::AN2D, (; consvar, q, u, ψ)::NamedTuple)
    velocity = mean_flow(model, u) .+ CFAnelastics.divergence_free_velocity(model, ψ)
    return CFAnelastics.initialize(model; consvar, q, velocity)
end
function initialize(model::FC2D, (; consvar, q, u, ψ)::NamedTuple; pref)
    density = CFCompressible.balanced_density(model; consvar, q, pref)
    velocity = mean_flow(model, u) .+ CFCompressible.divergence_free_velocity(model, density, ψ)
    return CFCompressible.initialize(model; density, consvar, q, velocity)
end

# face arrays (u, w) of a mean flow u(z): u(z) on the u-faces, zero on the w-faces
function mean_flow((; domain, dz), u)
    ū = [u(cellpoint(domain, dz, 2, j)) for I in 1:nfaces(domain, 1), j in 1:ncells(domain, 2)]
    return ū, zeros(typeof(dz), ncells(domain, 1), nfaces(domain, 2))
end
