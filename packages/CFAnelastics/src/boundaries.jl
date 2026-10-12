## BOUNDARY CONDITIONS

# Fill the halo of the state (never its interior) from the boundary conditions: velocity ghost values realize the
# wall (or periodic) conditions; the halo of m mirrors the interior, so that the equation of state is only ever
# evaluated on valid states. Tracer fluxes through walls do not use the halo of m (see `wall_fluxes!`).
function state_bc!(model::AN2D, (; m, u, w))
    (; domain, boundary, dx, dz) = model
    fill_velocity_halo!(boundary, domain, (u, w), (dx, dz), model.viscosity.dyn_visc)
    mirror_halo!(domain, m)
    return (; m, u, w)
end

# tendencies vanish in the halo, and for the normal velocity on walls
function dstate_bc!(model::AN2D, (; m, u, w))
    (; domain) = model
    zero_halo!(domain, m, (Centred(), Centred()))
    zero_halo!(domain, u, (Faced(), Centred()))
    zero_halo!(domain, w, (Centred(), Faced()))
    return (; m, u, w)
end
