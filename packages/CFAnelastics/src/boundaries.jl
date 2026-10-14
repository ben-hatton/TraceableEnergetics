## BOUNDARY CONDITIONS

# State halo (never the interior): ghost velocities realise the wall conditions; m has a zero-gradient halo, so that
# the equation of state only sees valid states. Tracer fluxes through walls come from `apply_tracer_bcs!`.
function state_bc!(model::AN2D, (; m, u, w))
    (; domain, boundary, dx, dz) = model
    fill_velocity_halo!(boundary, domain, (u, w), (dx, dz), model.viscosity.dyn_visc)
    fill_halo!(m, domain, ZeroGradient())
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
