## BOUNDARY CONDITIONS

# State halo (never the interior): ghost velocities realise the wall conditions; m has a zero-gradient halo, so that
# the equation of state only sees valid states. Tracer fluxes through walls come from `apply_tracer_bcs!`.
function state_bc!(model::AN2D, (; m, u, w))
    (; domain, boundary, dx, dz, viscosity) = model
    fill_velocity_halo!(boundary, domain, (u, w), (dx, dz), viscosity.dyn_visc)
    fill_halo!(m, domain, ZeroGradient())
    return (; m, u, w)
end

# tendencies vanish in the halo, and for the normal velocity on walls
function dstate_bc!(model::AN2D, (; m, u, w))
    (; domain) = model
    zero_halo!(domain, m, (Cell(), Cell()))
    zero_halo!(domain, u, (Face(), Cell()))
    zero_halo!(domain, w, (Cell(), Face()))
    return (; m, u, w)
end
