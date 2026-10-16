# Boundary conditions of AN2D: the halo of the state and of its tendencies

"""
    state = state_bc!(state, model::AN2D)

Fill the halo of `state = (; m, u, w)`, never its interior: the ghost velocities realize the wall conditions (see
`fill_velocity_halo!`), and m gets a zero-gradient halo, so that the equation of state only sees valid states. The
tracer fluxes through walls are set by `apply_tracer_bcs!`, not through the halo.
"""
function state_bc!((; m, u, w), model::AN2D)
    (; domain, boundary, dx, dz, viscosity) = model
    fill_velocity_halo!(boundary, domain, (u, w), (dx, dz), viscosity.dyn_visc)
    fill_halo!(domain, m, ZeroGradient())
    return (; m, u, w)
end

"""
    dstate = dstate_bc!(dstate, model::AN2D)

Set the tendencies `dstate = (; m, u, w)` to zero in the halo, and the tendency of the normal velocity on the walls.
"""
function dstate_bc!((; m, u, w), model::AN2D)
    (; domain) = model
    zero_halo!(domain, m, (Cell(), Cell()))
    zero_halo!(domain, u, (Face(), Cell()))
    zero_halo!(domain, w, (Cell(), Face()))
    return (; m, u, w)
end
