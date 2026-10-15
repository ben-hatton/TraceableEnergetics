"""
    state = loop(model::AN2D, state0, time_scheme, params; outputs = nothing)
Main time integration loop for the AN2D model; returns the final state.
- `model`: AN2D model instance
- `state0`: initial state (see `initialize`)
- `time_scheme`: time integration scheme
- `params`: (; Nslice, slice_size, dt_max, cfl): `Nslice` time slices of `slice_size` [s], each split into equal
  steps of at most `dt_max` [s] and `cfl` (≤ 1 for stability) times the stability limit of `time_scheme`, see
  `cfl_condition`
- `outputs`: callable `outputs(model, state, dstate, scratch, book, t)`, called at t = 0 and at the end of each
  time slice, with the diagnostics `book` and the time `t` [s] (e.g. to record diagnostics), then closed with
  `close(outputs)` when the run ends, even on error; `nothing` records nothing
"""
function loop(model::AN2D{F}, state0, time_scheme, params; outputs = nothing) where {F}
    (; Nslice, slice_size, dt_max, cfl) = params

    # copy the initial state
    state = deepcopy(state0)

    # initialise time
    t = zero(F)

    # create dstate and scratch space
    dstate, scratch = CFTimeSchemes.tendencies!(void, void, model::AN2D, state, t)

    # diagnostics, built once: the book owns the output arrays of its computed diagnostics
    book = diagnostics(model)

    # the outputs are closed when the run ends, even on error
    try
        # record the initial state
        record_outputs(outputs, model, state, dstate, scratch, book, t)

        # scratch space of the time scheme
        solver_scratch = CFTimeSchemes.scratch_space(time_scheme, state0, t)

        # run loop
        for t_iter = 1:Nslice
            # time step from the CFL condition, and a solver with that time step
            dt, Nperslice = cfl_condition(model, time_scheme, state, cfl, slice_size, dt_max)
            solver = CFTimeSchemes.IVPSolver(dt, time_scheme, solver_scratch)

            # advance state
            state, t = CFTimeSchemes.advance!(state, solver, state, t, Nperslice)

            # record the state at the end of the slice
            record_outputs(outputs, model, state, dstate, scratch, book, t)

            # advance progress meter
            progress(t_iter, slice_size, Nslice * slice_size)
        end
    finally
        close_outputs(outputs)
    end
    return state
end

# call the outputs, if any, with the tendencies and scratch at time t, and close them
function record_outputs(outputs, model, state, dstate, scratch, book, t)
    dstate, scratch = CFTimeSchemes.tendencies!(dstate, scratch, model, state, t)
    return outputs(model, state, dstate, scratch, book, t)
end
record_outputs(::Nothing, args...) = nothing
close_outputs(outputs) = close(outputs)
close_outputs(::Nothing) = nothing

"""
    dt, Nperslice = cfl_condition(model::AN2D, scheme, state, cfl, slice_size, dt_max)

Return the time step `dt` [s] and the number `Nperslice` of steps per time slice: the largest `dt` that fits a whole
number of times in `slice_size` [s] and is at most `dt_max` [s] and `cfl` times the stability limit of the explicit
`scheme` (`CFTimeSchemes.max_time_step`) for the fastest discrete frequency `max_frequency(model, state)`. `cfl` is a
fraction of that limit: `cfl ≤ 1` is stable for any explicit scheme.
"""
function cfl_condition(model::AN2D{F}, scheme, state, cfl, slice_size, dt_max) where {F}
    dt        = min(cfl * CFTimeSchemes.max_time_step(scheme, inv(max_frequency(model, state))), slice_size, dt_max)
    Nperslice = ceil(Int, slice_size / dt)
    return slice_size / Nperslice, Nperslice
end

"""
    ω = max_frequency(model::AN2D, state)

Return an upper bound `ω` [s⁻¹] on the fastest discrete frequency of the state over the interior, the sum of:
- advection, centred on the C-grid: max|u|/dx + max|w|/dz;
- diffusion: 4 max(ν, k_T, k_q) (1/dx² + 1/dz²), with ν = (μ + μ_b) / min(ρ_R);
- internal gravity waves (ω = N |kₓ|/|k| ≤ N): max(N), see `max_buoyancy_frequency`.
"""
function max_frequency(model::AN2D{F}, (; m, u, w)) where {F}
    (; domain, dx, dz, viscosity, thermoflux, reference) = model
    irange, jrange = interior_cells(domain, 1), interior_cells(domain, 2)
    inv_d2         = inv(dx^2) + inv(dz^2)
    advection      = maximum(abs, view(u, irange, jrange)) / dx + maximum(abs, view(w, irange, jrange)) / dz
    ν              = (viscosity.dyn_visc + viscosity.bulk_visc) / minimum(view(reference.ρ_R, jrange))
    diffusion      = 4 * max(ν, thermoflux.k_T, thermoflux.k_q) * inv_d2
    return advection + diffusion + max_buoyancy_frequency(model, m) + floatmin(F)
end

# max(N) over the w-faces J between interior cells J-1 and J, N² = g (α ∂consvar/∂z - β ∂q/∂z), with
# α = ∂v/∂consvar / v and β = -∂v/∂q / v at the face (mean reference pressure, consvar and q of the two cells);
# N² < 0 (static instability) does not oscillate and gives 0
function max_buoyancy_frequency((; fluid, domain, space, dz, reference)::AN2D{F}, m) where {F}
    (; ρ_R, p_R) = reference
    volume_functions = fluid(:p, :consvar, :q).volume_functions
    irange, jrange = interior_cells(domain, 1), interior_cells(domain, 2)
    N2max = zero(F)
    for J in first(jrange)+1:last(jrange), i in irange
        v_1, v_2       = inv(ρ_R[J-1]), inv(ρ_R[J])
        cons_1, cons_2 = m[i, J-1, 1] * v_1, m[i, J, 1] * v_2
        q_1, q_2       = m[i, J-1, 2] * v_1, m[i, J, 2] * v_2
        v, _, ∂v_∂cons, ∂v_∂q = volume_functions((p_R[J-1] + p_R[J]) / 2, (cons_1 + cons_2) / 2, (q_1 + q_2) / 2)
        N2max = max(N2max, space.g * (∂v_∂cons * (cons_2 - cons_1) + ∂v_∂q * (q_2 - q_1)) / (v * dz))
    end
    return sqrt(N2max)
end

"""
    progress(t_iter, slice_size, total_time)

Display progress of the simulation in the terminal.
"""
function progress(t_iter, slice_size, total_time)
    seconds = round(t_iter*slice_size; digits=2)
    print("\rProgress: $seconds / $total_time s", " " ^ 10)
    flush(stdout)       
end
