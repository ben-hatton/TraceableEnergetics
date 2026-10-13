"""
    state = loop(model::AN2D, state0, time_scheme, params; outputs = nothing)
Main time integration loop for the AN2D model; returns the final state.
- `model`: AN2D model instance
- `state0`: initial state (see `initialize`)
- `time_scheme`: time integration scheme
- `params`: (; Nslice, slice_size, dt_max, cfl)
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

        # initialise solver
        solver_scratch = CFTimeSchemes.scratch_space(time_scheme, state0, t)
        solver = MutIVPSolver(dt_max, time_scheme, solver_scratch)

        # run loop
        for t_iter = 1:Nslice
            # determine dt from CFL condition
            dt, Nperslice   = cfl_condition(model, state, cfl, slice_size, dt_max)
        
            # update solver with new dt
            solver.dt = dt

            # advance state
            state, t = advance!(state, solver, state, t, Nperslice)

            # compute dstate and scratch space at new time
            dstate, scratch = CFTimeSchemes.tendencies!(dstate, scratch, model, state, t)

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

# call and close the outputs, if any
record_outputs(outputs, args...) = outputs(args...)
record_outputs(::Nothing, args...) = nothing
close_outputs(outputs) = close(outputs)
close_outputs(::Nothing) = nothing

"""
    cfl_condition(model::AN2D, state, cfl, slice_size, dt_max)

Compute the time step based on the CFL condition.
"""
function cfl_condition((; domain, dx, dz)::AN2D{F}, (; u, w), cfl, slice_size, dt_max) where {F}
    max_u       = maximum(abs, u[xrange_interior(domain), zrange_interior(domain)]) + 1e-99
    max_w       = maximum(abs, w[xrange_interior(domain), zrange_interior(domain)]) + 1e-99
    dt_cfl      = min(cfl * dx / max_u, cfl * dz / max_w, slice_size, dt_max)
    Nperslice   = ceil(Int, slice_size / dt_cfl)
    dt_cfl      = slice_size / Nperslice
    return dt_cfl, Nperslice
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


"""
    MutIVPSolver{F, Scheme, Scratch}
Mutable struct to hold the time step, time integration scheme, and scratch space.
Similar to CFTimeSchemes.IVPSolver but mutable to allow changing dt.
- `F`: floating point type
- `Scheme`: time integration scheme type
- `Scratch`: type of scratch space
"""
mutable struct MutIVPSolver{F, Scheme, Scratch}
    dt::F               # time step
    scheme::Scheme
    scratch::Scratch    # scratch space, or void
end

"""
    advance!(storage::Union{Void, State}, (; dt, scheme, scratch)::MutIVPSolver, state::State, t, N::Int) where State
Advance the state by N time steps using the specified time integration scheme. Extension of CFTimeSchemes.advance! to handle mutable solver.
- `storage`: optional storage for intermediate states
- `dt`: time step size
- `scheme`: time integration scheme
- `scratch`: scratch space for computations
- `state`: current state of the model
- `t`: current time
- `N`: number of time steps to advance
"""
function advance!(storage::Union{Void, State}, (; dt, scheme, scratch)::MutIVPSolver, state::State, t, N::Int) where State
    @assert N>0
    @assert typeof(t)==typeof(dt)
    state = CFTimeSchemes.advance!(storage, scheme, state, t, dt, scratch)::State
    for i=2:N
        state = CFTimeSchemes.advance!(storage, scheme, state, t+(i-1)*dt, dt, scratch)::State
    end
    return state, t+N*dt
end