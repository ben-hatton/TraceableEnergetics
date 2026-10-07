#####
##### Generic property checks shared by the anelastic and compressible tests
#####

# Fresh scratch arrays come from `MutatingOrNot.similar!(void, …)`. In the tests they are filled with
# `ALLOC_FILL[]`, so that we can check that results never depend on the contents of new allocations.
const ALLOC_FILL = Ref(0.0)
MutatingOrNot.similar!(::MutatingOrNot.Void, y...) = fill!(similar(y...), ALLOC_FILL[])

function with_fill(f, value)
    old = ALLOC_FILL[]
    ALLOC_FILL[] = value
    try
        return f()
    finally
        ALLOC_FILL[] = old
    end
end

# apply f to every numeric array in a (nested) tuple / named tuple
foreach_array(f, x::AbstractArray{<:Number}) = f(x)
foreach_array(f, x::Union{Tuple, NamedTuple}) = foreach(y -> foreach_array(f, y), x)
foreach_array(f, x) = nothing

all_finite(x) = (ok = Ref(true); foreach_array(a -> (ok[] &= all(isfinite, a)), x); ok[])

copy_state(s) = map(copy, s)

# ‖a - b‖∞ / ‖b‖∞
relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), floatmin())

# interior index ranges: cells, x-faces (u), z-faces incl. walls (w)
cells(d)  = (xrange_interior(d), zrange_interior(d))
ufaces(d) = (xrange_interior(d), zrange_interior(d))
wfaces(d) = (xrange_interior(d), first(zrange_interior(d)):last(zrange_interior(d))+1)

# interior x-index of the cell/face east of interior index i, wrapping periodically (never reads the halo)
east(d, i) = i == last(xrange_interior(d)) ? first(xrange_interior(d)) : i + 1

# periodic x-halo fill for any array whose first dimension is x (cells, u- or w-points)
function fill_xhalo!(a, d)
    Hx, _ = halo_size(d)
    Mx, _ = dims(d)
    for h in 1:Hx
        selectdim(a, 1, h)            .= selectdim(a, 1, h + Mx)
        selectdim(a, 1, Mx + Hx + h)  .= selectdim(a, 1, Hx + h)
    end
    return a
end

# b[i] = a[perm(i)] over interior x-indices, then refill the periodic halo; `sign` flips e.g. u under mirroring
function xpermute(a, d, perm; sign = 1)
    Hx, _ = halo_size(d)
    Mx, _ = dims(d)
    b = copy(a)
    for i in 1:Mx
        selectdim(b, 1, i + Hx) .= sign .* selectdim(a, 1, perm(i) + Hx)
    end
    return fill_xhalo!(b, d)
end

# x-shift by k cells, and x-mirror. Cells and w-points: cell i ↦ Mx+1-i.
# u-points: face I is the west face of cell I, so it ↦ west face of cell Mx+2-I (mod Mx), and u ↦ -u.
function shift_state(s, d, k)
    Mx, _ = dims(d)
    p(i) = mod1(i - k, Mx)
    return (m = xpermute(s.m, d, p), u = xpermute(s.u, d, p), w = xpermute(s.w, d, p))
end
function mirror_state(s, d)
    Mx, _ = dims(d)
    pc(i) = Mx + 1 - i
    pf(I) = mod1(Mx + 2 - I, Mx)
    return (m = xpermute(s.m, d, pc), u = xpermute(s.u, d, pf; sign = -1), w = xpermute(s.w, d, pc))
end

# compare two tendencies on interior points only
function tendency_mismatch(a, b, d)
    (ic, jc), (iu, ju), (iw, jw) = cells(d), ufaces(d), wfaces(d)
    return max(relerr(a.m[ic, jc, :], b.m[ic, jc, :]),
               relerr(a.u[iu, ju], b.u[iu, ju]),
               relerr(a.w[iw, jw], b.w[iw, jw]))
end

# total and Σ|·| of the per-cell energy tendency dE/dt = d/dε E(state + ε dstate) at ε = 0
function energy_tendency(cell_energies, model, state, dstate)
    perturbed(ε) = map((x, dx) -> x .+ ε .* dx, state, dstate)
    dE = ForwardDiff.derivative(ε -> cell_energies(model, perturbed(ε)), 0.0)
    return sum(dE), sum(abs, dE)
end

#####
##### Test sets common to both models; `tend!(model, dstate, scratch, state)` wraps each model's tendencies!
#####

# tendencies from freshly allocated, zero-filled scratch: only the memory-safety test sees other contents
evaluate(tend!, model, state) = with_fill(() -> tend!(model, void, void, copy_state(state)), 0.0)

function test_memory_safety(tend!, model, state)
    d = model.domain
    ref, _ = evaluate(tend!, model, state)
    ref = map(copy, ref)

    # fresh allocations filled with NaN: nothing may read memory it has not written
    s = copy_state(state)
    out, scratch = with_fill(() -> tend!(model, void, void, s), NaN)
    @test all_finite(out)
    @test all(map(isequal, out, ref))

    # re-using dstate and scratch gives the same result
    out2, _ = tend!(model, out, scratch, copy_state(state))
    @test all(map(isequal, out2, ref))

    # the state interior is not modified (bitwise)
    @test s.m[cells(d)..., :] == state.m[cells(d)..., :]
    @test s.u[ufaces(d)...] == state.u[ufaces(d)...]
    @test s.w[wfaces(d)...] == state.w[wfaces(d)...]
end

function test_symmetries(tend!, model, state; tol = 1e-8)   # round-off from cancellation in B; index bugs give O(1e-3–1)
    d = model.domain
    tendency(s) = first(evaluate(tend!, model, s))
    ds = tendency(state)
    @test tendency_mismatch(tendency(shift_state(state, d, 5)), shift_state(ds, d, 5), d) < tol
    @test tendency_mismatch(tendency(mirror_state(state, d)), mirror_state(ds, d), d) < tol
end

function test_walls(tend!, model, state; tol = 1e-12)
    d = model.domain
    dstate, _ = evaluate(tend!, model, state)
    iw, jw = wfaces(d)
    scale = max(maximum(abs, dstate.w[iw, jw]), floatmin())
    @test maximum(abs, dstate.w[iw, first(jw)]) / scale < tol   # bottom wall face
    @test maximum(abs, dstate.w[iw, last(jw)]) / scale < tol    # top wall face
end

# Σ dm over the interior vanishes for conserved densities, and equals Σ σ for the conservative variable
function test_conservation(tend!, model, state, conserved, consvar; tol = 1e-12)
    d = model.domain
    dstate, scratch = evaluate(tend!, model, state)
    for k in conserved
        dm = dstate.m[cells(d)..., k]
        @test abs(sum(dm)) / sum(abs, dm) < tol
    end
    dm = dstate.m[cells(d)..., consvar]
    σ = scratch.irreversible.σ_cons[cells(d)...]
    @test abs(sum(dm) - sum(σ)) / (sum(abs, dm) + sum(abs, σ)) < tol
end

function test_energy(tend!, cell_energies, model, state; tol = 1e-10)
    dstate, _ = evaluate(tend!, model, state)
    total, scale = energy_tendency(cell_energies, model, state, dstate)
    @test abs(total) / scale < tol
    return abs(total) / scale
end

function test_no_allocations(tend!, model, state)
    dstate, scratch = evaluate(tend!, model, state)
    s = copy_state(state)
    tend!(model, dstate, scratch, s)
    t = @elapsed tend!(model, dstate, scratch, s)
    bytes = @allocated tend!(model, dstate, scratch, s)
    @info "$(nameof(typeof(model))) tendencies!" time_ms = 1e3t bytes
    @test bytes == 0
end

# RK4 steps; returns (states before/after) for model-specific checks
function integrate(model, state, dt, nsteps)
    solver = CFTimeSchemes.IVPSolver(CFTimeSchemes.RungeKutta4(model), dt, copy_state(state), 0.0)
    s = copy_state(state)
    for n in 1:nsteps
        s, _ = CFTimeSchemes.advance!(s, solver, s, (n - 1) * dt, 1)
    end
    return s
end

function test_integration(model, state, dt, nsteps, cell_energies, conserved, kinetic; tol_E = 1e-6)
    d = model.domain
    s = integrate(model, state, dt, nsteps)
    @test all_finite(s)
    for k in conserved
        @test abs(sum(s.m[cells(d)..., k]) / sum(state.m[cells(d)..., k]) - 1) < 1e-13
    end
    ΔE = sum(cell_energies(model, s)) - sum(cell_energies(model, state))
    @test abs(ΔE) / kinetic(model, state) < tol_E
    return s
end

# every diagnostics recipe evaluates to a finite array
function test_diagnostics(book, tend!, model, state)
    dstate, scratch = evaluate(tend!, model, state)
    session = open(book; state = copy_state(state), dstate, scratch, model)
    for sym in keys(getfield(book, :recipes))
        @testset "$sym" begin
            @test all(isfinite, CookBooks.get(session, sym))
        end
    end
    close(session)
end
