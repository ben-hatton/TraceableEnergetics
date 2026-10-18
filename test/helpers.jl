# Helpers of the canonical-setup and diagnostics tests (the part of the model suites' helpers that they use)

#=
Fresh scratch arrays come from `CFBoxes.alloc!(void, …)`. In the tests they are filled with `ALLOC_FILL[]`, so that
we can check that results never depend on the contents of new allocations. This method is deliberate type piracy,
confined to the tests.
=#
const ALLOC_FILL = Ref(0.0)
CFBoxes.alloc!(::MutatingOrNot.Void, F::Type, mgr, box::CFBoxes.AbstractBox, location::Tuple, dims...) =
    fill!(CFBoxes.alloc(F, mgr, box, location, dims...), ALLOC_FILL[])

# run f() with new allocations filled with `value`
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

# interior index ranges: cells, x-faces (u) and z-faces (w), including walls in bounded directions
interior_xz(d) = (interior_cells(d, 1), interior_cells(d, 2))
interior_Xz(d) = (interior_faces(d, 1), interior_cells(d, 2))
interior_xZ(d) = (interior_cells(d, 1), interior_faces(d, 2))

# x-index of the face east of cell i, wrapping periodically (never reads the halo)
east(d, i) = boundary_topology(d)[1] isa Periodic && i == last(interior_cells(d, 1)) ? first(interior_cells(d, 1)) : i + 1

# tendencies from freshly allocated, zero-filled scratch: only the memory-safety test sees other contents
evaluate(tend!, model, state) = with_fill(() -> tend!(void, void, model, copy_state(state)), 0.0)
