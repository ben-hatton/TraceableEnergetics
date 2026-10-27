## DIAGNOSTICS OF EITHER MODEL
#
# The diagnostics of AN2D and FC2D are those of the model packages, the energy budget of energy.tex (energetics/) and
# the Lorenz reference state of available potential energy with the available energy of its parcels, the budgets of the
# available energy and the tendency of the reference state (reference_state/). They are full arrays with halo, for
# post-processing; for plotting, `interior_diagnostic` strips the halo and `diagnostic_coordinates` gives the coordinates
# of the remaining points, from the grid location of the diagnostic.

"""
    book = diagnostics(model; reference = default, reference_state = nothing)
    info = diagnostic_info(model; reference_state = nothing)

`CookBook` of the diagnostics of an AN2D or FC2D model, opened with `open(book; state, dstate, scratch, model)`, and
the long name, units and grid location of each diagnostic: one `Face()`/`Cell()` per dimension, e.g. `(Cell(), Cell())`
cells, `(Face(), Cell())` u-faces, `()` for a number. They are those of `CFAnelastics.diagnostics` and
`CFCompressible.diagnostics`, and the energy budget of energy.tex, listed in `energetics_entries`. K and P are specific
energies [J kg⁻¹]; ρK, ρP and every term of the budget are per unit volume [W m⁻³] (ρ = ρ_R for AN2D); the
`integrated_` diagnostics are their integrals over the interior cells, per unit length along y.

The geopotential is g (z - z0), with z0 the origin `model.space.z0` [m] of the tank. The manipulated conversion needs the
reference state (`AnelasticReference` with the density ρ_R(z) and the hydrostatic pressure p_R(z)): the model's own for
AN2D, the keyword `reference` (usually that of the matching AN2D, on the same grid) for FC2D, which has none.

With a `reference_state` (an `AnelasticLorenzReference` or a `CompressibleLorenzReference` of the model), the book
also computes it for the state of each session, once, and has its diagnostics and those of the available energy of its
parcels (`reference_state_entries`): on the cells, profiles on the z axis (location `(nothing, Cell())`), and
numbers.

Build the book once, as each session of it overwrites the same output arrays.
"""
function diagnostics(model::AN2D{F}; reference = model.reference, reference_state = nothing) where F
    book = CFAnelastics.diagnostics(model)
    add_energetics!(book, F, model, reference)
    return add_reference_state!(book, F, model, reference_state)
end
function diagnostics(model::FC2D{F}; reference = nothing, reference_state = nothing) where F
    book = CFCompressible.diagnostics(model)
    add_energetics!(book, F, model, reference)
    return add_reference_state!(book, F, model, reference_state)
end
diagnostic_info(model::AN2D; reference_state = nothing) =
    merge(CFAnelastics.diagnostic_info(model), map(first, energetics_entries(model, nothing, nothing, nothing)), reference_state_info(model, reference_state))
diagnostic_info(model::FC2D; reference_state = nothing) =
    merge(CFCompressible.diagnostic_info(model), map(first, energetics_entries(model, nothing, nothing, nothing)), reference_state_info(model, reference_state))
reference_state_info(model, ::Nothing) = (;)
reference_state_info(model, ref) =
    map(first, merge(reference_state_entries(model, ref, nothing), available_energy_budget_entries(model, nothing, nothing),
                     reference_tendency_entries(model, nothing)))

"""
    entry = entry(long_name, units, location, recipe)

A diagnostic: its `info = (; long_name, units, location)` and its `recipe` for a `CookBook`.
"""
entry(long_name, units, location, recipe) = ((; long_name, units, location), recipe)

"""
    book = add_energetics!(book, F, model, reference)

Adds the recipes of `energetics_entries` to `book`. The output array of each diagnostic is allocated on its first use,
on its grid location, in the float type `F`; `faces` are the scratch arrays on the x and z faces shared by the terms
built from face products.
"""
function add_energetics!(book, F, model, reference)
    (; mgr, domain) = model
    entries = energetics_entries(model, nothing, nothing, nothing)
    cache = Dict{Symbol, Any}()
    out(name) = get!(() -> alloc(F, mgr, domain, first(entries[name]).location), cache, name)::Matrix{F}
    faces = (alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face())))
    merge!(book.recipes, pairs(map(last, energetics_entries(model, out, faces, reference))))
    return book
end

"""
    book = add_reference_state!(book, F, model, ref)

Adds to `book` the recipe `reference_state`, which computes the reference state `ref` for the state of the session, the
recipe `reference_state_tendency`, which computes its tendency from the diabatic rates (`reference_tendency!`), and
the recipes of `reference_state_entries`, `available_energy_budget_entries` and `reference_tendency_entries`; nothing if
`ref` is `nothing`. The output array of each diagnostic on the cells is allocated on its first use, in the float type
`F`, as for the energetics.
"""
add_reference_state!(book, F, model, ::Nothing) = book
function add_reference_state!(book, F, model, ref)
    (; mgr, domain) = model
    entries = merge(reference_state_entries(model, ref, nothing), available_energy_budget_entries(model, nothing, nothing),
                    reference_tendency_entries(model, nothing))
    cache = Dict{Symbol, Any}()
    out(name) = get!(() -> alloc(F, mgr, domain, first(entries[name]).location), cache, name)::Matrix{F}
    # scratch arrays on the x and z faces of the terms built from face products, apart from those of the energetics
    faces = (alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face())))
    book.recipes[:reference_state] = (model, state) -> reference_state!(ref, model, state)
    # the tendency of the reference state, from the N_*⁻²-weighted means of v̇ over the slots, after the reference state
    book.recipes[:reference_state_tendency] = (model, reference_state, consvar_diabatic_rate, composition_diabatic_rate) ->
        reference_tendency!(reference_state, model, consvar_diabatic_rate, composition_diabatic_rate)
    merge!(book.recipes, pairs(map(last, reference_state_entries(model, ref, out))))
    merge!(book.recipes, pairs(map(last, available_energy_budget_entries(model, out, faces))))
    merge!(book.recipes, pairs(map(last, reference_tendency_entries(model, out))))
    return book
end

"""
    field = interior_diagnostic(session, name)
    x, z = diagnostic_coordinates(model, name)

The diagnostic `name` of an open diagnostics session without its halo (a view), for plotting, and the coordinates
[m] of its points, from the grid location of the diagnostic (e.g. both wall faces for `w` in a bounded box).
"""
interior_diagnostic(session, name) = interior(session.model.domain, CookBooks.get(session, name))
diagnostic_coordinates(model, name) =
    coordinates(model.domain, (model.dx, model.dz), diagnostic_info(model)[name].location)
