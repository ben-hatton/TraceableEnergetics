## DIAGNOSTICS OF EITHER MODEL
#
# The diagnostics are full arrays with halo, for post-processing; for plotting, `interior_diagnostic` strips the halo
# and `diagnostic_coordinates` gives the coordinates of the remaining points, from the grid location of the diagnostic.

"""
    book = diagnostics(model)
    info = diagnostic_info(model)

`CookBook` of the diagnostics of an AN2D or FC2D model, opened with `open(book; state, dstate, scratch, model)`, and
the long name, units and grid location of each diagnostic: one `Face()`/`Cell()` per dimension, e.g. `(Cell(), Cell())`
cells, `(Face(), Cell())` u-faces. See `CFAnelastics.diagnostics` and `CFCompressible.diagnostics`: build the book
once, as each session of it overwrites the same output arrays.
"""
diagnostics(model::AN2D) = CFAnelastics.diagnostics(model)
diagnostics(model::FC2D) = CFCompressible.diagnostics(model)
diagnostic_info(model::AN2D) = CFAnelastics.diagnostic_info(model)
diagnostic_info(model::FC2D) = CFCompressible.diagnostic_info(model)

"""
    field = interior_diagnostic(session, name)
    x, z = diagnostic_coordinates(model, name)

The diagnostic `name` of an open diagnostics session without its halo (a view), for plotting, and the coordinates
[m] of its points, from the grid location of the diagnostic (e.g. both wall faces for `w` in a bounded box).
"""
interior_diagnostic(session, name) = interior(session.model.domain, CookBooks.get(session, name))
diagnostic_coordinates(model, name) =
    coordinates(model.domain, (model.dx, model.dz), diagnostic_info(model)[name].location)
