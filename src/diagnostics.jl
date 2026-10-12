## DIAGNOSTICS OF EITHER MODEL, AND THEIR VISUALISATION
#
# The diagnostics are full arrays with halo, for post-processing; for plotting, `interior_diagnostic` strips the halo
# and `diagnostic_coordinates` gives the coordinates of the remaining points, from the grid location of the diagnostic.

"""
    book = diagnostics(model)
    info = diagnostic_info(model)

`CookBook` of the diagnostics of an AN2D or FC2D model, opened with `open(book; state, dstate, scratch, model)`, and
the long name, units and grid location (`:xz`, `:Xz`, `:xZ`, `:XZ`) of each diagnostic. See `CFAnelastics.diagnostics`
and `CFCompressible.diagnostics`: build the book once, as each session of it overwrites the same output arrays.
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

"""
    video = DiagnosticVideo(model, state, name; label = "AN2D" or "FC2D")

Frames of the diagnostic `name` of `model` on the interior of the box, starting from `state`: one frame per call of
`record_frame!`, which is a `plot_func` for `loop`. Save one or more videos of the same diagnostic as heatmaps side
by side, with a shared colorbar of its long name and units, with
`save(file, videos...; colormap = :thermal, colorrange, framerate = 20)`; the colour range defaults to that of all
frames. Needs a Makie backend (`using CairoMakie` or `using GLMakie`).

# Example
```julia
using CairoMakie
an_video = DiagnosticVideo(an, an_state, :potential_temperature)
CFAnelastics.loop(an, an_state, scheme, params; plot_func = record_frame!, plot_obj = an_video)
fc_video = ...
save("video.mp4", an_video, fc_video)
```
"""
struct DiagnosticVideo{F, Coordinates}
    name::Symbol
    label::String
    long_name::String
    units::String
    coordinates::Coordinates    # (x, z) of the interior points of the diagnostic [m]
    frames::Vector{Matrix{F}}   # the diagnostic on the interior, one matrix per frame
    times::Vector{F}            # time of each frame [s]
end

"""
    video = record_frame!(model, state, dstate, scratch, book, video::DiagnosticVideo, t_iter, params)

Add the frame of the diagnostic at time slice `t_iter` (time `t_iter * params.slice_size`) to `video`: the
`plot_func` of `loop`, with `plot_obj = video`. Needs a Makie backend.
"""
function record_frame! end
