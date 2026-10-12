module TraceableEnergeticsMakieExt
# Makie videos of the diagnostics of either model: the frames of a diagnostic on the interior of the box, recorded during
# `loop`, saved as heatmaps side by side (one panel per video, e.g. AN2D and FC2D) with a shared colorbar

using Makie: Makie, Figure, Axis, Colorbar, Observable, heatmap!, lift, record
using CFTimeSchemes: CFTimeSchemes
using MutatingOrNot: void
using TraceableEnergetics: TraceableEnergetics, DiagnosticVideo, diagnostics, diagnostic_info, interior_diagnostic,
                           diagnostic_coordinates

function TraceableEnergetics.DiagnosticVideo(model, state, name::Symbol; label = string(nameof(typeof(model))))
    (; long_name, units) = diagnostic_info(model)[name]
    video = DiagnosticVideo(name, label, long_name, units, diagnostic_coordinates(model, name),
                            Matrix{typeof(model.dx)}[], typeof(model.dx)[])
    # first frame: the diagnostic of the initial state, with its own scratch and book
    dstate, scratch = CFTimeSchemes.tendencies!(void, void, model, deepcopy(state), zero(model.dx))
    return push_frame!(video, diagnostics(model), model, state, dstate, scratch, zero(model.dx))
end

TraceableEnergetics.record_frame!(model, state, dstate, scratch, book, video::DiagnosticVideo, t_iter, params) =
    push_frame!(video, book, model, state, dstate, scratch, t_iter * params.slice_size)

# a copy of the interior of the diagnostic, and its time
function push_frame!(video, book, model, state, dstate, scratch, t)
    session = open(book; state, dstate, scratch, model)
    push!(video.frames, copy(interior_diagnostic(session, video.name)))
    close(session)
    push!(video.times, t)
    return video
end

function Makie.save(file::AbstractString, videos::DiagnosticVideo...; colormap = :thermal, colorrange = nothing,
                    framerate = 20)
    (; name, long_name, units) = first(videos)
    all(v -> v.name == name, videos) || throw(ArgumentError("the videos must show the same diagnostic"))
    nframes = minimum(v -> length(v.frames), videos)
    # colour range of all frames of all videos, by default
    colorrange = something(colorrange, (minimum(v -> minimum(minimum, v.frames), videos),
                                        maximum(v -> maximum(maximum, v.frames), videos)))
    frame = Observable(1)
    # panels 450 px wide, with the aspect ratio of the box, and a colorbar of the same height
    x, z = first(videos).coordinates
    width, height = 450, round(Int, 450 * (z[end] - z[1]) / (x[end] - x[1]))
    figure = Figure()
    for (k, video) in enumerate(videos)
        x, z = video.coordinates
        ax = Axis(figure[1, k]; title = lift(n -> "$(video.label)   t = $(round(video.times[n], digits = 2)) s", frame),
                  xlabel = "x [m]", ylabel = "z [m]", width, height)
        heatmap!(ax, x, z, lift(n -> video.frames[n], frame); colormap, colorrange)
    end
    Colorbar(figure[1, length(videos) + 1]; colormap, limits = colorrange, label = "$long_name [$units]", height)
    Makie.resize_to_layout!(figure)
    record(figure, file, 1:nframes; framerate) do n
        frame[] = n
    end
    return file
end

end # module TraceableEnergeticsMakieExt
