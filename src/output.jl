## OUTPUT OF THE DIAGNOSTICS DURING A RUN
#
# `loop` calls its `outputs` at t = 0 and at the end of each time slice, and closes them when the run ends. `Outputs`
# opens the diagnostics session once per call and hands it to each output: a `DataFile` (NetCDF or JLD2) writes the
# full arrays, with halo, for post-processing, and is closed at the end; a `DiagnosticVideo` keeps the interior, for
# viewing, and is saved at the end. Any other output only needs `record!(output, session, t)` and `close(output)`.

"""
    outputs = Outputs(output...)

Outputs of a run, the `outputs` of `loop`: called as `outputs(model, state, dstate, scratch, book, t)` at time `t`
[s], it records the diagnostics of the state in each output with `record!`, and `close(outputs)`, called by `loop`
when the run ends, closes the data files and saves the videos. `outputs[k]` is the k-th output.

# Example
```julia
outputs = Outputs(DataFile("run.nc", model, (:u, :w)), DataFile("run.jld2", model, (:u, :w)),
                  DiagnosticVideo(model, :potential_temperature; file = "run.mp4"))
CFAnelastics.loop(model, state, scheme, params; outputs)     # writes run.nc, run.jld2 and run.mp4
```
"""
struct Outputs{O <: Tuple}
    outputs::O
end
Outputs(outputs...) = Outputs(outputs)

function (outputs::Outputs)(model, state, dstate, scratch, book, t)
    session = open(book; state, dstate, scratch, model)
    foreach(output -> record!(output, session, t), outputs.outputs)
    close(session)
    return outputs
end
Base.getindex(outputs::Outputs, k) = outputs.outputs[k]
Base.close(outputs::Outputs) = foreach(close, outputs.outputs)

"""
    output = record!(output, session, t)

Record the diagnostics of the open diagnostics `session` at time `t` [s] in `output` (a `DataFile`, a
`DiagnosticVideo`, or any output of `Outputs`).
"""
function record! end

#####
##### Data files
#####

"""
    file = DataFile(path, model, names)
    field, t = read_diagnostic(path, name)

File of the diagnostics `names` (a tuple of symbols, see `diagnostic_info(model)`) of `model`, as full arrays with
halo, one per time slice, for post-processing; the format is chosen from the extension of `path`:
- `.nc`: NetCDF, a variable per diagnostic on the dimensions of its grid location and time, e.g. `(X, z, t)` for
  `(Face(), Cell())`, with its `long_name`, `units` and `location` ("Xz"), the coordinates (x, X, z, Z) [m] of every
  point, halo included, the times `t` [s], and the grid (Mx, Mz, Hx, Hz, Lx, Lz, dx, dz) as global attributes;
- `.jld2`: JLD2, the same contents, with the diagnostic at slice n under `"name/n"` and its time under `"t/n"`.
An existing file is overwritten. `loop` closes it when the run ends (otherwise, `close(file)`).

`read_diagnostic` reads back the diagnostic `name` from either format: an (x, z, t) array, with halo, and the times
[s].
"""
struct DataFile{Handle, Names}
    handle::Handle              # NCDataset or JLD2 file
    names::Names                # tuple of the recorded diagnostics
    slices::Base.RefValue{Int}  # number of slices recorded so far
end

function DataFile(path::AbstractString, model, names::Tuple)
    mkpath(dirname(abspath(path)))
    info = NamedTuple{names}(diagnostic_info(model))
    handle = create_file(Val(Symbol(extension(path))), path, model, info)
    return DataFile(handle, names, Ref(0))
end

Base.close(file::DataFile) = close(file.handle)

read_diagnostic(path::AbstractString, name) = read_diagnostic(Val(Symbol(extension(path))), path, name)

extension(path) = lstrip(splitext(path)[2], '.')
create_file(::Val{E}, path, model, info) where E = throw(ArgumentError("unknown data file format .$E, use .nc or .jld2"))
read_diagnostic(::Val{E}, path, name) where E = throw(ArgumentError("unknown data file format .$E, use .nc or .jld2"))

# grid of the model, and the coordinates [m] of each grid location, halo included
grid((; domain, space, dx, dz)) = let (Mx, Mz) = dims(domain), (Hx, Hz) = halo_size(domain)
    (; Mx, Mz, Hx, Hz, space.Lx, space.Lz, dx, dz)
end
grid_coordinates((; domain, dx, dz)) =
    (; x = cellgrid(domain, dx, 1), X = facegrid(domain, dx, 1), z = cellgrid(domain, dz, 2), Z = facegrid(domain, dz, 2))

## NetCDF

function create_file(::Val{:nc}, path, model, info)
    ds = NCDataset(path, "c")
    ds.attrib["model"] = string(nameof(typeof(model)))
    foreach(((key, value),) -> ds.attrib[string(key)] = value, pairs(grid(model)))
    # a dimension per grid letter, halo included, with its coordinates, and the time, of unlimited length
    for (dim, coordinate) in pairs(grid_coordinates(model))
        defVar(ds, string(dim), coordinate, (string(dim),); attrib = ["units" => "m"])
    end
    defDim(ds, "t", Inf)
    defVar(ds, "t", typeof(model.dx), ("t",); attrib = ["units" => "s"])
    # a variable per diagnostic, on the dimensions of its grid location, e.g. (Face(), Cell()) ↦ ("X", "z", "t")
    for (name, (; long_name, units, location)) in pairs(info)
        dims = map(dimname, location, (1, 2))
        defVar(ds, string(name), typeof(model.dx), (dims..., "t");
               attrib = ["long_name" => long_name, "units" => units, "location" => join(dims)])
    end
    return ds
end

# NetCDF dimension name along dimension d (1 = x, 2 = z) of a grid location: lower case cells, upper case faces
dimname(::Cell, d) = ("x", "z")[d]
dimname(::Face, d) = ("X", "Z")[d]

function record!(file::DataFile{<:NCDataset}, session, t)
    n = file.slices[] += 1
    ds = file.handle
    ds["t"][n] = t
    for name in file.names
        ds[string(name)][:, :, n] = CookBooks.get(session, name)
    end
    return file
end

read_diagnostic(::Val{:nc}, path, name) =
    NCDataset(ds -> (Array(ds[string(name)].var), Array(ds["t"].var)), path)

## JLD2

function create_file(::Val{:jld2}, path, model, info)
    file = jldopen(path, "w")
    file["model"] = string(nameof(typeof(model)))
    file["grid"] = grid(model)
    file["coordinates"] = grid_coordinates(model)
    file["info"] = info
    return file
end

function record!(file::DataFile{<:JLD2.JLDFile}, session, t)
    n = file.slices[] += 1
    file.handle["t/$n"] = t
    for name in file.names
        file.handle["$name/$n"] = CookBooks.get(session, name)
    end
    return file
end

read_diagnostic(::Val{:jld2}, path, name) = jldopen(path, "r") do file
    slices = 1:length(keys(file["t"]))
    return stack(n -> file["$name/$n"], slices), [file["t/$n"] for n in slices]
end

#####
##### Videos
#####

"""
    video = DiagnosticVideo(model, name; file = nothing, label = "AN2D" or "FC2D", style...)
    save(file, videos...; colormap = :thermal, colorrange, framerate = 20)

Frames of the diagnostic `name` of `model` on the interior of the box, for viewing: one frame per `record!`, e.g. as
an output of `Outputs`. With a `file`, `close(video)` (called by `loop` when the run ends) saves it there, with the
keywords `style` of `save`. `save` saves one or more videos of the same diagnostic, e.g. of two models, as heatmaps
side by side, with a shared colorbar of its long name and units; the colour range defaults to that of all frames.
"""
struct DiagnosticVideo{F, Coordinates, Style}
    file::Union{Nothing, String}    # where `close` saves the video, if anywhere
    style::Style                    # keywords of `save`
    name::Symbol
    label::String
    long_name::String
    units::String
    coordinates::Coordinates    # (x, z) of the interior points of the diagnostic [m]
    frames::Vector{Matrix{F}}   # the diagnostic on the interior, one matrix per frame
    times::Vector{F}            # time of each frame [s]
end

function DiagnosticVideo(model, name::Symbol; file = nothing, label = string(nameof(typeof(model))), style...)
    (; long_name, units) = diagnostic_info(model)[name]
    F = typeof(model.dx)
    return DiagnosticVideo(file, NamedTuple(style), name, label, long_name, units, diagnostic_coordinates(model, name),
                           Matrix{F}[], F[])
end

function record!(video::DiagnosticVideo, session, t)
    push!(video.frames, copy(interior_diagnostic(session, video.name)))
    push!(video.times, t)
    return video
end

function Base.close(video::DiagnosticVideo)
    isnothing(video.file) || Makie.save(video.file, video; video.style...)
    return video
end

function Makie.save(file::AbstractString, videos::DiagnosticVideo...; colormap = :thermal, colorrange = nothing,
                    framerate = 20)
    mkpath(dirname(abspath(file)))
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
