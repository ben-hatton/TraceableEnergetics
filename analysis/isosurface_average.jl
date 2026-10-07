include("../tools/setup.jl")
cd("$(@__DIR__)")

using Plots
using JLD2
using NCDatasets

# histogram types

abstract type AbstractHistogram end

"""
Histogram with uniform partition between min_b and max_b
"""
mutable struct UniformHistogram{T} <: AbstractHistogram
    edges::Vector{T}
    frequencies::Vector{T}
end

function UniformHistogram(n::Int, T=Float64)
    edges = zeros(T, n+1)
    frequencies = zeros(T, n)
    UniformHistogram(edges, frequencies)
end

function update!(h::UniformHistogram, map, b, nudge=1e-6)
    min_b = minimum(b) - nudge
    max_b = maximum(b) + nudge

    n = length(h.edges)
    Δ = (max_b - min_b) / (n - 1)

    # update edges
    @inbounds for i in 1:n
        h.edges[i] = min_b + (i-1)*Δ
    end

    # fill frequency and record mapping
    @inbounds for i in eachindex(b)
        # record bin that parcel is mapped to
        idx = searchsortedlast(h.edges, b[i])
        map[i] = idx

        # add to the count
        h.frequencies[idx] += 1
    end

    return h, map
end


# isosurface average

function average!(X_avg, X, parcel_to_bin, dζ, nbins)
    for i in 1:nbins
        X_avg[i] = sum(X[parcel_to_bin .== i]) / dζ[i]
    end
end


##  buoyancy field

data_file = "../../experiments/AN/shear_diagram/data/fields.nc"
ds = NCDatasets.NCDataset(data_file, "a")

vars = JLD2.load("../../experiments/AN/shear_diagram/data/vars.jld2")

(; M, N) = vars["choices"]
(; L, dx) = vars["params"]


t = 50

b = ds["s"][1:M, 1:N, t]

# grid dimensions
Mx, Mz = size(b)

b = reshape(b, M*N)

# physical scales
Lx = L
Lz = (Lx / Mx) * Mz

# grid resolution
dx = Lx / Mx
dz = Lz / Mz

# # parcel volume
dV = dx * dz

# # domain area
A = Lx

# reshape into array and plot
Plots.heatmap(reshape(b, Mx, Mz)', aspect_ratio = :equal)

# initialisations
nbins = Mz
nparcels = Mx*Mz

hist = UniformHistogram(nbins)          # parcel histogram
parcel_to_bin = zeros(Int, nparcels)    # map: parcel → bin

ζ   = zeros(nbins + 1)                  # reference state edge grid on
ζ_c = zeros(nbins)                      # reference state center grid
dζ  = zeros(nbins)                      # normalised difference of ζ
z_star = zeros(Mx * Mz)                 # reference height in situ

X_avg = zeros(nbins)                    # average of field X

b_min = minimum(b) - 1e-6
b_max = maximum(b) + 1e-6
idx = floor.(Int, nbins * (b .- b_min) / (b_max - b_min).+1)
minimum(idx)

# update buoyancy histogram and record where parcels are mapped
update!(hist, parcel_to_bin, b)

# update reference state grid
cumsum!(@view(ζ[2:end]), Lz * hist.frequencies / nparcels)

# apply grid operations in ref state
ζ_c[:] = 0.5 * (ζ[1:end-1] .+ ζ[2:end])
dζ[:] = (ζ[2:end] .- ζ[1:end-1]) * A / dV

# reference height field
z_star[:] = ζ_c[parcel_to_bin]

# average X
X = ds["ape"][1:M, 1:N, t]
average!(X_avg, X, parcel_to_bin, dζ, nbins)

# plot reference height
Plots.heatmap(reshape(z_star, Mx, Mz)')

# plot average X
Plots.plot(ζ_c, X_avg, xlabel="ζ", ylabel="<X>")

clims = extrema(X)
Plots.heatmap(reshape(X_avg[parcel_to_bin], Mx, Mz)', clims=clims)
Plots.heatmap(X', clims=clims)

Xp = X .- reshape(X_avg[parcel_to_bin], Mx, Mz)
Plots.heatmap(Xp', colormap = :balance,
    clims = (-1, 1) .* maximum(abs, Xp))



# alernative: bin according to zeta (still required a full sort?)
nbins = 128

per_bin = ceil(Int, length(b) / nbins)
sort_idx = sortperm(b)
inv_idx = invperm(sort_idx)

Plots.heatmap(reshape(inv_idx, Mx, Mz))

bin_idx = div.(inv_idx, per_bin).+1

Plots.heatmap(reshape(bin_idx, Mx, Mz))

z_star_zeta = (bin_idx .- 0.5) * (Mz / nbins) * dz

Plots.heatmap(reshape(z_star_zeta, Mx, Mz)')



X_sort = X[sort_idx]

X_avg_zeta = [sum(X_sort[((k-1)*per_bin+1):min((k*per_bin), length(X))])/per_bin for k in 1:nbins]

zeta_vec = [0.5 * (Mz / nbins) * dz + k * (Mz / nbins) * dz for k in 0:nbins-1]
Plots.plot(zeta_vec, X_avg_zeta)