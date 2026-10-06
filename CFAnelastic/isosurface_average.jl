using Plots
using JLD2
using NCDatasets

## PARTITIONS


# create a linear partition `part` of vector `b` with `K` bins
function linear_partition!(part, K, b)
    # extremal points
    b_min = minimum(b)
    b_max = maximum(b)

    # partition edges (exclude min)
    for k in 1:K
        part[k] = b_min + (k / K) * (b_max - b_min)
    end

    # extend max and min
    part[1] -= 1e-6
    part[end] += 1e-6
end

function linear_partition(K, b)
    part = zeros(K)
    linear_partition!(part, K, b)
    return part
end

# label each parcel of field `b` according to where it is placed in the partition `part`
function label_parcels!(labels, b, part)
    for i in eachindex(b)
        # find partition label of parcel
        label = findfirst(x -> x >= b[i], part)

        # map parcel to partition
        labels[i] = label
    end
end

function label_parcels(b, part)
    labels = zeros(Int, length(b))
    label_parcels!(labels, b, part)
    return labels
end

# compute the volume `vols` of each partition section based the map `labels` and parcel volume `dV`
function part_volumes!(vols, labels, dV)
    for i in eachindex(labels)
        # compute partition label of parcel
        label = labels[i]

        # increment volume of label
        vols[label] += 1 * dV
    end
end

function part_volumes(K, labels, dV)
    vols = zeros(K)
    part_volumes!(vols, labels, dV)
    return vols
end

function ref_heights!(zeta, K, Lx, vols)
    for i in 2:K+1
        zeta[i] = sum(vols[1:i-1]) / Lx
    end
end

function ref_heights(K, Lx, vols)
    zeta = zeros(K+1)
    ref_heights!(zeta, K, Lx, vols)
    return zeta
end

function average!(X_avg, X, parcel_labels, zeta, dzeta0, K)
    for k in 1:K
        X_avg[k] = sum(X[parcel_labels .== k]) * dzeta0 / (zeta[k+1] - zeta[k])
    end
end

function average(X, parcel_labels, zeta, dzeta0, K)
    X_avg = zeros(K)
    average!(X_avg, X, parcel_labels, zeta, dzeta0, K)
    return X_avg
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

# parcel volume
dV = dx * dz

# reshape into array and plot
Plots.heatmap(reshape(b, Mx, Mz)', aspect_ratio = :equal)


## compute linear partition of buoyancy

# partition size
K = 128

# compute linear partition of array b
part = linear_partition(K, b)

# map parcels to partition of b
parcel_labels = label_parcels(b, part)

# find volume of each partition interval
vols = part_volumes(K, parcel_labels, dV)

# compute reference heights separating partition intervals
zeta = ref_heights(K, Lx, vols)

Plots.scatter(zeta)

# field X

X = ds["ape"][1:M, 1:N, t]

Plots.heatmap(X')

dzeta0 = dV / Lx

X_avg = average(X, parcel_labels, zeta, dzeta0, K)


zeta_centre = 0.5 * (zeta[1:end-1] .+ zeta[2:end])

Plots.plot(zeta_centre, X_avg)

Plots.heatmap(reshape(parcel_labels, Mx, Mz)')


X_vec = [X_avg[parcel_labels[i]] for i in eachindex(parcel_labels)]

Plots.heatmap(reshape(X_vec, Mx, Mz)')

