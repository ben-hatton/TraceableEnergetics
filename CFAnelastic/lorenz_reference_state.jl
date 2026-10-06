"""
    Reference

Abstract type for reference state types for anelastic available energetics calculations.
"""
abstract type Reference end

"""
    SortedReference

Abstract type for sorted reference state types for anelastic available energetics calculations.
"""
abstract type SortedReference <: Reference end

"""
    PDFReference

Abstract type for PDF reference state types for anelastic available energetics calculations.
"""
abstract type PDFReference <: Reference end

"""
    BoussinesqReference

Abstract type for sorted reference state types for simple Boussinesq available energetics calculations.
"""
abstract type BoussinesqSortedReference <: SortedReference end

"""
    BoussinesqPDFAvailableReference

Abstract type for PDF reference state types for simple Boussinesq available energetics calculations.
"""
abstract type BoussinesqPDFReference <: PDFReference end

"""
    consSorted

Reference state sorted by conservative variable.
"""
struct consvarSorted <: SortedReference end

"""
    vSorted

Reference state sorted by specific volume, cf. Winters et al. (1995).
"""
struct vSorted <: SortedReference end

"""
    vpotSorted
Reference state sorted by potential specific volume, as in Bisits et al. (2025).
"""
struct vpotSorted{F} <: SortedReference
    pref::F
end   

"""
    SimplePDF
Reference state based on single component fluid PDF method (analastic version of Tseng & Ferziger, 2001).
"""
struct SimplePDF{Int} <: PDFReference
    nbins::Int
end   

"""
    BinarySorted
Reference state based on binary sorting method (see Hieronymus & Nycander, 2015; Hosking et al., 2025).
"""
struct BinarySorted <: SortedReference end

"""
    BinaryPDF
Reference state based on binary PDF method (Saenz et al., 2015).
"""
struct BinaryPDF <: PDFReference
    nbins::Int
    # direction of sorting?
    # reference pressure?
end

"""
    UniformBoussinesqPDF
Reference state for a simple Boussinesq model, uniformly spaced PDF in buoyancy (Tseng & Ferziger, 2001)
"""
struct UniformBoussinesqPDF <: BoussinesqPDFReference
    nbins::Int
end
"""
    ChebyshevBoussinesqPDF
Reference state for a simple Boussinesq model, PDF in buoyancy discretised with Chebyshev polynomials (Tseng & Ferziger, 2001)
"""
struct ChebyshevBoussinesqPDF <: BoussinesqPDFReference
    nbins::Int
    # extra arguments here
end
"""
    BinnedBoussinesqSorted
Reference state for a simple Boussinesq model, uniformly spaced in reference height coordinate ζ
"""
struct BinnedBoussinesqSorted <: BoussinesqPDFReference
    nbins::Int
end

# compute reference state in simple Boussinesq using a PDF method
function reference_state!((bvec_, edges_, freqs_, map_, zeta_, zeta_c_, z_star_), b, ref_state_type::UniformBoussinesqPDF, model, nudge=1e-6)
    (; domain, space) = model
    (; nbins) = ref_state_type

    # initialise vectors
    bvec = similar!(bvec_, b, prod(dims(domain)))
    edges = similar!(edges_, b, nbins+1)
    freqs = similar!(freqs_, [0], nbins)
    map = similar!(map_, [0], prod(dims(domain)))
    zeta = similar!(zeta_, b, nbins+1)
    zeta_c = similar!(zeta_c_, b, nbins)
    z_star = similar!(z_star_, b)
    
    # cut off halo cells of buoyancy and put into a vector
    bvec .= reshape(@view(b[xrange_interior(domain), zrange_interior(domain)]), length(bvec)) 

    # compute min, max and bin interval size for buoyancy
    b_min = minimum(bvec) - nudge
    b_max = maximum(bvec) + nudge
    Δ = (b_max - b_min) / nbins

    # compute bin edges
    @inbounds for i in 1:(nbins+1)
        edges[i] = b_min + (i-1)*Δ
    end
    # map state into bins
    freqs .= 0
    @inbounds for i in eachindex(bvec)
        # record bin that parcel is mapped to
        # idx = searchsortedlast(edges, bvec[i])
        idx = floor.(Int, (bvec[i] .- b_min) / Δ + 1)
        map[i] = idx

        # count parcels mapped into each bin
        freqs[idx] += 1
    end

    # create vertical coordinate grid in reference state
    zeta[1] = 0.
    cumsum!(@view(zeta[2:end]), space.Lz * freqs / length(bvec))

    # create cell centre grid by interpolating
    zeta_c .= (zeta[1:end-1] .+ zeta[2:end])/2

    # reference height field
    z_star_int = @view(z_star[xrange_interior(domain), zrange_interior(domain)])
    @inbounds for i in eachindex(bvec)
        z_star_int[i] = zeta_c[map[i]]
    end
    periodize!(model, (z_star,))

    return bvec, edges, freqs, map, zeta, zeta_c, z_star
end

# compute reference state in simple Boussinesq using sorting then binning
function reference_state!((edges_, freqs_, map_, zeta_, zeta_c_, z_star_, bsorted_, perm_, iperm_), b, ref_state_type::BinnedBoussinesqSorted, model, nudge=1e-6)
    (; domain, space, dz) = model
    (Mx, Mz) = dims(domain)
    (; Lz) = space
    (; nbins) = ref_state_type

    # initialise vectors
    edges = similar!(edges_, b, nbins+1)
    freqs = similar!(freqs_, [0], nbins)
    zeta = similar!(zeta_, b, nbins+1)
    zeta_c = similar!(zeta_c_, b, nbins)
    
    map = similar!(map_, [0], prod(dims(domain)))
    perm = similar!(perm_, [0], prod(dims(domain)))
    iperm = similar!(iperm_, [0], prod(dims(domain)))
    bsorted = similar!(bsorted_, b, prod(dims(domain)))
    
    z_star = similar!(z_star_, b)

    # cut off halo cells of buoyancy and put into a vector
    bvec = vec(@view b[xrange_interior(domain), zrange_interior(domain)])

    # sort b, record perm and iperm
    sortperm!(perm, bvec)
    # iperm = invperm(perm)
    # bsorted .= bvec[perm]
    for i in eachindex(perm)
        bsorted[i] = bvec[perm[i]]
        iperm[perm[i]] = i
    end

    # number of parcels per bin
    per_bin = ceil(Int, length(bvec) / nbins)

    # create map to bins
    map = div.(iperm.-1, per_bin).+1

    # zeta coordinate
    @inbounds for i in 0:nbins
        zeta[i] = i*Lz/nbins
    end
    @inbounds for i in 1:nbins
        zeta_c[i] = (i-0.5)*Lz/nbins
    end

    # # reference height field (binned)
    # z_star_int = @view(z_star[xrange_interior(domain), zrange_interior(domain)])
    # @inbounds for i in eachindex(bvec)
    #     z_star_int[i] = zeta_c[map[i]]
    # end

    # reference height field (exact) : this code is a bit dodgy...
    z_star_int = vec(@view z_star[xrange_interior(domain), zrange_interior(domain)])
    for i in eachindex(z_star_int)
        z_star_int[i] = Lz * iperm[i] / (Mx * Mz)
    end
    periodize!(model, (z_star,))

    return edges, freqs, map, zeta, zeta_c, z_star, bsorted, perm, iperm
end

function centred_dif(vec, J)
    return 0.5 * (vec[J+1] - vec[J-1])
end

function reference_state!(ds, keys, model, ref_state_type, params) where {F}
    (; Nslice, slice_size, ae_variables) = params

    # define ref state vertical coordinates zeta and zeta_c
    ~("zeta" in NCDatasets.keys(ds.dim)) && defDim(ds, "zeta", ref_state_type.nbins+1)
    ~("zeta_c" in NCDatasets.keys(ds.dim)) && defDim(ds, "zeta_c", ref_state_type.nbins)

    # define variables from ae_variables
    # assume coordinates x, z, t already defined
    for sym in ae_variables.fields
        ~(string(sym) in NCDatasets.keys(ds)) && defVar(ds, string(sym), F, ("x", "z", "t"))
    end    
    for sym in ae_variables.vecs_center
        ~(string(sym) in NCDatasets.keys(ds)) && defVar(ds, string(sym), F, ("zeta_c", "t"))
    end    
    for sym in ae_variables.vecs_edge
        ~(string(sym) in NCDatasets.keys(ds)) && defVar(ds, string(sym), F, ("zeta", "t"))
    end

    # initialise energetics state (; m, u, w ) from simulation dataset
    state0 = initialise_energetics_state(ds, keys)
    state = deepcopy(state0)

    # initialise dynamics scratch space
    dynamics_dstate, dynamics_scratch = CFTimeSchemes.tendencies!(void, void, model, state0, nothing)

    # initialise energetics scratch space
    ae_scratch = ae_tendencies!(void, model, ref_state_type, state0, dynamics_scratch)

    # run loop
    for t_iter = 1:Nslice
        # update energetics state from dataset
        update_energetics_state!(state, ds, keys, t_iter)

        # compute dynamics scratch space
        dynamics_dstate, dynamics_scratch = CFTimeSchemes.tendencies!(dynamics_dstate, dynamics_scratch, model, state, t_iter * slice_size)

        # compute available energetics
        ae_tendencies!(ae_scratch, model, ref_state_type, state, dynamics_scratch)

        # write data from energetics diagnostics
        session = open(ae_diagnostics(); ae_scratch, model)
        let data = CookBooks.get(session, ae_variables.fields)
            for sym in ae_variables.fields
                ds[sym][:, :, t_iter] = data[sym]
            end
        end
        let data = CookBooks.get(session, ae_variables.vecs_center)
            for sym in ae_variables.vecs_center
                ds[sym][:, t_iter] = data[sym]
            end
        end
        let data = CookBooks.get(session, ae_variables.vecs_edge)
            for sym in ae_variables.vecs_edge
                ds[sym][:, t_iter] = data[sym]
            end
        end
        close(session);

        # advance progress meter
        progress(t_iter, slice_size, Nslice * slice_size)
    end
end