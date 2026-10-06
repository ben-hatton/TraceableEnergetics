function compute_pe!(pe, b, model)
    print("\nComputing PE...\n")
    (; mgr, domain, dz) = model

    # number of time slices
    Nslice  = size(pe, 3)
    
    # scratch
    pe_  = Matrix{Float64}(undef, xdim(domain), zdim(domain))
    b_   = Matrix{Float64}(undef, xdim(domain), zdim(domain))

    # z coordinates
    z_ = [zpoint(domain, dz, k) for i in 1:xdim(domain), k in 1:zdim(domain)]

    for t in 1:Nslice
        # read slices
        b_ .= b[:, :, t]

        # compute epsp_
        @with mgr, let (irange, jrange) = (axes(pe_, 1), axes(pe_, 2))
            @vec for i in irange, j in jrange
                pe_[i, j] = - b_[i, j] * z_[i, j]
            end
        end

        # write slice
        pe[:, :, t] = pe_
        
        # progress monitor
        print("\rProgress: $t / $Nslice slices")
        flush(stdout) 
    end
end
# compute reference state [b: buoyancy array without halo]
function compute_ref_state!((perm, iperm, ipermbin), (b, zeta, Zeta), model; del = 0.)
    print("\nComputing reference state...\n")
    (; domain) = model
    Mx, Mz = dims(domain)
    int_i = xrange_interior(domain)
    int_j = zrange_interior(domain)
    
    # number of time slices
    Nslice  = size(b, 3)

    # number of bins
    Nbins   = size(zeta, 1)

    # number of parcels per bin
    per_bin     = ceil(Int, Mx*Mz / Nbins)

    # define scratch arrays
    bvec        = Vector{Float64}(undef, Mx*Mz)
    b_          = reshape(bvec, Mx, Mz)         # same data as bvec
    perm_       = Vector{Int}(undef, Mx*Mz)
    iperm_      = Vector{Int}(undef, Mx*Mz)
    ipermbin_   = Vector{Int}(undef, Mx*Mz)


    # define tiny vertical profile
    bmin = minimum(b)
    bmax = maximum(b)
    Δb = bmax - bmin 
    b_add = vec([del * (bmin + Δb * j / Mz) for i in 1:Mx, j in 1:Mz])

    # time loop
    for t in 1:Nslice
        # read buoyancy at time t
        b_ .= b[int_i, int_j, t]

        # add tiny gradient to reduce noise effect on sorting
        bvec .+= b_add

        # sort buoyancy
        sortperm!(perm_, bvec)

        # compute inverse permutation
        @inbounds for i in 1:length(perm_)
            iperm_[perm_[i]] = i
        end

        # bin parcels
        @. ipermbin_ = div(iperm_ - 1, per_bin) + 1

        # write slices
        perm[:, t]      = perm_
        iperm[:, t]     = iperm_
        ipermbin[:, t]  = ipermbin_

        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end


# X[perm] : sorts a state array X into the reference state
# iperm     = [level1, level2, level3, ...] : gives a list of parcel levels in the reference state
# ipermbin  = [bin1, bin2, bin3, ...] : tells us which bin each parcel is mapped to


function compute_zref!(zref, (iperm, zeta), model)
    print("\nComputing z*(x, z, t)...\n")
    (; domain, dz, space) = model
    Mx, Mz = dims(domain)
    Hx, Hz = halo_size(domain)
    int_i = xrange_interior(domain)
    int_j = zrange_interior(domain)
    
    # number of time slices
    Nslice  = size(zref, 3)
    
    # define scratch arrays
    zrefvec = Vector{Float64}(undef, Mx * Mz)
    zref_ = reshape(zrefvec, Mx, Mz)
    iperm_ = Vector{Int}(undef, Mx * Mz)
    
    # sorted reference height resolution
    dζ = dz / Mx
    
    for t in 1:Nslice
        # read slice
        iperm_ .= iperm[:, t]
        
        # compute sorted reference height for each parcel
        
        zrefvec .= (0.5 .+ iperm_) .* dζ
        
        # # compute reference bin for each parcel
        # for i in 1:(Mx*Mz)
        #     zrefbinvec[i] = zeta[ipermbin_[i]]
        # end
        
        # write slice
        zref[int_i, int_j, t] = zref_
        
        # fill lateral halo cells
        zref[1:Hx, (1+Hz):(Mz+Hz), t] = zref_[(1+Mx-Hx):Mx, :]
        zref[(1+Mx+Hz):(Mx+2Hz), (1+Hz):(Mz+Hz), t] = zref_[1:Hx, :]
        
        # fill top and bottom halo cells
        zref[:, 1:Hz, t] .= - 0.5 * dζ
        zref[:, (1+Mz+Hz):(Mz+2Hz), t] .= space.Lz + 0.5 * dζ
        
        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
    
    # # fill halo cells for zref
    # zref[1:Hx, :, :] = zref[(1+Mx):(Hx+Mx), :]
    # zref[(1+Mx+Hz):(Mx+2Hz), :, :] = zref[(1+Hx):2Hx, :]
end


function compute_grad!((fx, fz), f, model)
    print("\nComputing gradient...\n")
    (; mgr, domain, inv_dx, inv_dz) = model
    
    # number of time slices
    Nslice  = size(zref, 3)
    
    # define scratch arrays
    f_      = Matrix{Float64}(undef, xdim(domain), zdim(domain))
    # gradf_  = Matrix{Float64}(undef, xdim(domain), zdim(domain))
    fx_     = Matrix{Float64}(undef, Xdim(domain), zdim(domain))
    fz_     = Matrix{Float64}(undef, xdim(domain), Zdim(domain))
    
    for t in 1:Nslice
        # read slice
        f_ .= f[:, :, t]
        
        # compute derivatives
        @with mgr, let (irange, jrange) = (Xrange(domain), axes(fx_, 2))
            @vec for I in irange, j in jrange
                fx_[I, j] = inv_dx * dif_X(f_, I, j)
            end
        end
        @with mgr, let (irange, jrange) = (axes(fz, 1), Zrange(domain))
            @vec for i in irange, J in jrange
                fz_[i, J] = inv_dz * dif_Z(f_, i, J)
            end
        end

        # periodize arrays
        periodize!(model, (fx_, fz_))

        # # interpolate onto cell centres
        # @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        #     @vec for i in irange, j in jrange
        #         gradf_[i, j] = sqrt( avg_x(abs2, fx_, i, j) + avg_z(abs2, fz_, i, j))
        #     end
        # end

        # # periodize array
        # periodize!(model, (gradf_,))

        # write arrays
        fx[:, :, t] = fx_
        fz[:, :, t] = fz_
        # gradf[:, :, t] = gradf_
        
        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end

function isoaverage!(Xa, X, ipermbin, Zeta, model)
    print("\nComputing isopycnal average...\n")
    (; dx, dz, space) = model
    (; Lx) = space
    Mx, Mz = dims(domain)
    
    # factor in front of average sum
    dV_A = dx * dz / Lx
    
    # number of time slices
    Nslice  = size(zref, 3)
    
    # interior ranges
    int_i = xrange_interior(domain)
    int_j = zrange_interior(domain)
    
    # scratch
    Xa_         = Vector{Float64}(undef, size(Xa, 1))
    Xvec        = Vector{Float64}(undef, Mx * Mz)
    X_          = reshape(Xvec, Mx, Mz)
    ipermbin_   = Vector{Int}(undef, size(ipermbin, 1))
    
    for t in 1:Nslice
        X_          .= X[int_i, int_j, t]
        ipermbin_   .= ipermbin[:, t]
        
        # compute average
        fill!(Xa_, 0.0)
        @inbounds for i in eachindex(Xvec)
            Xa_[ipermbin_[i]] += Xvec[i]
        end
        @inbounds for k in eachindex(Xa_)
            Xa_[k] *= dV_A / (Zeta[k+1] - Zeta[k])
        end
        
        # write array
        Xa[:, t] = Xa_
        
        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end

function compute_N2!((N2, N2ref), (bref, Zeta, ipermbin), model)
    print("\nComputing N²* = ∂b*/∂ζ...\n")
    (; domain) = model
    Mx, Mz = dims(domain)
    Hx, Hz = halo_size(domain)
    int_i = xrange_interior(domain)
    int_j = zrange_interior(domain)
    
    # number of time slices
    Nslice  = size(b, 3)

    # number of bins
    Nbins   = size(zeta, 1)

    ipermbin_   = Vector{Int}(undef, Mx*Mz)    
    bref_       = Vector{Float64}(undef, Nbins)

    N2_         = Vector{Float64}(undef, Nbins)
    N2refvec    = Vector{Float64}(undef, Mx*Mz)
    N2ref_      = reshape(N2refvec, Mx, Mz) 

    dZeta_ = Zeta[2:end] .- Zeta[1:end-1]

    # prefill boundaries
    N2_[1]    = 0.
    N2_[end]  = 0.

    for t in 1:Nslice
        # load slices
        bref_   .= bref[:, t]
        ipermbin_ .= ipermbin[:, t]

        # compute gradient
        @inbounds for k in 2:(Nbins-1)
            N2_[k] = (bref_[k+1] - bref_[k-1]) / (2 * dZeta_[k])
        end

        # fill in field
        for i in eachindex(N2refvec)
            N2refvec[i] = N2_[ipermbin_[i]]
        end

        # write slice
        N2[:, t] = N2_
        N2ref[int_i, int_j, t] = N2ref_
        
        # fill lateral halo cells
        N2ref[1:Hx, (1+Hz):(Mz+Hz), t] = N2ref_[(1+Mx-Hx):Mx, :]
        N2ref[(1+Mx+Hz):(Mx+2Hz), (1+Hz):(Mz+Hz), t] = N2ref_[1:Hx, :]

        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end


# compute β(x, z, t) = b(x, z, t) - b*(z, t)
function compute_beta!(beta, (b, bref), model)
    print("\nComputing β(x, z, t) = b(x, z, t) - b*(z, t)...\n")

    (; domain) = model
    Mx, Mz = dims(domain)
    Hx, Hz = halo_size(domain)
    int_i, int_j = xrange_interior(domain), zrange_interior(domain)

    # number of time slices
    Nslice  = size(b, 3)

    # number of bins
    Nbins = size(bref, 1)

    # assert that in situ and ref state have same resolution
    # can generalise and remove this restriction later
    @assert Nbins == Mz
    
    # scratch
    beta_   = Matrix{Float64}(undef, Mx, Mz)
    b_      = Matrix{Float64}(undef, Mx, Mz)
    bref_   = Vector{Float64}(undef, Nbins)

    for t in 1:Nslice
        # read slices
        b_      .= b[int_i, int_j, t]
        bref_   .= bref[:, t]

        # compute beta
        @. beta_ = b_ - bref_'    # bref_' broadcasts across rows
        
        
        # write slice
        beta[int_i, int_j, t] = beta_
        
        # manually periodize
        beta[1:Hx, (1+Hz):(Mz+Hz), t] = beta_[(1+Mx-Hx):Mx, :]
        beta[(1+Mx+Hz):(Mx+2Hz), (1+Hz):(Mz+Hz), t] = beta_[1:Hx, :]
        
        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end

function center_w!(wc, w, model)
    print("\nInterpolating w variable onto cell centre...\n")

    (; mgr, domain) = model

    # number of time slices
    Nslice = size(w, 3)

    # scratch
    w_  = Matrix{Float64}(undef, xdim(domain), Zdim(domain))
    wc_ = Matrix{Float64}(undef, xdim(domain), zdim(domain))

    for t in 1:Nslice
        w_ .= w[:, :, t]
        @with mgr, let (irange, jrange) = (axes(w, 1), zrange(domain))
            @vec for i in irange, j in jrange                      
                wc_[i, j] = avg_z(w_, i, j)
            end       
        end
        wc[:, :, t] = wc_

        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end

function center_u!(uc, u, model)
    print("\nInterpolating u variable onto cell centre...\n")
    (; mgr, domain) = model

    # number of time slices
    Nslice = size(u, 3)

    # scratch
    u_  = Matrix{Float64}(undef, Xdim(domain), zdim(domain))
    uc_ = Matrix{Float64}(undef, xdim(domain), zdim(domain))

    for t in 1:Nslice
        u_ .= u[:, :, t]
        @with mgr, let (irange, jrange) = (xrange(domain), axes(u_, 2))
            @vec for i in irange, j in jrange                      
                uc_[i, j] = avg_x(u_, i, j)
            end       
        end
        uc[:, :, t] = uc_

        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end

function call_diagnostics!(vars, symbols, (m, u, w), model)
    print("\nComputing diagnostics $(String.(symbols))...\n")

    (; domain) = model

    Nslice = size(u, 3)

    @assert length(vars) == length(symbols)

    # scratch
    state = (
        m = Array{Float64}(undef, xdim(domain), zdim(domain), 2),
        u = Array{Float64}(undef, Xdim(domain), zdim(domain)), 
        w = Array{Float64}(undef, xdim(domain), Zdim(domain))
        )

    dstate, scratch = CFTimeSchemes.tendencies!(void, void, model, state, nothing)


    for t in 1:Nslice
        # read state, scratch and dstate slices
        state.m .= m[:, :, :, t]
        state.u .= u[:, :, t]
        state.w .= w[:, :, t]
        
        dstate, scratch = CFTimeSchemes.tendencies!(dstate, scratch, model, state, nothing)
        
        # open diagnostics
        session = open(CFAnelastic.diagnostics(model); state, dstate, scratch, model)

        # write data from diagnostics
        let data = CookBooks.get(session, symbols)
            for i in eachindex(vars)
                vars[i][:, :, t] = data[symbols[i]]
            end
        end
        close(session)

        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end


# compute εₚ = (z - z*)σ_b + j_b⋅∇(z - z*)
function compute_epsp!((epsp, epsp_tur, epsp_lam), (sigmab, jbx, jbz, dzref_x, dzref_z, zref), model)
    print("\nComputing εₚ...\n")
    
    (; mgr, domain, dz) = model

    # number of time slices
    Nslice  = size(epsp, 3)
    
    # scratch
    sigmab_ = Matrix{Float64}(undef, xdim(domain), zdim(domain))
    jbx_    = Matrix{Float64}(undef, Xdim(domain), zdim(domain))
    jbz_    = Matrix{Float64}(undef, xdim(domain), Zdim(domain))
    dzref_x_= Matrix{Float64}(undef, Xdim(domain), zdim(domain))
    dzref_z_= Matrix{Float64}(undef, xdim(domain), Zdim(domain))
    zref_   = Matrix{Float64}(undef, xdim(domain), zdim(domain))

    # z coordinates
    z_ = [zpoint(domain, dz, k) for i in 1:xdim(domain), k in 1:zdim(domain)]

    epsp_   = Matrix{Float64}(undef, xdim(domain), zdim(domain))
    epsp_tur_   = Matrix{Float64}(undef, xdim(domain), zdim(domain))
    epsp_lam_   = Matrix{Float64}(undef, xdim(domain), zdim(domain))

    for t in 1:Nslice
        # read slices
        sigmab_     .= sigmab[:, :, t]
        jbx_        .= jbx[:, :, t]
        jbz_        .= jbz[:, :, t]
        dzref_x_    .= dzref_x[:, :, t]
        dzref_z_    .= dzref_z[:, :, t]
        zref_       .= zref[:, :, t]

        # compute epsp_
        @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
            @vec for i in irange, j in jrange
                epsp_prod = 0. #(z_[i, j] - zref_[i, j]) * sigmab_[i, j]
                epsp_tur_[i, j] = - avg_x(jbx_, dzref_x_, i, j) - avg_z(jbz_, dzref_z_, i, j)
                epsp_lam_[i, j] = avg_z(jbz_, i, j)
                epsp_[i, j] =  epsp_prod + epsp_tur_[i, j] + epsp_lam_[i, j]
            end
        end
        periodize!(model, (epsp_, epsp_lam_, epsp_tur_))

        # write slice
        epsp[:, :, t] = epsp_
        epsp_tur[:, :, t] = epsp_tur_
        epsp_lam[:, :, t] = epsp_lam_
        
        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end

function dzeta!(df_zeta, f, Zeta, model)
    print("\nComputing ∂f/∂ζ*...\n")
    (; domain) = model
    Mx, Mz = dims(domain)
    
    # number of time slices
    Nslice  = size(f, 2)

    # number of bins
    Nbins   = size(zeta, 1)

    f_       = Vector{Float64}(undef, Nbins)
    df_zeta_ = Vector{Float64}(undef, Nbins)

    dZeta_ = Zeta[2:end] .- Zeta[1:end-1]

    # prefill boundaries with Neumann conditions
    df_zeta_[1]    = 0.
    df_zeta_[end]  = 0.

    for t in 1:Nslice
        # load slice
        f_   .= f[:, t]

        # compute gradient
        @inbounds for k in 2:(Nbins-1)
            df_zeta_[k] = (f_[k+1] - f_[k-1]) / (2 * dZeta_[k])
        end
        # write slice
        df_zeta[:, t] = df_zeta_

        # progress monitor
        print("\rProgress: $t / $Nslice slices", " " ^ 10)
        flush(stdout) 
    end
end

function dt!(df_t, f, dt)
    print("\nComputing ∂f/∂t*...\n")

    f_fwd = f[:, 2:end]
    f_bwd = f[:, 1:end-1]

    diff = f_fwd .- f_bwd

    df_t[:, 1:end-1] = diff / dt
    df_t[:, end] = diff[:, end] / dt
end
