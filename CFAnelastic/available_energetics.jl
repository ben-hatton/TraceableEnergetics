"""
    isosurface_average_boussinesq!(X_avg, X, map, dζ, nbins)
Updates X_avg in place with a thickness-weighted average of X, given a `map` to the reference state,
the normalised reference grid seperations `dζ`
"""
function isosurface_average_boussinesq!(X_avg_, X, (map, zeta), ref_state_type::BoussinesqPDFReference, model)
    X_avg = similar!(X_avg_, X, ref_state_type.nbins)
    
    (; nbins) = ref_state_type
    A = model.space.Lx
    dV = model.dx * model.dz
    dV_A = dV / A
    @inbounds for i in 1:nbins
        X_avg[i] =  dV_A * sum(X[map .== i]) / (zeta[i+1] - zeta[i])
    end
    return X_avg
end

# # maps an averaged field to the in situ state
# function ref_to_state!(X, Xa, map_field)
#     @with model.mgr, for i in xrange_interior(domain), j in zrange_interior(domain)
#         X[i, j] = Xa[round(Int, map_field[i, j])]
#     end
#     periodize!(model, X)
#     return X
# end

# yields Xp(x, z) = X(x, z) - Xa(z*(x, z))
function prime!(Xp_, (X, Xa), map_field, model)
    (; domain) = model
    Xp = similar!(Xp_, X)
    @with model.mgr, let (irange, jrange) = (xrange_interior(domain), zrange_interior(domain))
        @vec for i in irange, j in jrange
            Xp[i, j] = X[i, j] - Xa[map_field[i, j]]
        end
    end
    periodize!(model, X)
    return Xp
end

## AVAILABLE ENERGETICS
# available energetics tendencies: BoussinesqPDFReference
function ae_tendencies!(ae_scratch, model::AN2D, ref_state_type::BoussinesqPDFReference, state, dynamics_scratch)
    (; domain, reference) = model
    (Mx, Mz) = dims(domain)
    (; ρ_R) = reference
    rho0 = ρ_R[1]

    # read buoyancy
    b = dynamics_scratch.thermo.b

    # compute reference state
    (; bsorted, edges, freqs, map, zeta, zeta_c, perm, iperm) = ae_scratch.reference_state
    (; z_star) = ae_scratch.fields
    edges, freqs, map, zeta, zeta_c, z_star, bsorted, perm, iperm = reference_state!((edges, freqs, map, zeta, zeta_c, z_star, bsorted, perm, iperm), b, ref_state_type, model)

    # create a field that maps to the reference state
    (; map_field) = ae_scratch.fields
    map_field = similar!(map_field, [0], xdim(domain), zdim(domain))
    map_field .= 0
    map_field[xrange_interior(domain), zrange_interior(domain)] = @views reshape(map, Mx, Mz)
    periodize!(model, (map_field,))

    # compute reference stratification
    (; N2) = ae_scratch.fields
    # (; N2a) = ae_scratch.averages
    N2 = buoyancy_freq!(N2, (bsorted, perm, z_star), model)

    # compute buoyancy profile
    (; b_star) = ae_scratch.averages
    b_star = @views isosurface_average_boussinesq!(b_star, interior(domain, b), (map, zeta), ref_state_type, model)

    # interpolate velocities to centre
    (; uc, wc) = ae_scratch.fields
    uc = centre_Xz!(uc, state.u, model)
    wc = centre_xZ!(wc, state.w, model)

    # average velocities
    (; ua, wa) = ae_scratch.averages
    ua = @views isosurface_average_boussinesq!(ua, interior(domain, uc), (map, zeta), ref_state_type, model)
    wa = @views isosurface_average_boussinesq!(wa, interior(domain, wc), (map, zeta), ref_state_type, model)
    
    # resolved kinetic energy
    (; rke) = ae_scratch.averages
    rke = similar!(rke, ua)
    rke = rke!(rke, (ua, wa))

    # velocity perturbations
    (; up, wp) = ae_scratch.fields
    up = prime!(up, (uc, ua), map_field, model)
    wp = prime!(wp, (wc, wa), map_field, model)
    
    # perturbation kinetic energy
    (; kep) = ae_scratch.fields
    kep = kep!(kep, (up, wp), model)
    
    # turbulent kinetic energy
    (; tke) = ae_scratch.averages
    tke = @views isosurface_average_boussinesq!(tke, interior(domain, kep), (map, zeta), ref_state_type, model)
    
    # grad z_star
    (; ∂z_star∂x, ∂z_star∂z, ∂z_star∂x_c, ∂z_star∂z_c, grad_z_star ) = ae_scratch.fields
    ∂z_star∂x, ∂z_star∂z, ∂z_star∂x_c, ∂z_star∂z_c, grad_z_star = grad!((∂z_star∂x, ∂z_star∂z, ∂z_star∂x_c, ∂z_star∂z_c, grad_z_star), z_star, state, model) 

    # <|∇z*|>
    (; grad_z_star_a) = ae_scratch.averages
    grad_z_star_a = @views isosurface_average_boussinesq!(ua, interior(domain, grad_z_star), (map, zeta), ref_state_type, model)

    # energy conversions
    (; bw, eps_k) = ae_scratch.averages
    bw = similar!(bw, b_star)
    eps_k = similar!(eps_k, b_star)
    bw .= b_star .* wa
    eps_k_state = dynamics_scratch.irreversible.ρε / rho0
    eps_k = @views isosurface_average_boussinesq!(eps_k, interior(domain, eps_k_state), (map, zeta), ref_state_type, model)
    
    ae_scratch = (;
        fields = (; z_star, uc, wc, up, wp, kep, map_field, ∂z_star∂x, ∂z_star∂z, ∂z_star∂x_c, ∂z_star∂z_c, grad_z_star, N2),
        averages = (; b_star, ua, wa, rke, tke, bw, eps_k, grad_z_star_a),
        reference_state = (; bsorted, edges, freqs, map, zeta, zeta_c, perm, iperm)
    )
    return ae_scratch
end

function centre_Xz!(uc_, u, model)
    (; mgr, domain) = model
    uc = similar!(uc_, u, xdim(domain), zdim(domain))
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange                      
            uc[i, j] = avg_x(u, i, j)
        end       
    end
    return uc
end

function centre_xZ!(wc_, w, model)
    (; mgr, domain) = model
    wc = similar!(wc_, w, xdim(domain), zdim(domain))
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange                      
            wc[i, j] = avg_z(w, i, j)
        end       
    end
    return wc
end


function ae_buoyancy!(b_, (; consvar, comp), model, buoyancy_scheme)
    b = similar!(b_, consvar)
    @with mgr, let (irange, jrange) = (axes(b, 1), axes(b, 2))
        @vec for i in irange, j in jrange
            b[i, j] = buoyancy(model, buoyancy_scheme, consvar, comp, i, j)
        end
    end
    return b
end

function buoyancy_freq!(N2_, (bsorted, perm, z_star), model)
    (; domain, inv_dz) = model
    Mx, Mz = dims(domain)

    N2 = similar!(N2_, z_star)

    for i in 2:length(bsorted)-1
        i_field = perm[i]
        @info i
        @info i_field
        N2[i_field] = 0.5 * Mx * (bsorted[i+1] - bsorted[i-1]) * inv_dz
    end
    N2[perm[1]] = 0.5 * Mx * (bsorted[2] - bsorted[1]) * inv_dz
    N2[perm[end]] = 0.5 * Mx * (bsorted[end] - bsorted[end-1]) * inv_dz
    return N2    
end

# rke = (<u>^2 + <w>^2) / 2
function rke!(rke_, (ua, wa))
    rke = similar!(rke_, ua)
    @inbounds for i in axes(ua, 1)
        rke[i] = 0.5 * ( ua[i]^2 + wa[i]^2 )
    end
    return rke
end

# kep = (u'^2 + w'^2) / 2
function kep!(kep_, (up, wp), model)
    kep = similar!(kep_, up)
    @with model.mgr, let (irange, jrange) = (axes(kep, 1), axes(kep, 2))
        @vec for i in irange, j in jrange
            kep[i, j] = 0.5 * ( up[i, j]^2 + wp[i, j]^2 )
        end
    end
    return kep
end

function grad!((∂f∂x_, ∂f∂z_, ∂f∂x_c_, ∂f∂z_c_, absgradf_), f, state, model)
    (; mgr, domain, inv_dx, inv_dz) = model

    ∂f∂x = similar!(∂f∂x_, state.u)
    ∂f∂z = similar!(∂f∂z_, state.w)
    ∂f∂x_c = similar!(∂f∂x_c_, state.m, size(state.m)[1:end-1]...)
    ∂f∂z_c = similar!(∂f∂z_c_, state.m, size(state.m)[1:end-1]...)
    absgradf = similar!(absgradf_, state.m, size(state.m)[1:end-1]...)
    
    # horizontal derivative
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(∂f∂x, 2))
        @vec for I in irange, j in jrange
            ∂f∂x[I, j] = inv_dx * dif_X(f, I, j)
        end
    end
    periodize!(model, ∂f∂x)

    # vertical derivative
    @with mgr, let (irange, jrange) = (axes(∂f∂z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            ∂f∂z[i, J] = inv_dz * dif_Z(f, i, J)
        end
    end

    # boundary: set gradient to zero
    ∂f∂z[:, 1] .= 0.
    ∂f∂z[:, end] .= 0.

    # interpolation onto cell centres
    @with mgr, let (irange, jrange) = (xrange(domain), axes(∂f∂x_c, 2))
        @vec for i in irange, j in jrange
            ∂f∂x_c[i, j] = avg_x(∂f∂x, i, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(∂f∂z_c, 1), zrange(domain))
        @vec for i in irange, j in jrange
            ∂f∂z_c[i, j] = avg_z(∂f∂z, i, j)
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            absgradf[i, j] = sqrt(avg_x(abs2, ∂f∂x, i, j) + avg_z(abs2, ∂f∂z, i, j))
            # absgradf[i, j] = sqrt(avg_x(∂f∂x, i, j)^2 + avg_z(∂f∂z, i, j)^2)
        end
    end

    return ∂f∂x, ∂f∂z, ∂f∂x_c, ∂f∂z_c, absgradf
end


# # available energetics tendencies: SortedLorenzState
# function ae_tendencies!(energetics_scratch, model::AN2D, ref_state_type::consvarSorted, state, dynamics_scratch)
#     (; ape, bse, W) = energetics_scratch.available_energies

#     # energies
#     ape = available_potential_energy!(ape, state, model)
#     bse = background_static_energy!(bse, state, model)
#     W   = background_pressure_work!(W, state, model)
#     periodize!(model, (ape, bse, W))

#     energetics_scratch = (;
#         available_energies = (; ape, bse, W),
#     )
#     return energetics_scratch
# end


function consvar_sorted_reference_state!((mass_), state, model::AN2D{F}, dynamics_scratch) where {F}
    (; mgr, reference, space, domain, inv_dx, inv_dz, dx, dz, fluid) = model
    (; ρ_R, p_R, ρ_func) = reference

    Mx, Mz = intdims(domain)

    (; consvar) = dynamics_scratch.thermo
    
    dz_star_0 = (dx / M) * ones(M*N)

    mass = similar!(mass_, consvar)

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            # calculate mass of each parcel in original position
            mass[i, j] = dx * dz * ρ_func(zpoint(domain, dz, j))    
        end
    end

    consvar_sort = similar!(consvar_sort_, consvar, Mx*Mz)

    # reshape consvar into vector
    s_sort  .= reshape(consvar, (M*N,))
    S       .= reshape(S_slice, (M*N,))

    # sort s
    perm        .= sortperm(s)
    inv_perm    .= invperm(perm)
    s           .= s[perm]
    S           .= S[perm]

    # sort mass
    mass_sorted .= mass_in_situ[perm]

    # mass conservation to find new dz_star, z_star
    dz_star .= dz_star_0
    mass_cons!(dz_star, z_star, z_star_edge, ρ_R_z_star, ρ_R_z_star_edge, ρ_R_func, mass_sorted, M*dx, 10)

    # hydrostatic balance to find p_R(z_star)
    hydrostatic_pressure!(p_R_z_star, p_R_z_star_edge, ρ_R_z_star, p_ref_top, dz_star, g, M*N)

    # buoyancy at z_star
    buoy!(b_star, ρ_R_z_star, p_R_z_star, s, S, fluid, g)

    # solve the level of neutral buoyancy equation to find P_star = (p_star - p_R) / ρ_R
    hydrostatic_pressure!(P_star, P_star_edge, b_star, 0, dz_star, -1, length(P_star))

    # compute reference pressure as a function of z
    reference_pressure!(P_star_z, b_star_z, z, P_star_edge, z_star_edge)

    # reshape vectors
    z_star          .= z_star[inv_perm]
    P_star          .= P_star[inv_perm]
    b_star          .= b_star[inv_perm]
    p_R_z_star      .= p_R_z_star[inv_perm]

    z_star_slice        .= reshape(z_star, (M, N))
    P_star_slice        .= reshape(P_star, (M, N))
    b_star_slice        .= reshape(b_star, (M, N))
    p_R_z_star_slice    .= reshape(p_R_z_star, (M, N))

    # evaluated at parcel position in ref state
    ds["z_star"][:, :, t]   = z_star_slice
    ds["P_star"][:, :, t]   = P_star_slice
    ds["b_star"][:, :, t]   = b_star_slice
    ds["p_R_z_star"][:, :, t] = p_R_z_star_slice

    # evaluated at parcel position in in situ state
    ds["P_star_z"][:, :, t] = P_star_z
    ds["b_star_z"][:, :, t] = b_star_z
end

## LOOP

# available energetics loop
# energetics loop
function available_energetics!(ds, keys, model::AN2D{F}, ref_state_type, params) where {F}
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

## DIAGNOSTICS

# diagnostics for energetics
ae_diagnostics() = CookBook(;
    b = ae_scratch -> ae_scratch.fields.b,
    z_star = ae_scratch -> ae_scratch.fields.z_star,
    N2 = ae_scratch -> ae_scratch.fields.N2,
    # N2c = ae_scratch -> ae_scratch.reference_state.N2c,
    zeta = ae_scratch -> ae_scratch.reference_state.zeta,
    zeta_c = ae_scratch -> ae_scratch.reference_state.zeta_c,
    b_star = ae_scratch -> ae_scratch.averages.b_star,
    ua = ae_scratch -> ae_scratch.averages.ua,
    wa = ae_scratch -> ae_scratch.averages.wa,
    uc = ae_scratch -> ae_scratch.fields.uc,
    wc = ae_scratch -> ae_scratch.fields.wc,
    up = ae_scratch -> ae_scratch.fields.up,
    wp = ae_scratch -> ae_scratch.fields.wp,
    map_field = ae_scratch -> ae_scratch.fields.map_field,
    kep = ae_scratch -> ae_scratch.fields.kep,
    rke = ae_scratch -> ae_scratch.averages.rke,
    tke = ae_scratch -> ae_scratch.averages.tke,
    bw = ae_scratch -> ae_scratch.averages.bw,
    eps_k = ae_scratch -> ae_scratch.averages.eps_k,
    ∂z_star∂x = ae_scratch -> ae_scratch.fields.∂z_star∂x, 
    ∂z_star∂z = ae_scratch -> ae_scratch.fields.∂z_star∂z, 
    ∂z_star∂x_c = ae_scratch -> ae_scratch.fields.∂z_star∂x_c, 
    ∂z_star∂z_c = ae_scratch -> ae_scratch.fields.∂z_star∂z_c, 
    grad_z_star = ae_scratch -> ae_scratch.fields.grad_z_star,
    grad_z_star_a = ae_scratch -> ae_scratch.averages.grad_z_star_a
)