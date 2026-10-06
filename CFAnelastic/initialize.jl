# define initial condition types
UWSQ    = NamedTuple{(:u, :w, :s,       :q)}
UWThQ   = NamedTuple{(:u, :w, :theta,   :q)}
UWConsQ = NamedTuple{(:u, :w, :consvar, :q)}
IC = Union{UWSQ, UWThQ, UWConsQ}

# create initial state (m, u, w) from functions or fields
function initialize(model::AN2D{F}, ic::IC) where {F}
    (; reference, domain) = model
    (; u, w, q) = ic

    # allocate empty state
    state = (m = alloc_xz(F, domain, 2), u = alloc_Xz(F, domain), w = alloc_xZ(F, domain))

    # allocate m
    let (irange, jrange) = (axes(state.m, 1), axes(state.m, 2))
        @vec for i in irange, j in jrange
            state.m[i, j, 1] = reference.ρ_R[j] * consvar_ij(model, ic, i, j)
            state.m[i, j, 2] = reference.ρ_R[j] * call_xz(model, q, i, j)
        end
    end    

    # allocate u
    let (irange, jrange) = (axes(state.u, 1), axes(state.u, 2))
        @vec for I in irange, j in jrange
            state.u[I, j] = call_Xz(model, u, I, j)
        end
    end  
    
    # allocate w
    let (irange, jrange) = (axes(state.w, 1), axes(state.w, 2))
        @vec for i in irange, J in jrange
            state.w[i, J] = call_xZ(model, w, i, J)
        end
    end  
    return state
end

# compute consvar at (i, j) from either Cons, Th, or S inputs

function consvar_ij(model, (; u, w, consvar, q)::UWConsQ, i, j)
    return call_xz(model, consvar, i, j)
end
function consvar_ij(model, (; u, w, theta, q, i, j)::UWThQ)
    return fluid(:p, :theta, :q).conservative_variable(p_R[j], call_xz(model, theta, i, j), call_xz(model, q, i, j))
end
function consvar_ij(model, (; u, w, s, q, i, j)::UWSQ)
    return fluid(:p, :s, :q).conservative_variable(p_R[j], call_xz(model, s, i, j), call_xz(model, q, i, j))
end


# call from functions and arrays the same way

@inline call_xz(model, arr::AbstractArray, i, j) = arr[i, j]
@inline call_Xz(model, arr::AbstractArray, I, j) = arr[I, j]
@inline call_xZ(model, arr::AbstractArray, i, J) = arr[i, J]

function call_xz(model, func::Function, i, j)
    (; dx, dz, domain) = model
    x = xpoint(domain, dx, i)
    z = zpoint(domain, dz, j)
    return func(x, z)
end

function call_Xz(model, func::Function, I, j)
    (; dx, dz, domain) = model
    X = Xpoint(domain, dx, I)
    z = zpoint(domain, dz, j)
    return func(X, z)
end

function call_xZ(model, func::Function, i, J)
    (; dx, dz, domain) = model
    x = xpoint(domain, dx, i)
    Z = Zpoint(domain, dz, J)
    return func(x, Z)
end

# ## INITIALISE STATE
# function initialize(model::AN2D{F}, params) where F
#     (; domain) = model
#     (; rest_variable, experiment) = params

#     # allocate empty state
#     state = (m = alloc_xz(F, domain, 2), u = alloc_Xz(F, domain), w = alloc_xZ(F, domain))

#     # compute rest profile from chosen variable
#     if rest_variable == "θ" || rest_variable == "theta" || rest_variable == :θ || rest_variable == :theta
#         rest_θ!(state, model, params)
#     else
#         error("Unknown rest variable: $rest_variable")
#     end

#     # add experiment layer
#     if experiment == "shear_sin" || experiment == "shear"
#         # add velocity shear
#         shear!(state, model, params)
#         # add sin perturbation
#         sin_pert!(state, model, params)
#     elseif experiment == "shear_rand"
#         # add velocity shear
#         shear!(state, model, params)
#         # add random kick
#         kick!(state, model, params)
#     elseif experiment == "jet_rand"
#         # add jet profile
#         jet!(state, model, params)
#         # add sin perturbation
#         kick!(state, model, params)
#     elseif experiment == "jet_sin"
#         # add jet profile
#         jet!(state, model, params)
#         # add sin perturbation
#         sin_pert!(state, model, params)   
#     elseif experiment == "jet_kick"
#         # add jet profile
#         jet!(state, model, params)
#         # add sin perturbation
#         kick!(state, model, params)
#     elseif experiment == "random"
#         kick!(state, model, params)
#     elseif experiment == "none"
#     else
#         error("Unknown experiment: $experiment")
#     end
#     return state
# end

# ## REST LAYER

# function rest_θ!((; m, u, w), model::AN2D, params)
#     (; domain, space, fluid, reference, dz) = model
#     (; Lz) = space
#     (; p_R, ρ_R) = reference
#     (; rest_profile) = params

#     # vertical coordinate vector (incl halo)
#     z = zgrid(domain, dz)

#     # choose initial temperature profile
#     if rest_profile == "linear"
#         (; T₀, ΔT, q₀, Δq) = params
#         θ = linear_profile(z, T₀, ΔT, Lz)
#         q = linear_profile(z, q₀, Δq, Lz)
#     elseif rest_profile == "tanh"
#         (; T₀, ΔT, δ_T, q₀, Δq, δ_q) = params
#         θ = tanh_profile(z, T₀, ΔT, Lz, δ_T)
#         q = tanh_profile(z, q₀, Δq, Lz, δ_q)
#     elseif rest_profile == "constant" || rest_profile == "const"
#         (; T₀, q₀) = params
#         θ = const_profile(z, T₀)
#         q = const_profile(z, q₀)
#     elseif rest_profile == "sin"
#         (; T₀, ΔT, q₀, Δq) = params
#         θ = sin_profile(z, T₀, ΔT, Lz)
#         q = sin_profile(z, q₀, Δq, Lz)
#     else
#         error("Unknown temperature profile: $rest_profile")
#     end

#     # construct ρ_R * s and ρ_R * q 
#     let (irange, jrange) = (axes(m, 1), axes(m, 2))
#         @vec for i in irange, j in jrange
#             m[i, j, 1] = ρ_R[j] * fluid(:p, :theta, :q).conservative_variable(p_R[j], θ[j], q[j])
#             m[i, j, 2] = ρ_R[j] * q[j]
#         end
#     end
#     # set velocity to zero
#     u[:, :] .= 0
#     w[:, :] .= 0
    
#     # periodize m
#     periodize!(model, (@views m[:, :, 1], @views m[:, :, 2]))
    
#     return (; m, u, w)
# end

# ## EXPERIMENT LAYER

# # shear flow
# function shear!((; m, u, w), model, params)
#     (; mgr, domain, space, dz) = model
#     (; Lz) = space
#     (; δ_u, Δu) = params
#     # add on horizontal shear flow
#     @with mgr, let (irange, jrange) = (axes(u, 1), axes(u, 2))
#         @vec for I in irange, j in jrange        
#             z = zpoint(domain, dz, j)
#             u[I, j] += Δu * tanh((z .- Lz/2) ./ (δ_u / 2))
#         end
#     end
#     return (; m, u, w)
# end

# # jet
# function jet!((; m, u, w), model, params)
#     (; mgr, domain, space, dz) = model
#     (; Lz) = space
#     (; δ_u, Δu) = params
#     # add on horizontal shear flow
#     @with mgr, let (irange, jrange) = (axes(u, 1), axes(u, 2))
#         @vec for I in irange, j in jrange        
#             z = zpoint(domain, dz, j)
#             u[I, j] += Δu * exp(-(z .- Lz/2)^2 ./ (δ_u / 2)^2)
#         end
#     end
#     return (; m, u, w)
# end

# function sin_pert!((; m, u, w), model::AN2D{F}, params) where F
#     (; mgr, domain, space, reference, dx, dz, inv_dx, inv_dz) = model
#     (; Lx, Lz) = space
#     (; ρ_R, ρ_R_J, ρtop) = reference
#     (; kick_size, Δu, δ_u, λ_pert) = params
    
#     ψ = alloc_XZ(F, domain)
#     let (irange, jrange) = (axes(ψ, 1), axes(ψ, 2))
#         @vec for I in irange, J in jrange  
#             # shift coordinates to centre (Lx/2, Lz/2)
#             x = Xpoint(domain, dx, I) - Lx/2
#             z = Zpoint(domain, dz, J) - Lz/2
#             ψ[I, J] = -ρtop * exp(- z^2 / (δ_u / 2)^2) * kick_size * Δu * (λ_pert / (2*π)) * cos(2*π*x / λ_pert)
#         end
#     end
#     periodize!(model, ψ)

#     # add u, w perturbations from streamfunction
#     @with mgr, let (irange, jrange) = (axes(u, 1), zrange(domain))
#         @vec for I in irange, j in jrange  
#             u[I, j] += inv(ρ_R[j]) * dif_z(ψ, I, j) * inv_dz
#         end
#     end
#     @with mgr, let (irange, jrange) = (xrange(domain), axes(w, 2))
#         @vec for i in irange, J in jrange  
#             w[i, J] -= inv(ρ_R_J[J]) * dif_x(ψ, i, J) * inv_dx
#         end
#     end
#     periodize!(model, (u, w))

#     return (; m, u, w)
# end

# # random perturbation of m = [s, q], u, w
# function kick!((; m, u, w), model::AN2D{F}, params) where F
#     (; mgr, domain, reference, inv_dz, inv_dx, dx, dz) = model 
#     (; p_R, ρ_R, ρ_R_J, ρtop, v_R) = reference
#     (; kick_size, Δu) = params

#     Mx, Mz = dims(domain)
#     Hx, Hz = halo_size(domain)

#     # random streamfunction (with two extra vertical ghost cells)
#     ψ = alloc_XZ(F, domain)
#     let (irange, jrange) = (axes(ψ, 1), axes(ψ, 2))
#         @vec for I in irange, J in jrange  
#             ψ[I, J] = rand() * ρtop * Δu * kick_size * 0.5 * (dx + dz)
#         end
#     end
#     # set ψ on boundaries to ensure that w(z=0) = w(z=Lz) = 0
#     ψ[:, Hz+1]     .= rand() * ρtop * Δu * kick_size * 0.5 * (dx + dz)
#     ψ[:, Hz+Mz+1]  .= rand() * ρtop * Δu * kick_size * 0.5 * (dx + dz)

#     periodize!(model, ψ)
    
#     # add u, w perturbations from streamfunction
#     let (irange, jrange) = (axes(u, 1), zrange(domain))
#         for I in irange, j in jrange  
#             u[I, j] += inv(ρ_R[j]) * dif_z(ψ, I, j) * inv_dz
#         end
#     end
#     let (irange, jrange) = (xrange(domain), axes(w, 2))
#         for i in irange, J in jrange  
#             w[i, J] -= inv(ρ_R_J[J]) * dif_x(ψ, i, J) * inv_dx
#         end
#     end
#     periodize!(model, (u, w))
    
#     # apply boundary conditions to state
#     cons, comp, T = cons_comp_T!(void, model, (; m, u, w))
#     (; m, u, w) = state_bc!(model, (; m, u, w), (cons, comp, T))

#     # compute max/min of conservative variable and composition 
#     Δcons = maximum(cons) - minimum(cons)
#     Δcomp = maximum(comp) - minimum(comp)

#     # add on perturbed consvar
#     let (irange, jrange) = (axes(m, 1), axes(m, 2))
#         for i in irange, j in jrange 
#             m[i, j, 1] += ρ_R[j] * rand() * Δcons * kick_size
#             m[i, j, 2] += ρ_R[j] * rand() * Δcomp * kick_size
#         end
#     end
#     # apply boundary conditions again to state
#     cons, comp, T = cons_comp_T!(void, model, (; m, u, w))
#     (; m, u, w) = state_bc!(model, (; m, u, w), (cons, comp, T))
#     return (; m, u, w)
# end

# # # hot bubble (function of θ)
# # function hot_bubble(x, z, (; m, u, w), (; environment, fluid, reference), params, pTv)
# #     (; M, N) = environment
# #     (; ρ_R, p_R) = reference
# #     (; Δθ_bubble, r_bubble, z_bubble) = params

# #     # compute potential temperature and entropy of bubble
# #     for i in 1:M+1, j in 1:N
# #         r = sqrt(x[i]^2 + (z[j]-z_bubble)^2)
# #         if r < r_bubble
# #             θ = potential_temperature(fluid, (; p = 0., consvar = m[i, j, 1] / ρ_R[j]), q = m[i, j, 2] / ρ_R[j]) + Δθ_bubble * (cos(π * r / (2 * r_bubble)))^2
# #             # add on σ = ρ_R * s(p_R, T)
# #             m[i, j, 1] = ρ_R[j] * conservative_variable(fluid, (; p = p_R[j], theta = θ, q = m[i, j, 2] / ρ_R[j]))
# #         end
# #     end
# #     return (; m, u, w)
# # end

# ## VERTICAL PROFILES
# const_profile(z, f0)            = f0 .* ones(size(z))
# linear_profile(z, f0, Δf, Lz)    = f0 .+ Δf * (z .- Lz/2) / Lz
# tanh_profile(z, f0, Δf, Lz, δz)  = f0 .+ 0.5 * Δf * tanh.( (z .- Lz/2) / (0.5 * δz)) 
# sin_profile(z, f0, Δf, Lz)       = f0 .+ 0.5 * Δf * sin.( π * (z .- Lz/2) / Lz)

# ## USEFUL FUNCTIONS
# function kin_visc(ρ, dyn_visc)
#     return dyn_visc / ρ
# end

# function Re(ρ, U, ℓ, dyn_visc)
#     return ρ * U * ℓ / dyn_visc
# end

# function Mach(U, c)
#     return U / c
# end

# function Fr(U, g, ℓ)
#     return U / sqrt(g * ℓ)
# end

# function Pr(ρ, k_T, dyn_visc)
#     return dyn_visc / ( k_T * ρ )
# end

# function N0(ΔT, δ_T, g, T₀)
#     return ΔT * g / (δ_T * T₀)
# end