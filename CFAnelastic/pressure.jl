## PRESSURE SOLVER(S)

# generic pressure solver taking momentum tendencies dU, dW and returning modified pressure φ
function pressure! end

## Mixed Fourier - Finite Difference Poisson solver
# - for 2D anelastic equations with Neumann condition ∂p/∂z = 0 on top/bottom, and Periodic BCs in x
# - uses a centred finite difference in z and a Fourier spectral method in x
#   to solve ∇·(ρ_R ∇φ) = ∇·(ρ_R dU_) = R for φ = (p - p_R)/ρ_R
# - first wavenumber k=1 is solved as an ODE with BC φ = 0 at top and bottom

function pressure!((φ, poisson_scratch), model, state, (dU, dW))
    (R, R_fft, W_fft, β, γ, D, cc, φ, φ_fft, R_fft_plan, W_fft_plan, φ_ifft_plan) = poisson_scratch

    R, R_fft, W_fft, R_fft_plan, W_fft_plan = poisson_rhs!((R, R_fft, W_fft, R_fft_plan, W_fft_plan), model, state, (dU, dW))
    β, γ, D = poisson_coeffs!((β, γ, D), model)
    φ, φ_fft, φ_ifft_plan, cc = poisson_solve!((φ, φ_fft, φ_ifft_plan, cc), model, state, (β, γ, R_fft, W_fft))
    
    periodize!(model, φ)

    return φ, (R, R_fft, W_fft, β, γ, D, cc, φ, φ_fft, R_fft_plan, W_fft_plan, φ_ifft_plan)
end

# take Fourier transforms in the horizontal to construct the rhs of finite difference system
function poisson_rhs!((R_, R_fft_, W_fft_, R_fft_plan_, W_fft_plan_), model::AN2D, (; m), (dU, dW))
    (; mgr, domain, reference, inv_dx, inv_dz) = model
    (; ρ_R, ρ_R_J) = reference
    Mx, Mz = dims(domain)
    Hx, Hz = halo_size(domain)

    R       = similar!(R_, m, size(m)[1:2]...)
    R_fft   = similar!(R_fft_, Array{ComplexF64}, Mx, Mz)
    W_fft   = similar!(W_fft_, Array{ComplexF64}, Mx, Mz+1+2Hz)

    # compute R = ∇·(ρ_R * dU, ρ_R* dW)
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            R[i, j] = ρ_R[j] * dif_x(dU, i, j) * inv_dx + dif_z(ρ_R_J, dW, i, j) * inv_dz
        end
    end
    periodize!(model, R)

    # write input into the fft arrays
    @with mgr, let (irange, jrange) = (1:Mx, 1:Mz)
        for i in irange, j in jrange
            R_fft[i, j] = R[i+Hx, j+Hz]
        end
    end
    @with mgr, let (irange, jrange) = (1:Mx, Hz+1:Hz+Mz+1)
        for i in irange, j in jrange
            W_fft[i, j] = dW[i+Hx, j]
        end
    end
    # create FFT plans
    R_fft_plan, W_fft_plan = fft_plans!(R_fft_plan_, W_fft_plan_, (R_fft, W_fft))

    # apply FFT
    mul!(R_fft, R_fft_plan, R_fft)
    mul!(W_fft, W_fft_plan, W_fft)

    # apply boundary condition to R_fft
    # j = 1
    @with mgr, let krange = 1:Mx
        for k in krange
            R_fft[k, 1] +=  ρ_R_J[Hz+1] * W_fft[k, Hz+1] * inv_dz
        end
    end
    @with mgr, let krange = 1:Mx
        for k in krange
            R_fft[k, end] -=  ρ_R_J[Hz+Mz+1] * W_fft[k, Hz+Mz+1] * inv_dz
        end
    end

    return R, R_fft, W_fft, R_fft_plan, W_fft_plan
end

# solve the vertical finite difference system for pressure
function poisson_solve!((φ_, φ_fft_, φ_ifft_plan_, cc_), model::AN2D, (; m), (β, γ, R_fft, W_fft))
    (; mgr, domain, dz) = model
    Mx, Mz = dims(domain)
    Hx, Hz = halo_size(domain)

    φ       = similar!(φ_, m, size(m)[1:2]...)                  # modified pressure φ = (p - p_R)/ρ_R
    φ_fft   = similar!(φ_fft_, Array{ComplexF64}, Mx, Mz+2Hz)   # FFT of φ with vertical halo
    cc      = similar!(cc_, γ)                                  # scratch for tridiagonal solver

    # k=1: system reduces to δᴶφ⁰ = Ŵ⁰ᴶ
    φ_fft_slice = @view φ_fft[1, :]
    W_fft_slice = @view W_fft[1, :]

    # # solve with BC φ⁰ = 0 at top => φₙ₊₁ + φₙ = 0 => φₙ = -φₙ₊₁ => -2φₙ = dz * Ŵ⁰ₙ₊₁
    # val = -0.5 * dz * W_fft_slice[Hz+Mz+1]
    # φ_fft_slice[Hz+Mz+1] = -val
    # φ_fft_slice[Hz+Mz] = val
    
    # # integrate top down
    # @inbounds @simd for j = Hz+Mz-1:-1:Hz
    #     val -= W_fft_slice[j+1] * dz
    #     φ_fft_slice[j] = val
    # end

    # integrate bottom up with BC φ⁰ = 0 at bottom => φ₀ + φ₁ = 0 => φ₀ = -φ₁
    val = 0.5 * dz * W_fft_slice[Hz+1]
    φ_fft_slice[Hz+1] = val
    φ_fft_slice[Hz] = -val
    @inbounds @simd for j = Hz+2:Mz+Hz
        val += W_fft_slice[j] * dz
        φ_fft_slice[j] = val
    end

    # k=2,...Mx: solve centered finite difference symmetric tridiagonal system

    # solve γⱼ₋₁ ̂φₖⱼ₋₁ + βₖⱼ ̂φₖⱼ + γⱼ ̂φₖⱼ₊₁ = R̂ₖⱼ
    @with mgr, let krange = 2:Mx
        for k in krange
            @views thomas_sym!(β[k, :], γ, cc, φ_fft[k, Hz+1:Hz+Mz], R_fft[k, :], Mz)
            φ_fft[k, Hz] = -φ_fft[k, Hz+1]   # bottom BC: φ = 0 => φ₀ = -φ₁
            φ_fft[k, Hz+Mz+1] = -φ_fft[k, Hz+Mz]   # top BC: φ = 0 => φₙ₊₁ = -φₙ
        end
    end

    # create IFFT plan
    φ_ifft_plan = ifft_plans!(φ_ifft_plan_, φ_fft)

    # apply IFFT
    mul!(φ_fft, φ_ifft_plan, φ_fft)

    # recover φ = (p - p_R)/ρ_R (taking halo into account)
    @with mgr, let (irange, jrange) = (1:Mx, 1:Mz+2Hz)
        for i in irange, j in jrange
            φ[i+Hx, j] = real(φ_fft[i, j])
        end
    end
    return φ, φ_fft, φ_ifft_plan, cc
end

# symmetric tridiagonal Thomas algorithm
function thomas_sym!(b, c, cc, d, rhs, N)
    # forward sweep
    d[1]    = rhs[1] / b[1]
    cc[1]   = c[1] / b[1]
    @inbounds for j in 2:N-1
        denom = b[j] - c[j-1] * cc[j-1]
        cc[j] = c[j] / denom
        d[j] = (rhs[j] - c[j-1] * d[j-1]) / denom
    end
    d[N] = (rhs[N] - c[N-1] * d[N-1]) / (b[N] - c[N-1] * cc[N-1])
    # backward sweep
    @inbounds for j in N-1:-1:1
        d[j] -= cc[j] * d[j+1]
    end
    return d
end

# FFT plans
fft_plans!(R_fft_plan_, W_fft_plan_, (R_fft, W_fft)) = R_fft_plan_, W_fft_plan_
fft_plans!(R_fft_plan_::Void, W_fft_plan_::Void, (R_fft, W_fft)) = plan_fft!(R_fft, 1), plan_fft!(W_fft, 1) 

ifft_plans!(φ_ifft_plan_, φ_fft) = φ_ifft_plan_
ifft_plans!(φ_ifft_plan_::Void, φ_fft) = plan_ifft!(φ_fft, 1)

# Poisson solver pre-computations
diag!(β_, model::AN2D, D) = β_
off_diag!(γ_, model::AN2D) = γ_
D!(D_, model::AN2D) = D_

poisson_coeffs!((β_, γ_, D_), model::AN2D) = (β_, γ_, D_)

# finite difference Poisson coefficients
# βₖⱼ = - ρ_Rⱼ * Dₖ - (ρ_Rⱼ₊₁ + 2ρ_Rⱼ + ρ_Rⱼ₋₁) / (2Δz)² 
# γⱼ = (ρ_Rⱼ₊₁ + ρ_Rⱼ) / (2Δz)²
function poisson_coeffs!((β_, γ_, D_)::Tuple{Void, Void, Void}, model::AN2D)
    (; mgr, domain, reference, inv_dz) = model
    (; ρ_R) = reference
    Mx, Mz = dims(domain)
    Hx, Hz = halo_size(domain)
    β = similar!(β_, ρ_R, Mx, Mz)
    γ = similar!(γ_, ρ_R, Mz-1)

    D = D!(D_, model)

    # diagonal component of finite difference system
    @with mgr, let (krange, jrange) = (1:Mx, 2:Mz-1)
        @vec for k in krange, j in jrange
            β[k, j] = - ρ_R[j+Hz] * D[k] - ( ρ_R[j+Hz+1] + 2*ρ_R[j+Hz] + ρ_R[j+Hz-1] ) * 0.5 * inv_dz^2
        end
    end

    # top / bottom boundaries
    @with mgr, let krange = 1:Mx
        @vec for k in krange
            β[k, 1] = - D[k] * ρ_R[1+Hz] - ( ρ_R[1+Hz] + ρ_R[2+Hz] ) * 0.5 * inv_dz^2
            β[k, Mz] = - D[k] * ρ_R[Mz+Hz] - ( ρ_R[Mz-1+Hz] + ρ_R[Mz+Hz] ) * 0.5 * inv_dz^2
        end
    end

    # off-diagonal components of finite difference system
    @with mgr, let jrange = 1:Mz-1
        @vec for j in jrange
            γ[j] = ( ρ_R[j+1+Hz] + ρ_R[j+Hz] ) * 0.5 * inv_dz^2
        end
    end

    return β, γ, D
end

# Dₖ = (2 sin(π (k-1) / Mx) / Δx)²
function D!(D_::Void, model::AN2D)
    (; mgr, domain, reference, inv_dx) = model
    (; ρ_R) = reference
    Mx, _ = dims(domain)
    D = similar!(D_, ρ_R, Mx)
    @with mgr, let krange = 1:Mx
        for k in krange
            D[k] = 4*(sin(π*(k-1) / Mx))^2 * inv_dx^2
        end
    end
    return D
end