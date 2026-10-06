"""
    AnelasticReference{F, R}
A struct to hold reference state profiles for anelastic simulations.

# Fields
- `ρtop`: Reference density at the top of the domain.
- `ptop`: Reference pressure at the top of the domain.
- `ρ_R`: Reference density profile at cell centers.
- `ρ_R_J`: Reference density profile averaged to grid edges.
- `v_R`: Specific volume profile (inverse of density).
- `p_R`: Hydrostatic pressure profile at cell centers.
"""
struct AnelasticReference{F, Vec}
    ptop::F
    ρ_R::Vec
    ρ_R_J::Vec
    v_R::Vec
    p_R::Vec
    ρ_func::Function
end

"""
    AnelasticReference(domain, space, density_profile; ρtop, ptop)
Construct an `AnelasticReference` object given a density profile function.

# Arguments
- `domain`: The computational domain.
- `space`: The spatial configuration, containing domain dimensions and gravity.
- `density_profile`: The vertical density profile function with an argument `z` representing height, with 0 corresponding to the top of the domain.
- `ρtop`: Reference density at the top of the domain.
- `ptop`: Reference pressure at the top of the domain.

# Returns
- An `AnelasticReference` object containing the reference state profiles.
"""
function AnelasticReference(; domain::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}, space, density_profile, ptop) where {Mx, My, Mz, Hx, Hy, Hz}
    (; Lz, g) = space
    dz = Lz / Mz

    # convert to function with z=0 at bottom
    ρ_func(z) = density_profile(Lz - z)

    # density profile with halo + one extra cell either side
    ρ_R_big = [ρ_func(zpoint(domain, dz, j)) for j in 0:Mz+2Hz+1]

    # density profile with halo
    ρ_R     = ρ_R_big[2:end-1]

    # specific volume profile
    v_R     = inv.(ρ_R)

    # density profile averaged to grid edge
    ρ_R_J   = [avg_Z(ρ_R_big, J) for J in 2:Mz+2Hz+2]
    
    # pressure above halo
    phalo = ptop - sum(ρ_R[Hz+Mz+1:Mz+2Hz]) * g * dz

    # hydrostatic pressure profile
    p_R     = hydrostatic_pressure(ρ_R, phalo, dz, g)

    return AnelasticReference(ptop, ρ_R, ρ_R_J, v_R, p_R, ρ_func)
end

"""
    hydrostatic_pressure(ρ, ptop, dz, g)
Compute the hydrostatic pressure profile given a density profile.

# Arguments
- `ρ`: Density profile at cell centers.
- `ptop`: Pressure at the top of the domain.
- `dz`: Grid spacing in the vertical direction.
- `g`: Gravitational acceleration.

# Returns
- `p`: Hydrostatic pressure profile at cell centers.
"""
function hydrostatic_pressure(ρ, ptop, dz, g)
    N = length(ρ)
    p = similar(ρ)

    # pressure on upper edge
    pu = ptop
    for j in N:-1:1
        # pressure on lower edge
        pl = pu + ρ[j] * g * dz

        # average pressure at cell center
        p[j] = 0.5 * (pl + pu)

        # update pressure on upper edge
        pu = pl
    end
    return p
end