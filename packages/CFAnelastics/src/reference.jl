## ANELASTIC REFERENCE STATE

"""
    reference = AnelasticReference(domain, space, ρ; ptop)

Reference state of the anelastic model on the grid of `domain` spanning `space`, along the full z axis (halo
included): the density `ρ_R` and the hydrostatic pressure `p_R` (∂p_R/∂z = -ρ_R g) at cell centres, with pressure
`ptop` [Pa] at the top of the domain. The density `ρ` [kg m⁻³] is a number, a function ρ(z) of the height z above the
bottom, or a vector of its values in the Mz cells, from bottom to top (extended by its end values into the halo).
Face values and the specific volume are computed where needed: ρ_R on w-faces is the average of the two cells.

To build the density from a fluid state instead, see `hydrostatic_reference`.
"""
struct AnelasticReference{F, Vec}
    ptop::F
    ρ_R::Vec
    p_R::Vec
end
function AnelasticReference(domain, space, ρ; ptop)
    (; Lz, g) = space
    Mz, Hz = dims(domain)[end], halo_size(domain)[end]
    dz = Lz / Mz
    check_length(ρ, Mz)
    ρ_R = [cell_value(ρ, j - Hz, zpoint(domain, dz, j)) for j in 1:Mz+2Hz]
    # hydrostatic pressure, from the pressure above the halo
    p_R = hydrostatic_pressure(ρ_R, ptop - sum(ρ_R[Hz+Mz+1:end]) * g * dz, dz, g)
    return AnelasticReference(ptop, ρ_R, p_R)
end

# value of a profile in cell k (k = 1, …, Mz in the interior) at height z: a number, a function of z, or a vector of
# the Mz cell values, extended by its end values
cell_value(f::Number, k, z) = f
cell_value(f::Function, k, z) = f(z)
cell_value(f::AbstractVector, k, z) = f[clamp(k, 1, length(f))]

check_length(ρ, Mz) = nothing
check_length(ρ::AbstractVector, Mz) = length(ρ) == Mz || error("AnelasticReference: $(length(ρ)) density values for $Mz cells.")

# hydrostatic pressure at cell centres, from the pressure ptop above the last cell: p = p_upper + ρ g Δz/2
function hydrostatic_pressure(ρ, ptop, dz, g)
    p = similar(ρ)
    p_upper = ptop
    for j in length(ρ):-1:1
        p[j] = p_upper + ρ[j] * g * dz / 2
        p_upper += ρ[j] * g * dz
    end
    return p
end