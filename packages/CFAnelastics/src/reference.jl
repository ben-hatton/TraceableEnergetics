# Anelastic reference state

"""
    reference = AnelasticReference(domain, space, ρ; ptop)

Reference state of the anelastic model on the grid of `domain` spanning `space`, along the full z axis (halo
included): the density `ρ_R` and the hydrostatic pressure `p_R` (∂p_R/∂z = -ρ_R g) at cell centers, with pressure
`ptop` [Pa] at the top of the domain. The density `ρ` [kg m⁻³] is a number, a function ρ(z) of the height z above the
bottom, or a vector of its values in the Mz cells, from bottom to top (extended by its end values into the halo).
Face values and the specific volume are computed where needed: ρ_R on w-faces is the average of the two cells.
"""
struct AnelasticReference{F, Vec}
    "pressure at the top of the domain [Pa]"
    ptop::F
    "reference density at the cell centers of the full z axis, halo included [kg m⁻³]"
    ρ_R::Vec
    "hydrostatic reference pressure at the cell centers of the full z axis, halo included [Pa]"
    p_R::Vec
end
function AnelasticReference(domain, space, ρ; ptop)
    (; Lz, g) = space
    Mz, Hz = dims(domain)[end], halo_size(domain)[end]
    dz = Lz / Mz
    check_length(ρ, Mz)
    ρ_R = [cell_value(ρ, j - Hz, cellpoint(domain, dz, 2, j)) for j in 1:Mz+2Hz]
    # hydrostatic pressure, from the pressure above the halo
    p_R = hydrostatic_pressure(ρ_R, ptop - sum(ρ_R[Hz+Mz+1:end]) * g * dz, dz, g)
    return AnelasticReference(ptop, ρ_R, p_R)
end

"""
    value = cell_value(f, k, z)

Value of a profile in cell k (k = 1, …, Mz in the interior) at height z: a number, a function of z, or a vector of
the Mz cell values, extended by its end values.
"""
cell_value(f::Number, k, z) = f
cell_value(f, k, z) = f(z)
cell_value(f::AbstractVector, k, z) = f[clamp(k, 1, length(f))]

"""
    check_length(ρ, Mz)

Throw an error if the density profile `ρ` is a vector without one value per interior cell.
"""
check_length(ρ, Mz) = nothing
check_length(ρ::AbstractVector, Mz) = length(ρ) == Mz || error("AnelasticReference: $(length(ρ)) density values for $Mz cells.")

"""
    p = hydrostatic_pressure(ρ, ptop, dz, g)

Hydrostatic pressure at the cell centers, from the pressure `ptop` above the last cell: p = p_upper + ρ g Δz/2, with
p_upper the pressure on the upper face of the cell.
"""
function hydrostatic_pressure(ρ, ptop, dz, g)
    p = similar(ρ)
    p_upper = ptop
    for j in length(ρ):-1:1
        p[j] = p_upper + ρ[j] * g * dz / 2
        p_upper += ρ[j] * g * dz
    end
    return p
end