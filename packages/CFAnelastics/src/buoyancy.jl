# Buoyancy schemes: the momentum forcing of AN2D

"""
    BuoyancyScheme

Momentum forcing of the anelastic model by the thermodynamic state: `ThermodynamicForcing`, or a buoyancy force
(`BuoyancyForce`), `ConsistentBuoyancy` or `LegacyBuoyancy`.
"""
abstract type BuoyancyScheme end

"""
    ThermodynamicForcing()

Momentum forcing in thermodynamic form, -consvar ∇conjvar - q ∇chempot - ∇B - g ẑ, with B = h - conjvar consvar -
chempot q the potential of the forcing, all at the reference pressure p_R; it has a horizontal component.
"""
struct ThermodynamicForcing <: BuoyancyScheme end

"""
    BuoyancyForce

Momentum forcing by a buoyancy b [m s⁻²] along z only, computed at the cells and averaged to the w-faces:
`ConsistentBuoyancy` or `LegacyBuoyancy`.
"""
abstract type BuoyancyForce <: BuoyancyScheme end

"""
    ConsistentBuoyancy()

Buoyancy force with b = g (ρ_R v - 1), v the specific volume at the reference pressure p_R: the energetically
consistent buoyancy of Tailleux & Dubos (2024).
"""
struct ConsistentBuoyancy <: BuoyancyForce end

"""
    LegacyBuoyancy()

Buoyancy force with the traditional b = g (1 - 1/(ρ_R v)) = -g (ρ - ρ_R)/ρ_R, ρ = 1/v the density at the reference
pressure p_R.
"""
struct LegacyBuoyancy <: BuoyancyForce end
