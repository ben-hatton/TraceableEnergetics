## BUOYANCY FREQUENCY, SHEAR AND GRADIENT RICHARDSON NUMBER
# Sign convention: z points up, g > 0 is the magnitude of gravity, buoyancy b = -g (ρ - ρ_ref) / ρ_ref, N² > 0 is stable.

"Gradient Richardson number above which the Miles-Howard theorem ensures linear stability of a parallel shear flow."
const RICHARDSON_CRITICAL = 1 // 4

"""
    N² = buoyancy_frequency2_from_buoyancy(dbdz)

Squared buoyancy frequency [1/s²], `N² = ∂b/∂z`, from the vertical gradient of buoyancy [1/s²]. Positive if stable.
"""
buoyancy_frequency2_from_buoyancy(dbdz) = dbdz

"""
    N² = buoyancy_frequency2_from_density(g, ρ, dρdz; c² = Inf)

Squared buoyancy frequency [1/s²], `N² = -(g/ρ) ∂ρ/∂z - g²/c²`, from the density `ρ` [kg/m³] and its vertical gradient
`dρdz` [kg/m⁴]. `c²` [m²/s²] is the squared adiabatic sound speed `(∂p/∂ρ)ₛ` of the fluid at fixed composition, for any
equation of state: the default `Inf` gives the incompressible/Boussinesq form (potential-density gradient).
For a compressible fluid, pass the in-situ density gradient and `c²`; for a potential-density gradient keep `c² = Inf`.
"""
buoyancy_frequency2_from_density(g, ρ, dρdz; c² = Inf) = -g / ρ * dρdz - g^2 / c²

"""
    N² = buoyancy_frequency2_from_coefficients(g, α, dθdz, β = 0, dSdz = 0)

Squared buoyancy frequency [1/s²], `N² = g (α ∂θ/∂z - β ∂S/∂z)`, of a binary fluid: `α` [1/K] thermal expansion and
`β` [1/(unit of S)] haline contraction coefficients of the equation of state, `θ` potential temperature [K], `S` the
second component. Gradients are of the conservative/potential variables, so no compressibility term appears.
"""
buoyancy_frequency2_from_coefficients(g, α, dθdz, β = 0, dSdz = 0) = g * (α * dθdz - β * dSdz)

"Buoyancy frequency `√N²` [1/s] for `N² ≥ 0`; `NaN` for unstable `N² < 0` (use `N²` itself there)."
buoyancy_frequency(N²) = N² ≥ 0 ? sqrt(N²) : oftype(float(N²), NaN)

"Buoyancy period `2π / N` [s] for `N² > 0` (`Inf` for `N² = 0`, `NaN` if unstable)."
buoyancy_period(N²) = 2π / buoyancy_frequency(N²)

"Squared vertical shear `S² = (∂u/∂z)² + (∂v/∂z)²` [1/s²]; `dvdz` optional (2D box)."
shear2(dudz, dvdz = 0) = dudz^2 + dvdz^2

"Gradient Richardson number `Ri = N² / S²` [1]; `Ri < 1/4` is necessary for shear instability (`RICHARDSON_CRITICAL`)."
richardson_gradient(N², S²) = N² / S²

"Shear-to-stratification ratio `S² / N²` [1] (inverse gradient Richardson number)."
shear_ratio(S², N²) = S² / N²

"Ozmidov scale `√(ε / N³)` [m], largest overturning scale in stratified turbulence, from dissipation `ε` [m²/s³]."
ozmidov_scale(ε, N²) = sqrt(ε / N²^(3 // 2))

"Buoyancy scale `U / N` [m]: vertical displacement at which kinetic energy is exhausted against stratification."
buoyancy_scale(U, N²) = U / sqrt(N²)

"""
    dfdz = vertical_gradient(f, dz)

Centred difference `(f[:, k+1] - f[:, k-1]) / 2dz` along the second dimension of a 2D array, one-sided at the first and
last rows. For an interior array on a uniform grid (no halo): the caller supplies fields at the points wanted.
"""
function vertical_gradient(f::AbstractMatrix, dz)
    g = similar(f, float(eltype(f)))
    n = size(f, 2)
    @views begin
        g[:, 2:n-1] .= (f[:, 3:n] .- f[:, 1:n-2]) ./ 2dz
        g[:, 1] .= (f[:, 2] .- f[:, 1]) ./ dz
        g[:, n] .= (f[:, n] .- f[:, n-1]) ./ dz
    end
    return g
end
