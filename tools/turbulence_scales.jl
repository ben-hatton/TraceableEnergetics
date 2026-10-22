## DISSIPATION-BASED SCALES
# ε kinetic energy dissipation rate [m²/s³], ν kinematic viscosity [m²/s], D scalar diffusivity [m²/s].

"Kolmogorov length `η = (ν³/ε)^¼` [m]."
kolmogorov_length(ν, ε) = (ν^3 / ε)^(1 // 4)

"Kolmogorov time `(ν/ε)^½` [s]."
kolmogorov_time(ν, ε) = sqrt(ν / ε)

"Kolmogorov velocity `(ν ε)^¼` [m/s]."
kolmogorov_velocity(ν, ε) = (ν * ε)^(1 // 4)

"Batchelor length `(ν D² / ε)^¼ = η / √Sc` [m], smallest scale of a scalar of diffusivity `D` (Sc = ν/D ≫ 1)."
batchelor_length(ν, ε, D) = (ν * D^2 / ε)^(1 // 4)

"Batchelor wavenumber `1 / batchelor_length` [1/m]."
batchelor_wavenumber(ν, ε, D) = 1 / batchelor_length(ν, ε, D)

"Buoyancy Reynolds number `ε / (ν N²)` [1]; stratification matters at the smallest scales when it is O(1) or less."
buoyancy_reynolds(ε, ν, N²) = ε / (ν * N²)

"Ratio of the Ozmidov to the Kolmogorov scale, `(ε / (ν N²))^¾` [1] (= `buoyancy_reynolds^(3/4)`)."
ozmidov_kolmogorov_ratio(ε, ν, N²) = buoyancy_reynolds(ε, ν, N²)^(3 // 4)

"Eddy turnover time `L / u` [s] of eddies of size `L` and velocity `u`."
eddy_turnover_time(L, u) = L / u

"Taylor microscale `√(15 ν u² / ε)` [m] for isotropic turbulence of rms velocity `u` [m/s]."
taylor_microscale(ν, u, ε) = sqrt(15 * ν * u^2 / ε)

"Grid resolution ratio `Δx / η` [1], to check that a simulation resolves the Kolmogorov scale (≲ 1-2 for DNS)."
grid_resolution_ratio(Δx, η) = Δx / η
