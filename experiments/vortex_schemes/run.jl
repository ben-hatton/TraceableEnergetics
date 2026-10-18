# Nonlinear behaviour of the vortex-force schemes (diagnostic N), after Sadourny (1975): long inviscid AN2D runs,
# recording the total energy E, the potential enstrophy Z = ½ Σ ω² / ℑXZ(ρ_R) at the vertices, and the fraction of Z at
# high x-wavenumbers (kₓ > Mx/4), a measure of the grid-scale noise.
#
#   turbulence  homogeneous fluid, random large-scale vortices, at two resolutions: E and Z are both invariants, so the
#               energy-conserving scheme should keep E and let Z grow (a spurious cascade to the grid scale), the
#               enstrophy-conserving scheme the reverse
#   KH          the Kelvin-Helmholtz setup without viscosity or diffusion: E is invariant, Z is not (baroclinic torque)
#
# RK4's weak damping of the fastest modes plays the role of the dissipation of Sadourny's time filter.
#
#   julia -t 4 experiments/vortex_schemes/run.jl
#
# Output: output/turbulence.png, output/kelvin_helmholtz.png.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics
include("../../energetics/energetics.jl")      # energies!
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D
using MutatingOrNot: void
using CairoMakie, Random

space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
T0, q0, ρ0 = 280.0, 0.035, 1000.0
fluid = NonlinearBinaryFluid(:potential_temperature, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1 / ρ0, α_T = 2e-4,
    α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7))

function model(vortex_scheme; Mx, Mz)
    domain = Box2D(; Mx, Mz, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
    closed = Wall(tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))
    return AN2D((; mgr = SIMD(), domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme,
        buoyancy_scheme = ThermodynamicForcing(), thermoflux_scheme = ThermoFluxConsistent(k_T = 0.0, k_q = 0.0),
        viscosity_scheme = ViscosityScheme(dyn_visc = 0.0),
        anelastic_reference = AnelasticReference(domain, space, ρ0; ptop = 1e5),
        boundary_conditions = BoxBoundaryConditions(domain; bottom = closed, top = closed)))
end

#####
##### Setups
#####

# random large-scale vortices in a homogeneous fluid: modes kₓ, k_z ≤ 3 with random phases and amplitudes ∝ 1/k²,
# vanishing on the walls; the mass streamfunction is scaled for velocities ~ U
function turbulence(model; U = 0.01, seed = 1)
    (; Lx, Lz) = space
    rng = Random.Xoshiro(seed)
    modes = [(kx, kz, 2π * rand(rng), randn(rng) / (kx^2 + kz^2)) for kx in 0:3, kz in 1:3]
    ψ(X, Z) = ρ0 * U * Lz * sum(a * cos(2π * kx * X / Lx + φ) * sin(π * kz * Z / Lz) for (kx, kz, φ, a) in modes)
    return (; consvar = consvar_from_theta(fluid, T0, q0), q = (x, z) -> q0, u = z -> zero(z), ψ)
end

#####
##### Recording E, Z and the grid-scale fraction of Z
#####

struct Record
    t::Vector{Float64}
    E::Vector{Float64}
    Z::Vector{Float64}
    Z_high::Vector{Float64}
    ω::Ref{Matrix{Float64}}
end
Record() = Record(Float64[], Float64[], Float64[], Float64[], Ref(zeros(0, 0)))
Base.close(::Record) = nothing

function (r::Record)(model, state, dstate, scratch, book, t)
    (; domain, dx, dz, reference) = model
    (; ρ_R) = reference
    (; ℑ) = operators(model.spatial_scheme)
    X, Zf = interior_faces(domain, 1), interior_faces(domain, 2)
    x, z = interior_cells(domain, 1), interior_cells(domain, 2)
    _, _, te = energies!((void, void, void), model, state)
    ω = scratch.derivatives.∂u_z[X, Zf] .- scratch.derivatives.∂w_x[X, Zf]
    ρ_vertex = [ℑ(ρ_R, (2,), (Face(),), J) for J in Zf]'
    enstrophy = 0.5 .* ω .^ 2 ./ ρ_vertex
    spectrum = x_spectrum(ω ./ sqrt.(ρ_vertex))
    high = (size(ω, 1) ÷ 4 + 2):lastindex(spectrum)              # kₓ > Mx/4 (index kₓ + 1)
    push!(r.t, t)
    push!(r.E, sum(te[x, z]) * dx * dz)
    push!(r.Z, sum(enstrophy) * dx * dz)
    push!(r.Z_high, sum(spectrum[high]) / sum(spectrum))
    r.ω[] = ω
    return nothing
end

# Σ_z |ω̂(kₓ)|² for kₓ = 0 … N/2, by a direct discrete Fourier transform along x (N ≤ 64)
x_spectrum(a) = let N = size(a, 1)
    [sum(abs2, sum(a[n, :] .* cis(-2π * k * (n - 1) / N) for n in 1:N)) for k in 0:N÷2]
end

function run(model, fields, params)
    record = Record()
    state = initialize(model, fields)
    CFAnelastics.loop(state, model, CFTimeSchemes.RungeKutta4(model), params; outputs = record)
    return record
end

schemes = (VortexEnergyCons(), VortexEnstrophyCons(), VortexEnergyEnstrophyCons())
colours = Dict(VortexEnergyCons() => :firebrick, VortexEnstrophyCons() => :royalblue,
               VortexEnergyEnstrophyCons() => :seagreen)
out = mkpath(joinpath(@__DIR__, "output"))

function summary_figure(runs, title)
    fig = Figure(size = (1200, 800))
    axes = (Axis(fig[1, 1]; ylabel = "E / E(0) - 1", title),
            Axis(fig[1, 2]; ylabel = "Z / Z(0)"),
            Axis(fig[2, 1]; ylabel = "fraction of Z at kₓ > Mx/4", xlabel = "t [s]"))
    for (label, scheme, style, r) in runs
        kw = (; color = colours[scheme], linestyle = style, label)
        lines!(axes[1], r.t, r.E ./ r.E[1] .- 1; kw...)
        lines!(axes[2], r.t, r.Z ./ r.Z[1]; kw...)
        lines!(axes[3], r.t, r.Z_high; kw...)
    end
    Legend(fig[2, 2], axes[1])
    return fig
end

#####
##### Turbulence in a homogeneous fluid, two resolutions; about 200 turnover times Lx/(4U)
#####

turbulence_runs = []
for scheme in schemes, (Mx, Mz, style) in ((32, 16, :dash), (64, 32, :solid))
    m = model(scheme; Mx, Mz)
    r = run(m, turbulence(m), (; Nslice = 200, slice_size = 25.0, dt_max = 25.0, cfl = 0.9))
    push!(turbulence_runs, ("$(nameof(typeof(scheme))), $(Mx)×$(Mz)", scheme, style, r))
end
save(joinpath(out, "turbulence.png"), summary_figure(turbulence_runs, "turbulence, homogeneous fluid"))

#####
##### Inviscid Kelvin-Helmholtz
#####

kh_runs = []
for scheme in schemes
    m = model(scheme; Mx = 64, Mz = 32)
    r = run(m, kelvin_helmholtz(m; θ₀ = T0, Δθ = 0.1, q₀ = q0, Δu = 0.1, ρ₀ = ρ0),
            (; Nslice = 150, slice_size = 2.0, dt_max = 0.05, cfl = 0.9))
    push!(kh_runs, ("$(nameof(typeof(scheme)))", scheme, :solid, r))
end
fig = summary_figure(kh_runs, "Kelvin-Helmholtz, inviscid")
snapshots = fig[3, 1:2] = GridLayout()
for (col, (label, _, _, r)) in enumerate(kh_runs)
    ax = Axis(snapshots[1, col]; title = "ω at t = $(r.t[end]) s, $label", aspect = DataAspect())
    heatmap!(ax, r.ω[]; colormap = :balance, colorrange = (-1, 1) .* maximum(abs, r.ω[]))
end
save(joinpath(out, "kelvin_helmholtz.png"), fig)
println("\nsaved to $out")
