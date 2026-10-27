# Available energy budget of a thermohaline lock exchange in the anelastic (AN2D) and fully compressible (FC2D) models:
# warm, salty water on the left of a closed box and cold, fresh water on the right, the salt offsetting part of the
# difference of density due to heat, in a weakly stratified background. Every pathway of the budget of the available
# energy is active:
# - compression: the two waters are put in place at the density of the background (as if heated and salted at constant
#   volume), so their pressures differ from it by Δp = -(∂v/∂ϑ Δϑ + ∂v/∂q Δq)/(∂v/∂p), about ±360 Pa. In FC2D this is
#   available compressible energy (ACE) Π₁ = h(p) - h(P_*) - (p - P_*) v ≈ |∂v/∂p| (p - P_*)²/2, released into sound
#   waves through the conversion C_ACE (C_b,1* + C_D* - ∇·J_C*); AN2D, which has no sound, starts from the same density and
#   tracers, with no ACE;
# - displacement: the available potential energy (APE) Π₂ of the two waters side by side drives the gravity current,
#   through the conversion C_APE (C_b,2*, the buoyancy relative to the reference state);
# - heat and salt: the two diffuse at different rates (k_T = 1e-4, k_q = 3e-5 m² s⁻¹), and both generate or destroy
#   available energy, G = ∫(-Q_ϑ + Q*_ϑ - Q_q + Q*_q + Q - Q*) dV, through the heatings at the parcel's state (Q) and at
#   its reference state (Q*) (tools/budget_figures.jl gives the budgets).
# The background is stably stratified (N² ≈ 1e-3 s⁻²) so that the reference height of every parcel is well defined.
# The sound waves exchange energy between K and ACE at 16 to 32 Hz, faster than the outputs of the fields every 0.5 s:
# the global budgets are therefore recorded as integrals over the box at a finer interval (`IntegralSeries`).
#
#   julia -t 4 examples/thermohaline_lock_exchange/run.jl         (about 5 minutes, with the figures)
#   julia examples/thermohaline_lock_exchange/figures.jl          (the figures only, from the files of a run)
#
# The models and the initial state are in models.jl, the figures in figures.jl. The data files, time series and figures
# are saved to examples/thermohaline_lock_exchange/output/.
include("models.jl")

#####
##### Runs
#####

# the integrals over the box, saved to series_paths when the run ends
series = map(models, lorenz, series_paths) do model, ref, file
    IntegralSeries(model, budget_names(model); reference_state = ref, file)
end
for key in keys(models)
    model = models[key]
    println("\n", nameof(typeof(model)))
    book = diagnostics(model; reference = references[key], reference_state = lorenz[key])
    # the time series, a data file of the fields, and a video of the potential temperature
    outputs = Outputs(series[key],
                      Thinned(DataFile(paths[key], model, budget_names(model); reference_state = lorenz[key]), every),
                      Thinned(DiagnosticVideo(model, :conservative_variable; file = joinpath(dir, "theta_$key.mp4")), every))
    loops[key](states[key], model, CFTimeSchemes.RungeKutta4(model), params; outputs, book)
end

include("figures.jl")
