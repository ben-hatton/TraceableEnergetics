# Figures of the thermohaline lock exchange (see run.jl) from the files of a run, drawn by tools/budget_figures.jl: the
# global budgets of energy and of available energy, from the time series of the integrals over the box, the generation of
# available energy by heat and salt, the tendency of the reference state, the local budgets in the middle and at the end of the run, and maps of AE, ACE, APE.
#
#   julia examples/thermohaline_lock_exchange/figures.jl
@isdefined(models) || include(joinpath(@__DIR__, "models.jl"))

results = read_results(models, paths; series = map(read_series, series_paths))
files = String[]
save_figure(name, figure) = (file = joinpath(dir, name); save(file, figure); push!(files, file))
save_figure("global_budget.png", energy_budget_figure(results))
# the ACE budget over the first periods of the sound waves, the slow budgets as means over 0.25 s (about 8 periods)
save_figure("available_energy_budget.png", available_energy_budget_figure(results; fast = (0.0, 0.1), smooth = 0.25))
save_figure("generation.png", generation_figure(results; smooth = 0.25))
save_figure("reference_tendency.png", reference_tendency_figure(results; smooth = 0.25))
middle, final = cld(lastindex(results.times), 2), lastindex(results.times)
for key in keys(models), (n, when) in ((middle, "middle"), (final, "end"))
    save_figure("local_available_energy_budget_$(when)_$key.png", local_available_energy_budget_figure(results, key; n))
end
foreach(key -> save_figure("local_budget_$key.png", local_energy_budget_figure(results, key)), keys(models))
save_figure("available_energy.png", available_energy_maps(results))

print_summary(results)
println("figures saved to $(join(files, ", "))")
