## FIGURES OF THE ENERGY AND AVAILABLE ENERGY BUDGETS OF A RUN OF AN2D AND FC2D
#
# Shared by the examples that record the budgets (examples/energy_budget, examples/thermohaline_lock_exchange):
#     include("<relative path>/tools/budget_figures.jl")
# after the packages of the example (TraceableEnergetics, CFBoxes, CairoMakie). A run of each model records the
# diagnostics `budget_names(model)` in a data file, and possibly their integrals over the box at a finer interval in an
# `IntegralSeries` (`read_series`); `read_results` reads them back as `results = (; models, data, times, series)`, from which each
# function below draws a figure.
#
# The budgets are those of the diagnostics of TraceableEnergetics: with K the kinetic and P the potential energy per unit
# mass,
#     ∂ₜ(ρK) = -∇·(J_K^rev + J_K^irr) + C - ρε_K + VF,
#     ∂ₜ(ρP) = -∇·(J_P^rev + J_P^irr) - C - Q_ϑ - Q_q + Q,
# and, with the available compressible energy (ACE) Π₁, the available potential energy (APE) Π₂ and the potentials
# Σ_ϑ, Σ_q at the intermediate state (z, P_*(z)) (Σ̃) and at the reference state (z_*, P_*(z_*)) (Σ*), at a fixed
# reference state,
#     ∂ₜ(ρΠ₁) = -∇·(J_ACE^rev + J_ACE^irr) - C_ACE - (Q_ϑ - Q̃_ϑ) - (Q_q - Q̃_q) + Q - Q̃,
#     ∂ₜ(ρΠ₂) = -∇·(J_APE^rev + J_APE^irr) - C_APE - (Q̃_ϑ - Q*_ϑ) - (Q̃_q - Q*_q) + Q̃ - Q*,
# with C = C_ACE + C_APE + C_bg. With closed walls, the global budgets of the available energy and of the background are
#     d/dt ∫ρΠ dV = -∫(C_ACE + C_APE) dV + G,    d/dt ∫ρ(P - Π) dV = -∫C_bg dV + ∫(Q* - Q*_ϑ - Q*_q) dV,
# with the generation G = ∫(-Q_ϑ + Q*_ϑ - Q_q + Q*_q + Q - Q*) dV, up to ∫R_Π dV = 0 (the terms in ∂ₜP_*).

using CFBoxes: CFBoxes

#####
##### Diagnostics to record, and reading them back
#####

"""
    names = budget_names(model)

The diagnostics of `model` recorded for the figures: the local budgets of ρK and ρP and the manipulated conversion at the
cells [W m⁻³], their integrals [J m⁻¹ or W m⁻¹], the available energies AE = Π, ACE = Π₁, APE = Π₂ [J kg⁻¹] with their
integrals, the reference profiles P_*, v_*, the background potential energy ∫ρ(P - Π) dV and the checks of the reference
state, the terms of the budgets of ACE and APE at the cells [W m⁻³], with their tendencies and the terms in ∂ₜP_*, and
the tendency ∂ₜP_* of the reference pressure from the diabatic tendencies; for FC2D, the top pressure of the reference
column, its excess height and its tendency.
"""
function budget_names(model)
    kinetic = (:dρK, :div_kinetic_flux_rev, :div_kinetic_flux_irr, :conversion, :viscous_dissipation, :vortex_work,
               :kinetic_budget_residual)
    potential = (:dρP, :div_potential_flux_rev, :div_potential_flux_irr, :heating, :heating_consvar, :heating_composition,
                 :potential_budget_residual, :heating_residual)
    conversion = (:div_pressure_flux, :buoyancy_conversion, :pressure_dilatation, :conversion_residual)
    integrals = (:integrated_kinetic_energy, :integrated_potential_energy, :integrated_energy_tendency,
                 :integrated_boundary_flux, :integrated_vortex_work, :integrated_budget_residual)
    available = (:AE, :ACE, :APE, :p_star, :v_star, :integrated_AE, :integrated_ACE, :integrated_APE,
                 :background_potential_energy, :certificate_violations, :wall_minima, :pivots, :pricing_passes,
                 :sorted_energy_excess)
    available_budget = (:div_ACE_flux_rev, :div_ACE_flux_irr, :div_APE_flux_rev, :div_APE_flux_irr, :conversion_ACE,
                        :conversion_APE, :background_conversion, :heating_consvar_at_p_star, :heating_composition_at_p_star,
                        :heating_at_p_star, :heating_consvar_at_z_star, :heating_composition_at_z_star, :heating_at_z_star,
                        :buoyancy_conversion_ACE, :buoyancy_conversion_APE, :pressure_dilatation_star,
                        :div_pressure_flux_star, :ACE_conversion_residual, :APE_conversion_residual)
    reference_tendency = (:p_star_tendency, :plain_weight_slots, :reference_pressure_term_ACE, :reference_pressure_term_APE,
                          :dρACE, :dρAPE, :ACE_budget_residual, :APE_budget_residual)
    column = model isa FC2D ? (:p_top, :volume_residual, :p_top_tendency, :top_pressure_iterations) : ()
    return (kinetic..., potential..., conversion..., integrals..., available..., available_budget..., reference_tendency..., column...)
end

"""
    results = read_results(models, paths; series = nothing)

Every diagnostic recorded in the data files `paths` of the runs of `models` (named tuples with the same keys), read back
as `results = (; models, data, times, series)`: `data[key][name]` is an (x, z, t) array with halo for a field, a vector over
t for a number, and `times` [s] are those of the outputs. With `series` (an `IntegralSeries` per model), the time series
of the global figures are those of the series rather than of the data files.
"""
function read_results(models, paths; series = nothing)
    data = map(models, paths) do model, path
        names = budget_names(model)
        NamedTuple{names}(map(name -> first(read_diagnostic(path, name)), names))
    end
    return (; models, data, times = last(read_diagnostic(first(paths), :dρK)), series)
end

"""
    t = series_times(results, key)
    I = series_integral(results, key, name)

The times [s] of the time series of the model `key`, and the time series of the diagnostic `name` there: its integral
over the box for a field, its value for a number; from the `IntegralSeries` of the results if any, otherwise from the
data files.
"""
series_times(results, key) = isnothing(results.series) ? results.times : results.series[key].times
function series_integral(results, key, name)
    isnothing(results.series) || return results.series[key].values[name]
    value = results.data[key][name]
    return value isa AbstractVector ? value : integral(results, key, name)
end

"""
    f = field(results, key, name, n)
    I = integral(results, key, name)

The interior of the field `name` of the model `key` at output n, and its integral over the box at every output,
∫ field dx dz [W m⁻¹ for a term of a budget].
"""
field((; models, data), key, name, n) = interior(models[key].domain, view(data[key][name], :, :, n))
integral(results, key, name) =
    [sum(field(results, key, name, n)) * results.models[key].dx * results.models[key].dz for n in eachindex(results.times)]

# rates between successive outputs, at the midpoints of the intervals: the differences of a series y, and the average of
# a rate r (the trapezoidal rule), both second order in the output interval
midpoints(times) = (times[1:end-1] .+ times[2:end]) ./ 2
difference(times, y) = diff(y) ./ diff(times)
average(r) = (r[1:end-1] .+ r[2:end]) ./ 2
# means over consecutive blocks of n samples (the samples beyond the last full block are dropped), with n = 1 the series
# itself: a block mean of differences, (y_end - y_start)/(t_end - t_start) on a uniform series, telescopes exactly
block_means(y, n) = n == 1 ? y : [sum(view(y, k:k+n-1)) / n for k in 1:n:length(y)-n+1]
# the number of samples of the series `times` in a window of `smooth` seconds (1 for no smoothing)
block_size(times, smooth) = isnothing(smooth) ? 1 : max(1, round(Int, smooth / (times[2] - times[1])))

#####
##### Global budgets: the integrals over the box against time, one column per model
#####

"""
    figure = energy_budget_figure(results)

The global budgets of ∫ρK and ∫ρP and the closure of that of ∫ρ(K + P), against time, one column per model.
"""
function energy_budget_figure(results)
    (; models) = results
    I(key, name) = series_integral(results, key, name)
    figure = Figure(size = (1100, 1200))
    for (column, key) in enumerate(keys(models))
        times = series_times(results, key)
        label = nameof(typeof(models[key]))
        # energies, as changes from t = 0 (∫ρP is about 10¹⁰ times larger than ∫ρK)
        ax = Axis(figure[1, column]; title = "$label: energy change from t = 0", ylabel = "J m⁻¹")
        ΔK = I(key, :integrated_kinetic_energy) .- I(key, :integrated_kinetic_energy)[1]
        ΔP = I(key, :integrated_potential_energy) .- I(key, :integrated_potential_energy)[1]
        lines!(ax, times, ΔK; label = "∫ρK")
        lines!(ax, times, ΔP; label = "∫ρP")
        lines!(ax, times, ΔK .+ ΔP; label = "∫ρ(K + P)", linestyle = :dash, color = :black)
        axislegend(ax; position = :lt)
        # budget of ∫ρK: d/dt ∫ρK = -∮J_K·dS + ∫C - ∫ρε_K + ∫VF
        ax = Axis(figure[2, column]; title = "$label: budget of ∫ρK", ylabel = "W m⁻¹")
        lines!(ax, times, I(key, :dρK); label = "d/dt ∫ρK", color = :black)
        lines!(ax, times, I(key, :conversion); label = "∫C")
        lines!(ax, times, -I(key, :viscous_dissipation); label = "-∫ρε_K")
        lines!(ax, times, -I(key, :div_kinetic_flux_rev) .- I(key, :div_kinetic_flux_irr); label = "-∮J_K·dS")
        lines!(ax, times, I(key, :integrated_vortex_work); label = "∫VF")
        axislegend(ax; position = :rt)
        # budget of ∫ρP: d/dt ∫ρP = -∮J_P·dS - ∫C + ∫(Q - Q_ϑ - Q_q)
        ax = Axis(figure[3, column]; title = "$label: budget of ∫ρP", ylabel = "W m⁻¹")
        lines!(ax, times, I(key, :dρP); label = "d/dt ∫ρP", color = :black)
        lines!(ax, times, -I(key, :conversion); label = "-∫C")
        lines!(ax, times, I(key, :heating) .- I(key, :heating_consvar) .- I(key, :heating_composition); label = "∫(Q - Q_ϑ - Q_q)")
        lines!(ax, times, -I(key, :div_potential_flux_rev) .- I(key, :div_potential_flux_irr); label = "-∮J_P·dS")
        axislegend(ax; position = :rt)
        # closure of the global budget: d/dt ∫ρ(K + P) + ∮(J_K + J_P)·dS - ∫VF, against the scale of the exchanges
        ax = Axis(figure[4, column]; title = "$label: residual of the budget of ∫ρ(K + P)", xlabel = "t [s]", ylabel = "W m⁻¹")
        lines!(ax, times, I(key, :integrated_budget_residual); label = "residual", color = :black)
        lines!(ax, times, I(key, :integrated_energy_tendency); label = "d/dt ∫ρ(K + P)")
        lines!(ax, times, I(key, :integrated_boundary_flux); label = "∮(J_K + J_P)·dS")
        axislegend(ax; position = :rt)
    end
    return figure
end

"""
    figure = available_energy_budget_figure(results; fast = full range, smooth = nothing)

The global budgets of the available energy against time, one column per model, each budget as d/dt of its energy from
differences of the outputs against the diagnosed terms averaged between them (what they leave is the error of the
differences, measured by the same estimate for ∫ρK against its exact tendency, and the change of the reference state that
the diabatic tendencies do not give, from the discretisation of advection: see `reference_tendency_figure`), with the
terms R_Π₁, R_Π₂ in the tendency of the reference pressure ∂ₜP_*:
- the energies, as changes from t = 0;
- the APE, d/dt ∫ρΠ₂ = -∫C_APE + G_APE + ∫R_Π₂, with G_APE = ∫(-(Q̃_ϑ - Q*_ϑ) - (Q̃_q - Q*_q) + Q̃ - Q*);
- the ACE, d/dt ∫ρΠ₁ = -∫C_ACE + G_ACE + ∫R_Π₁, with G_ACE = ∫(-(Q_ϑ - Q̃_ϑ) - (Q_q - Q̃_q) + Q - Q̃), over the times
  `fast` (a range (t₀, t₁) [s]): in FC2D the ACE is exchanged with K by sound waves, much faster than the other budgets;
- the conversions to K, C = C_ACE + C_APE + C_bg, and the form of the energy notes C_APE ≈ C_b,2*;
- K and the available energy, d/dt ∫ρ(K + Π) = ∫C_bg - ∫ρε_K + ∫VF + G + ∫R_Π (closed walls), in which the sound waves
  cancel, and ∫R_Π = ∫(R_Π₁ + R_Π₂) is zero to the resolution of the slots;
- the background, d/dt ∫ρ(P - Π) = -∫C_bg + ∫(Q* - Q*_ϑ - Q*_q) - ∫R_Π.
With `smooth` [s], all but the energies and the ACE are means over blocks of that duration, which remove the sound waves
of FC2D from the slow budgets (the block means of the differences telescope, so the budgets still compare exactly).
"""
function available_energy_budget_figure(results; fast = nothing, smooth = nothing)
    (; models) = results
    I(key, name) = series_integral(results, key, name)
    figure = Figure(size = (1100, 2000))
    for (column, key) in enumerate(keys(models))
        times = series_times(results, key)
        mid = midpoints(times)
        d_dt(y) = difference(times, y)
        # the slow budgets as block means over `smooth` seconds, on the midpoints of the blocks
        n = block_size(times, smooth)
        slow(y) = block_means(y, n)
        mid_slow = slow(mid)
        label = nameof(typeof(models[key]))
        # the parts of the generation, at the intermediate state (ACE) and at the reference state (APE)
        G_ACE = average(.-I(key, :heating_consvar) .+ I(key, :heating_consvar_at_p_star) .- I(key, :heating_composition) .+
                        I(key, :heating_composition_at_p_star) .+ I(key, :heating) .- I(key, :heating_at_p_star))
        G_APE = average(.-I(key, :heating_consvar_at_p_star) .+ I(key, :heating_consvar_at_z_star) .-
                        I(key, :heating_composition_at_p_star) .+ I(key, :heating_composition_at_z_star) .+
                        I(key, :heating_at_p_star) .- I(key, :heating_at_z_star))
        C_ACE, C_APE = average(I(key, :conversion_ACE)), average(I(key, :conversion_APE))
        # the terms in ∂ₜP_*
        R_ACE, R_APE = average(I(key, :reference_pressure_term_ACE)), average(I(key, :reference_pressure_term_APE))
        # the error of the differences, from those of ∫ρK against the average of its exact tendency
        error = d_dt(I(key, :integrated_kinetic_energy)) .- average(I(key, :dρK))
        # energies, as changes from t = 0
        ax = Axis(figure[1, column]; title = "$label: energy change from t = 0", ylabel = "J m⁻¹")
        change(y) = y .- y[1]
        lines!(ax, times, change(I(key, :integrated_kinetic_energy)); label = "∫ρK")
        lines!(ax, times, change(I(key, :integrated_APE)); label = "∫ρΠ₂ (APE)")
        lines!(ax, times, change(I(key, :integrated_ACE)); label = "∫ρΠ₁ (ACE)")
        lines!(ax, times, change(I(key, :background_potential_energy)); label = "∫ρ(P - Π)")
        lines!(ax, times, change(I(key, :integrated_kinetic_energy) .+ I(key, :integrated_AE)); label = "∫ρ(K + Π)",
               linestyle = :dash, color = :black)
        axislegend(ax; position = :lb)
        # the APE
        ax = Axis(figure[2, column]; title = "$label: budget of ∫ρΠ₂ (APE)$(n > 1 ? ", means over $smooth s" : "")", ylabel = "W m⁻¹")
        lines!(ax, mid_slow, slow(d_dt(I(key, :integrated_APE))); label = "d/dt ∫ρΠ₂", color = :black)
        lines!(ax, mid_slow, slow(.-C_APE .+ G_APE .+ R_APE); label = "-∫C_APE + G_APE + ∫R_Π₂", color = :black, linestyle = :dot)
        lines!(ax, mid_slow, slow(.-C_APE); label = "-∫C_APE")
        lines!(ax, mid_slow, slow(G_APE); label = "G_APE")
        lines!(ax, mid_slow, slow(R_APE); label = "∫R_Π₂")
        lines!(ax, mid_slow, slow(d_dt(I(key, :integrated_APE)) .+ C_APE .- G_APE .- R_APE); label = "residual")
        axislegend(ax; position = :lb)
        # the ACE, over the times `fast`
        shown = isnothing(fast) ? eachindex(mid) : findall(t -> fast[1] <= t <= fast[2], mid)
        ax = Axis(figure[3, column]; title = "$label: budget of ∫ρΠ₁ (ACE)", ylabel = "W m⁻¹")
        lines!(ax, mid[shown], d_dt(I(key, :integrated_ACE))[shown]; label = "d/dt ∫ρΠ₁", color = :black)
        lines!(ax, mid[shown], (.-C_ACE .+ G_ACE .+ R_ACE)[shown]; label = "-∫C_ACE + G_ACE + ∫R_Π₁", color = :black, linestyle = :dot)
        lines!(ax, mid[shown], G_ACE[shown]; label = "G_ACE")
        lines!(ax, mid[shown], R_ACE[shown]; label = "∫R_Π₁")
        lines!(ax, mid[shown], (d_dt(I(key, :integrated_ACE)) .+ C_ACE .- G_ACE .- R_ACE)[shown]; label = "residual")
        lines!(ax, mid[shown], error[shown]; label = "error of d/dt (from ∫ρK)", color = :gray)
        axislegend(ax; position = :lb)
        # the conversions to K
        ax = Axis(figure[4, column]; title = "$label: conversions to K but C_ACE", ylabel = "W m⁻¹")
        lines!(ax, slow(times), slow(I(key, :conversion_APE)); label = "∫C_APE")
        lines!(ax, slow(times), slow(I(key, :buoyancy_conversion_APE)); label = "∫C_b,2*", linestyle = :dot)
        lines!(ax, slow(times), slow(I(key, :background_conversion)); label = "∫C_bg")
        axislegend(ax; position = :lb)
        # K and the available energy, in which the exchanges between K and ACE cancel
        generation = G_ACE .+ G_APE
        R = R_ACE .+ R_APE
        total = average(I(key, :background_conversion) .- I(key, :viscous_dissipation) .+ I(key, :integrated_vortex_work)) .+ generation .+ R
        d_total = d_dt(I(key, :integrated_kinetic_energy) .+ I(key, :integrated_AE))
        ax = Axis(figure[5, column]; title = "$label: budget of ∫ρ(K + Π)", ylabel = "W m⁻¹")
        lines!(ax, mid_slow, slow(d_total); label = "d/dt ∫ρ(K + Π)", color = :black)
        lines!(ax, mid_slow, slow(total); label = "∫C_bg - ∫ρε_K + ∫VF + G + ∫R_Π", color = :black, linestyle = :dot)
        lines!(ax, mid_slow, slow(generation); label = "G = G_ACE + G_APE")
        lines!(ax, mid_slow, slow(R); label = "∫R_Π")
        lines!(ax, mid_slow, slow(average(.-I(key, :viscous_dissipation))); label = "-∫ρε_K")
        lines!(ax, mid_slow, slow(d_total .- total); label = "residual")
        axislegend(ax; position = :lb)
        # the background
        background = average(.-I(key, :background_conversion) .+ I(key, :heating_at_z_star) .-
                             I(key, :heating_consvar_at_z_star) .- I(key, :heating_composition_at_z_star)) .- R
        ax = Axis(figure[6, column]; title = "$label: budget of ∫ρ(P - Π)", xlabel = "t [s]", ylabel = "W m⁻¹")
        lines!(ax, mid_slow, slow(d_dt(I(key, :background_potential_energy))); label = "d/dt ∫ρ(P - Π)", color = :black)
        lines!(ax, mid_slow, slow(background); label = "-∫C_bg + ∫(Q* - Q*_ϑ - Q*_q) - ∫R_Π", color = :black, linestyle = :dot)
        lines!(ax, mid_slow, slow(average(.-I(key, :background_conversion))); label = "-∫C_bg")
        lines!(ax, mid_slow, slow(d_dt(I(key, :background_potential_energy)) .- background); label = "residual")
        axislegend(ax; position = :lb)
    end
    return figure
end

"""
    figure = generation_figure(results; smooth = nothing)

The generation of available energy by heat and salt, against time, one column per model: G = G_ϑ + G_q + G_σ with
G_ϑ = -∫(Q_ϑ - Q*_ϑ), G_q = -∫(Q_q - Q*_q) and G_σ = ∫(Q - Q*), and the part that goes to the ACE (from the parcel's
pressure to P_*), as means over blocks of `smooth` seconds if given.
"""
function generation_figure(results; smooth = nothing)
    (; models) = results
    I(key, name) = series_integral(results, key, name)
    figure = Figure(size = (1100, 500))
    for (column, key) in enumerate(keys(models))
        n = block_size(series_times(results, key), smooth)
        times = block_means(series_times(results, key), n)
        label = nameof(typeof(models[key]))
        G_ϑ = block_means(.-I(key, :heating_consvar) .+ I(key, :heating_consvar_at_z_star), n)
        G_q = block_means(.-I(key, :heating_composition) .+ I(key, :heating_composition_at_z_star), n)
        G_σ = block_means(I(key, :heating) .- I(key, :heating_at_z_star), n)
        ax = Axis(figure[1, column]; title = "$label: generation of available energy", xlabel = "t [s]", ylabel = "W m⁻¹")
        lines!(ax, times, G_ϑ .+ G_q .+ G_σ; label = "G", color = :black)
        lines!(ax, times, G_ϑ; label = "-∫(Q_ϑ - Q*_ϑ), heat")
        lines!(ax, times, G_q; label = "-∫(Q_q - Q*_q), salt")
        lines!(ax, times, G_σ; label = "∫(Q - Q*), production of ϑ")
        # the part of the heat and salt terms that goes to ACE (from the parcel's pressure to P_*)
        lines!(ax, times, block_means(.-I(key, :heating_consvar) .+ I(key, :heating_consvar_at_p_star) .- I(key, :heating_composition) .+
                                      I(key, :heating_composition_at_p_star), n); label = "of which to ACE", linestyle = :dot)
        axislegend(ax; position = :rb)
    end
    return figure
end

#####
##### Local budgets: maps of each term at the end of the run, one figure per model
#####

# Each term is shown as its contribution to the tendency (with its sign in the budget), on its own symmetric colour
# range: the terms span many orders of magnitude (the divergence of ρΣu is large at a cell, but integrates to the wall
# flux), and the residuals are of round-off size. A panel is (diagnostic, sign in the budget, title), or
# (((diagnostic, sign), ...), title) for a sum of diagnostics, all on the cells. The titles write the terms at the
# intermediate state (z, P_*(z)) with "(P_*)" (Q̃_ϑ is Q_ϑ(P_*)), as the font of the figures has no combining tilde.
panel_terms((name, sign, label)::Tuple{Symbol, Any, Any}) = (((name, sign),), label)
panel_terms(panel::Tuple{Tuple, Any}) = panel
function budget_row!(figure, row, results, key, title, panels; n = lastindex(results.times))
    x, z = diagnostic_coordinates(results.models[key], :conversion)
    Label(figure[row, 1], title; rotation = π / 2, tellheight = false)
    for (k, panel) in enumerate(panels)
        terms, label = panel_terms(panel)
        map = sum(sign * field(results, key, name, n) for (name, sign) in terms)
        # a zero field (C_D for AN2D) gets the smallest nonzero range
        limit = max(maximum(abs, map), floatmin(eltype(map)))
        cell = figure[row, k + 1] = GridLayout()
        ax = Axis(cell[1, 1]; title = label, aspect = DataAspect())
        heatmap!(ax, x, z, map; colormap = :balance, colorrange = (-limit, limit))
        hidedecorations!(ax)
        Colorbar(cell[2, 1]; colormap = :balance, limits = (-limit, limit), vertical = false, flipaxis = false,
                 label = "W m⁻³", ticklabelsize = 10, ticks = Makie.WilkinsonTicks(3))
    end
end

"""
    figure = local_energy_budget_figure(results, key; n = last output)

Maps of the terms of the budgets of ρK and ρP, and of the manipulated conversion, of the model `key` at output n.
"""
function local_energy_budget_figure(results, key; n = lastindex(results.times))
    figure = Figure(size = (2000, 800))
    budget_row!(figure, 1, results, key, "budget of ρK",
        ((:dρK, 1, "∂ₜ(ρK)"), (:div_kinetic_flux_rev, -1, "-∇·J_K^rev"), (:div_kinetic_flux_irr, -1, "-∇·J_K^irr"),
         (:conversion, 1, "C"), (:viscous_dissipation, -1, "-ρε_K"), (:vortex_work, 1, "VF"),
         (:kinetic_budget_residual, 1, "residual (zero in ∫ only)")); n)
    budget_row!(figure, 2, results, key, "budget of ρP",
        ((:dρP, 1, "∂ₜ(ρP)"), (:div_potential_flux_rev, -1, "-∇·J_P^rev"), (:div_potential_flux_irr, -1, "-∇·J_P^irr"),
         (:conversion, -1, "-C"), (:heating_consvar, -1, "-Q_ϑ"), (:heating_composition, -1, "-Q_q"),
         (:heating, 1, "Q"), (:potential_budget_residual, 1, "residual")); n)
    budget_row!(figure, 3, results, key, "conversion",
        ((:conversion, 1, "C"), (:div_pressure_flux, -1, "-∇·J_C"), (:buoyancy_conversion, 1, "C_b"),
         (:pressure_dilatation, 1, "C_D"), (:conversion_residual, 1, "C + ∇·J_C - C_b - C_D")); n)
    Label(figure[0, :], "$(nameof(typeof(results.models[key]))): local energy budget at t = $(results.times[n]) s"; fontsize = 18)
    return figure
end

"""
    figure = local_available_energy_budget_figure(results, key; n = last output)

Maps of the terms of the budgets of ρΠ₂ (APE) and ρΠ₁ (ACE) of the model `key` at output n, each as its contribution to
the tendency, with the terms in ∂ₜP_*, the tendency and the residual of the budget, and the conversions against the forms
of the energy notes.
"""
function local_available_energy_budget_figure(results, key; n = lastindex(results.times))
    figure = Figure(size = (2250, 800))
    budget_row!(figure, 1, results, key, "budget of ρΠ₂ (APE)",
        ((:div_APE_flux_rev, -1, "-∇·J_APE^rev"), (:div_APE_flux_irr, -1, "-∇·J_APE^irr"), (:conversion_APE, -1, "-C_APE"),
         (((:heating_consvar_at_p_star, -1), (:heating_consvar_at_z_star, 1)), "-(Q_ϑ(P_*) - Q*_ϑ)"),
         (((:heating_composition_at_p_star, -1), (:heating_composition_at_z_star, 1)), "-(Q_q(P_*) - Q*_q)"),
         (((:heating_at_p_star, 1), (:heating_at_z_star, -1)), "Q(P_*) - Q*"), (:reference_pressure_term_APE, 1, "R_Π₂"),
         (:dρAPE, 1, "∂ₜ(ρΠ₂)"), (:APE_budget_residual, 1, "residual")); n)
    budget_row!(figure, 2, results, key, "budget of ρΠ₁ (ACE)",
        ((:div_ACE_flux_rev, -1, "-∇·J_ACE^rev"), (:div_ACE_flux_irr, -1, "-∇·J_ACE^irr"), (:conversion_ACE, -1, "-C_ACE"),
         (((:heating_consvar, -1), (:heating_consvar_at_p_star, 1)), "-(Q_ϑ - Q_ϑ(P_*))"),
         (((:heating_composition, -1), (:heating_composition_at_p_star, 1)), "-(Q_q - Q_q(P_*))"),
         (((:heating, 1), (:heating_at_p_star, -1)), "Q - Q(P_*)"), (:reference_pressure_term_ACE, 1, "R_Π₁"),
         (:dρACE, 1, "∂ₜ(ρΠ₁)"), (:ACE_budget_residual, 1, "residual")); n)
    budget_row!(figure, 3, results, key, "conversions",
        ((:conversion_APE, 1, "C_APE"), (:buoyancy_conversion_APE, 1, "C_b,2*"), (:APE_conversion_residual, 1, "C_APE - C_b,2*"),
         (:conversion_ACE, 1, "C_ACE"),
         (((:buoyancy_conversion_ACE, 1), (:pressure_dilatation_star, 1), (:div_pressure_flux_star, -1)), "C_b,1* + C_D* - ∇·J_C*"),
         (:ACE_conversion_residual, 1, "C_ACE - C_b,1* - C_D* + ∇·J_C*"), (:background_conversion, 1, "C_bg")); n)
    Label(figure[0, :], "$(nameof(typeof(results.models[key]))): local budgets of the available energy at t = $(results.times[n]) s";
          fontsize = 18)
    return figure
end

"""
    figure = reference_tendency_figure(results; smooth = nothing, profiles = 3)

The tendency of the reference state, one column per model:
- the change of ∫ρΠ₁ and ∫ρΠ₂ that the diagnosed tendencies ∫∂ₜ(ρΠ₁), ∫∂ₜ(ρΠ₂) do not give: d/dt of the energy from
  differences of the outputs, less the average of the diagnosed tendency between them, as means over blocks of `smooth`
  seconds. It is the change of the reference state that the diabatic tendencies do not give, from the discretisation of
  advection (spurious mixing, and the change of the slot masses in FC2D, which are those of the rows), plus the error of the
  differences, measured by the same estimate for ∫ρK;
- the integrals of the terms in ∂ₜP_*, ∫R_Π₁, ∫R_Π₂ and ∫R_Π, the last zero to the resolution of the slots;
- ∂ₜP_* from the diabatic tendencies against the centred differences of P_* between the outputs of the data files, on the
  rows, at `profiles` outputs spread over the run (the differences also hold the change from advection, and the noise of
  the sorted placement between outputs).
"""
function reference_tendency_figure(results; smooth = nothing, profiles = 3)
    (; models, data) = results
    I(key, name) = series_integral(results, key, name)
    figure = Figure(size = (1100, 1300))
    for (column, key) in enumerate(keys(models))
        model = models[key]
        label = nameof(typeof(model))
        times = series_times(results, key)
        mid = midpoints(times)
        d_dt(y) = difference(times, y)
        n = block_size(times, smooth)
        slow(y) = block_means(y, n)
        # the change not given by the diagnosed tendencies, and the error of the differences
        ax = Axis(figure[1, column]; title = "$label: d/dt ∫ρΠ - ∫∂ₜ(ρΠ)$(n > 1 ? ", means over $smooth s" : "")", ylabel = "W m⁻¹")
        lines!(ax, slow(mid), slow(d_dt(I(key, :integrated_APE)) .- average(I(key, :dρAPE))); label = "APE")
        lines!(ax, slow(mid), slow(d_dt(I(key, :integrated_ACE)) .- average(I(key, :dρACE))); label = "ACE")
        lines!(ax, slow(mid), slow(d_dt(I(key, :integrated_kinetic_energy)) .- average(I(key, :dρK))); label = "error of d/dt (from ∫ρK)",
               color = :gray)
        axislegend(ax; position = :lb)
        # the integrals of the terms in ∂ₜP_*
        R_ACE, R_APE = I(key, :reference_pressure_term_ACE), I(key, :reference_pressure_term_APE)
        ax = Axis(figure[2, column]; title = "$label: terms in ∂ₜP_*", xlabel = "t [s]", ylabel = "W m⁻¹")
        lines!(ax, slow(times), slow(R_APE); label = "∫R_Π₂")
        lines!(ax, slow(times), slow(R_ACE); label = "∫R_Π₁")
        lines!(ax, slow(times), slow(R_ACE .+ R_APE); label = "∫R_Π", color = :black)
        axislegend(ax; position = :lb)
        # profiles of ∂ₜP_*: from the diabatic tendencies, and from centred differences of the outputs
        z = CFBoxes.cellgrid(model.domain, model.dz, 2)
        rows = CFBoxes.interior_cells(model.domain, 2)
        P, ∂ₜP = data[key][:p_star], data[key][:p_star_tendency]
        ax = Axis(figure[3, column]; title = "$label: ∂ₜP_* (lines) and differences of P_* (dots)", xlabel = "Pa s⁻¹", ylabel = "z [m]")
        for (c, k) in enumerate(round.(Int, range(2, lastindex(results.times) - 1, profiles)))
            color = Makie.wong_colors()[c]
            centred = (P[rows, k+1] .- P[rows, k-1]) ./ (results.times[k+1] - results.times[k-1])
            lines!(ax, ∂ₜP[rows, k], z[rows]; color, label = "t = $(round(results.times[k]; sigdigits = 4)) s")
            scatter!(ax, centred, z[rows]; color, markersize = 6)
        end
        axislegend(ax; position = :rb)
    end
    return figure
end

"""
    figure = available_energy_maps(results; slices = (first, middle, last output))

Maps of AE = Π, ACE = Π₁ and APE = Π₂ of every model at the outputs `slices`, a row per model and term, each on its own
colour range (Π₁ is far smaller than Π₂, and zero for AN2D).
"""
function available_energy_maps(results; slices = (1, cld(lastindex(results.times), 2), lastindex(results.times)))
    (; models, times) = results
    terms = ((:AE, "Π (AE)"), (:ACE, "Π₁ (ACE)"), (:APE, "Π₂ (APE)"))
    figure = Figure(size = (1500, 250 * length(models) * length(terms)))
    for (k, key) in enumerate(keys(models)), (l, (name, symbol)) in enumerate(terms)
        model = models[key]
        row = length(terms) * (k - 1) + l
        # the terms are on the cells, and nonnegative; a field that is zero (ACE for AN2D) is shown on the range (0, 1)
        x, z = diagnostic_coordinates(model, :conversion)
        largest = maximum(n -> maximum(field(results, key, name, n)), slices)
        limit = iszero(largest) ? one(largest) : largest
        Label(figure[row, 1], "$(nameof(typeof(model))): $symbol$(iszero(largest) ? " = 0" : "")"; rotation = π / 2,
              tellheight = false)
        for (c, n) in enumerate(slices)
            ax = Axis(figure[row, c + 1]; title = "t = $(times[n]) s", aspect = DataAspect())
            heatmap!(ax, x, z, field(results, key, name, n); colormap = :viridis, colorrange = (0, limit))
            hidedecorations!(ax)
        end
        Colorbar(figure[row, length(slices) + 2]; colormap = :viridis, limits = (0, limit), label = "$symbol [J kg⁻¹]")
    end
    return figure
end

"""
    print_summary(results; at_rest = nothing)

The closure of the global energy budget, the available energies at the start and end of the run (and `at_rest`, those of
a state at rest, a named tuple per model of `integrated_AE`, `integrated_ACE`, `integrated_APE`), and the checks of the
reference state.
"""
function print_summary(results; at_rest = nothing)
    (; models) = results
    I(key, name) = series_integral(results, key, name)
    println()
    for key in keys(models)
        scale = maximum(abs, I(key, :conversion))
        println("$(nameof(typeof(models[key]))): max |global residual| = $(maximum(abs, I(key, :integrated_budget_residual))) W m⁻¹, ",
                "max |∫C| = $scale W m⁻¹, max |∫VF| = $(maximum(abs, I(key, :integrated_vortex_work))) W m⁻¹")
        for (name, symbol) in ((:integrated_AE, "∫ρΠ (AE)"), (:integrated_ACE, "∫ρΠ₁ (ACE)"), (:integrated_APE, "∫ρΠ₂ (APE)"))
            rest = isnothing(at_rest) ? "" : " ($(at_rest[key][name]) J m⁻¹ at rest)"
            println("    $symbol = $(I(key, name)[1]) J m⁻¹ at t = 0, $(I(key, name)[end]) J m⁻¹ at the end$rest")
        end
        println("    largest number of parcels failing the certificate: $(Int(maximum(I(key, :certificate_violations)))), ",
                "at a wall: $(Int(maximum(I(key, :wall_minima)))) (of $(length(field(results, key, :AE, 1)))); ",
                "of slots with plain weights: $(Int(maximum(I(key, :plain_weight_slots))))")
        top = models[key] isa FC2D ? ", alternations with P_top: $(Int(maximum(I(key, :top_pressure_iterations))))" : ""
        println("    largest number of pivots: $(Int(maximum(I(key, :pivots)))), of passes over the pairs: ",
                "$(Int(maximum(I(key, :pricing_passes))))$top; underestimate of E_P - E_* by the sort: ",
                "$(maximum(I(key, :sorted_energy_excess))) J m⁻¹ at most, ",
                "$(maximum(I(key, :sorted_energy_excess) ./ I(key, :integrated_AE))) of ∫ρΠ at most")
        R = I(key, :reference_pressure_term_ACE) .+ I(key, :reference_pressure_term_APE)
        println("    max |∫R_Π| = $(maximum(abs, R)) W m⁻¹, max |∫R_Π₂| = $(maximum(abs, I(key, :reference_pressure_term_APE))) W m⁻¹")
    end
end
