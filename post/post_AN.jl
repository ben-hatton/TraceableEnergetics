include("../inc/preamble.jl")
cd("$(@__DIR__)")

# ClimFlows & extensions
using CFTimeSchemes
using ClimFluids
using CFAnelastic
using CFBoxes

# Data handling
using JLD2
using NCDatasets

# Plot tools
using Plots
include("plots.jl")
include("integrals.jl")

## LOAD DATA
exp_name = "shear_highres"
exp_dir = "../../experiments/AN"

# params  = JLD2.load("$exp_dir/$exp_name/data/vars.jld2")["params"];
# model   = JLD2.load("$exp_dir/$exp_name/data/vars.jld2")["model"];
(; domain, reference) = model;

ds = NCDatasets.NCDataset(params.data_file, "a")
keys = (; s = :s, q = :q, u = :u, w = :w)

# construct m from (s, q, u, w)
ds, keys = construct_state!(ds, keys, model);

## TRADITIONAL ENERGETICS

# choose which energetic variables to record
energetics_variables = (
    :ke, :pe, :te,                                          # kinetic, potential, total energy
    :viscous_diss, :entropy_prod,                           # irreversible production terms
    :conjdivJcons, :chemdivJq, :Jconsgradconj, :Jqgradchem, # flux divergence and gradient terms
    :buoy_flux_Σ_ϕ, :buoy_flux_B,                           # buoyancy flux terms
    :pdivu,                                                 # pressure work term
    :divru, :conjdivrsu, :chemdivrqu                        # thermodynamic budget terms
    );

# run loop to compute and record energetics
CFAnelastic.energetics(ds, keys, model, (; params..., energetics_variables));

# entropy integral
s_int = rintegral(ds, :s, model)

# energy integrals
ke_int = dintegral(ds, :ke, model)
pe_int = dintegral(ds, :pe, model)
te_int = dintegral(ds, :te, model)

# energy derivatives
function time_deriv(vec, dt)
    vec_dt = similar(vec)
    vec_dt[1] = (vec[2] - vec[1]) / dt
    for i in 2:length(vec)-1
        vec_dt[i] = (vec[i+1] - vec[i-1]) / (2dt)
    end
    vec_dt[end] = (vec[end] - vec[end-1]) / dt
    return vec_dt
end
∂ke_∂t_int = time_deriv(ke_int, ds.attrib["slice_size"])
∂pe_∂t_int = time_deriv(pe_int, ds.attrib["slice_size"])
∂te_∂t_int = time_deriv(te_int, ds.attrib["slice_size"])

# budget integrals
viscous_diss_int    = integral(ds, :viscous_diss, model)
entropy_prod_int    = integral(ds, :entropy_prod, model)
conjdivJcons_int    = integral(ds, :conjdivJcons, model)
chemdivJq_int       = integral(ds, :chemdivJq, model)
Jconsgradconj_int   = integral(ds, :Jconsgradconj, model)
Jqgradchem_int      = integral(ds, :Jqgradchem, model)
buoy_flux_Σ_ϕ_int   = integral(ds, :buoy_flux_Σ_ϕ, model)
buoy_flux_B_int     = integral(ds, :buoy_flux_B, model)
pdivu_int           = integral(ds, :pdivu, model)
divru_int           = integral(ds, :divru, model)
conjdivrsu_int      = integral(ds, :conjdivrsu, model)
chemdivrqu_int      = integral(ds, :chemdivrqu, model)


## PLOTTING

# animation
animate(ds, :s, domain; fixed_clims=true)

# energy integrals
Plots.plot(ke_int,  label="KE")
Plots.plot!(pe_int, label="PE") 
Plots.plot(te_int,  label="TE")

# energy derivatives
Plots.plot(∂ke_∂t_int,  label="∂KE/∂t")
Plots.plot!(∂pe_∂t_int, label="∂PE/∂t") 
Plots.plot!(∂te_∂t_int,  label="∂TE/∂t")

# viscous dissipation and entropy production
Plots.plot!(viscous_diss_int, label="ρε")
Plots.plot!(entropy_prod_int, label="πσ_s")

# pressure work
Plots.plot(pdivu_int, label="p∇.u")

# flux terms
Plots.plot(-conjdivJcons_int, label="π∇.Js")
Plots.plot!(Jconsgradconj_int, label="Js.∇π")
Plots.plot!(chemdivJq_int, label="μ∇.Jq")
Plots.plot!(Jqgradchem_int, label="Jq.∇μ")

Plots.plot(buoy_flux_Σ_ϕ_int, label="ρ_R.Σ_ϕ.g.w")
Plots.plot!(-buoy_flux_B_int, label="-ρ_R.B.g.w")

# thermodynamic budget terms
Plots.plot(divru_int, label="∇.(ρu)")
Plots.plot(conjdivrsu_int, label="π∇.(ρsu)")
Plots.plot(chemdivrqu_int, label="μ∇.(ρqu)")

## AVAILABLE ENERGETICS
CFAnelastic.available_energetics(ds, keys, model, SortedLorenzState())


## Test

Plots.heatmap(interior(domain, ds["s"][:, :, 2])')

ds["s"][:, :, 2]