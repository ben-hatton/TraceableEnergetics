include("../tools/preamble.jl")
cd("$(@__DIR__)")

# ClimFlows & extensions
using CFTimeSchemes
using ClimFluids
using CFAnelastic
using CFBoxes
using ManagedLoops: @vec, @with
using MutatingOrNot: Void, void
using CookBooks: CookBooks, open, close, CookBook

# Data handling
using JLD2
using NCDatasets
include("../tools/nc_tools.jl")

# Plot tools
using CairoMakie

# diagnostics
using BenchmarkTools: @btime

## LOAD PARAMS

exp_name = "test_512_1000"
exp_dir  = "./data/$exp_name"

# load params from setup
params = JLD2.load("$exp_dir/params.jld2")["params"];
model_parts = JLD2.load("$exp_dir/params.jld2")["model_parts"];
(; mgr, domain, space, fluid, advection_scheme, buoyancy_scheme, heatflux_scheme, viscosity_scheme, rho0) = model_parts;

# load ClimFlows model
anelastic_reference = AnelasticReference(; domain, space, density_profile = z -> rho0, ptop = 1e5)
boundary_conditions = BoundaryConditions2D( domain, (;  u = (; bottom = NeumannBC(0.), top = NeumannBC(0.)), w = (; bottom = DirichletBC(0.), top = DirichletBC(0.)), consvar = (; bottom = NeumannBC(0.), top = NeumannBC(0.)), q = (; bottom = NeumannBC(0.), top = NeumannBC(0.))))
model = CFAnelastic.AN2D((; mgr, domain, space, fluid, advection_scheme, buoyancy_scheme, heatflux_scheme, viscosity_scheme, anelastic_reference, boundary_conditions))

## LOAD DATA

# filenames for output and analysis datasets
nc_output   = "$exp_dir/output.nc"
nc_analysis = "$exp_dir/analysis.nc"

# create analysis dataset
ds = Dataset(nc_analysis, "c")

# copy data from output dataset to analysis dataset
Dataset(nc_output, "r") do ds_output
    # copy dimensions
    for (name, length) in ds_output.dim 
        defDim(ds, name, length)
    end
    # copy global attributes
    for (attname, attval) in ds_output.attrib
        ds.attrib[attname] = attval
    end
    # copy variables
    for var in NCDatasets.keys(ds_output)
        defVar(ds, ds_output[var])
    end
    close(ds_output);
end

# quick visualise
t = params.Nslice
slice = interior(domain, ds[:b][:,:,t])
nx, ny = size(slice)

fig = Figure(size=(300,600))
ax = Axis(fig[1, 1])
hm = Makie.heatmap!(ax, slice)
Colorbar(fig[1, 2], hm, width = 20)
colsize!(fig.layout, 1, Aspect(1, nx/ny))
resize_to_layout!(fig)
fig
    
## SORTED REFERENCE STATE SETUP

# This is a sorted reference state, where parcels are binned according to reference height after sorting
# This is not a PDF reference state, where parcels are binned according to buoyancy and not sorted, 
    
# number of bins
Nbins = 512

# domain dimensions
Mx, Mz = dims(domain)

# add time variable to dataset
haskey(ds, "t") || defVar(ds, "t", [params.slice_size * i for i in 1:params.Nslice], ("t",));   

# add m dimension to dataset
defDim(ds, "mi", 2)

# add reference state dimensions to dataset
defDim(ds, "zeta", Nbins)
defDim(ds, "Zeta", Nbins+1)
defDim(ds, "ij", Mx*Mz)

# add m to dataset
haskey(ds, "m")         || defVar(ds, "m", Float64, ("x", "z", "mi", "t"))

# add reference state variables to dataset
haskey(ds, "zeta")      || defVar(ds, "zeta", Float64, ("zeta",));          # height coordinate at centres [Float; len = Nbins]
haskey(ds, "Zeta")      || defVar(ds, "Zeta", Float64, ("Zeta",));          # height coordinate at edges [Float; len = Nbins+1]
haskey(ds, "perm")      || defVar(ds, "perm", Int, ("ij", "t"));            # permutation to sort in situ into reference [Int; len = Mx*Mz]
haskey(ds, "iperm")     || defVar(ds, "iperm", Int, ("ij", "t"));           # lists the position in the reference state of each parcel [Int; len = Mx*Mz]
haskey(ds, "ipermbin")  || defVar(ds, "ipermbin", Int, ("ij", "t"));        # lists the bin of each parcel [Int; len = Mx*Mz]
haskey(ds, "zref")      || defVar(ds, "zref", Float64, ("x", "z", "t"));    # reference height array [Float; size = (Mx+2Hx, Mz+2Hz, Nslice)]
haskey(ds, "dzref_x")   || defVar(ds, "dzref_x", Float64, ("X", "z", "t"));    
haskey(ds, "dzref_z")   || defVar(ds, "dzref_z", Float64, ("x", "Z", "t"));    
haskey(ds, "zrefdif")      || defVar(ds, "zrefdif", Float64, ("x", "z", "t"));    # reference height array [Float; size = (Mx+2Hx, Mz+2Hz, Nslice)]
haskey(ds, "dzrefdif_x")   || defVar(ds, "dzrefdif_x", Float64, ("X", "z", "t"));    
haskey(ds, "dzrefdif_z")   || defVar(ds, "dzrefdif_z", Float64, ("x", "Z", "t"));    
haskey(ds, "bref")      || defVar(ds, "bref", Float64, ("zeta", "t"));
haskey(ds, "brefz")     || defVar(ds, "brefz", Float64, ("z", "t"));
haskey(ds, "beta")      || defVar(ds, "beta", Float64, ("x", "z", "t"));
haskey(ds, "N2")        || defVar(ds, "N2", Float64, ("zeta", "t"));
haskey(ds, "N2ref")        || defVar(ds, "N2ref", Float64, ("x", "z", "t"));

# add centered variables
haskey(ds, "uc")        || defVar(ds, "uc", Float64, ("x", "z", "t"))
haskey(ds, "wc")        || defVar(ds, "wc", Float64, ("x", "z", "t"))

# energy terms
haskey(ds, "ke")     || defVar(ds, "ke", Float64, ("x", "z", "t"))
haskey(ds, "pe")     || defVar(ds, "pe", Float64, ("x", "z", "t"))


# add energy budget terms
haskey(ds, "betaw")     || defVar(ds, "betaw", Float64, ("x", "z", "t"))
haskey(ds, "bw")        || defVar(ds, "bw", Float64, ("x", "z", "t"))
haskey(ds, "epsk")      || defVar(ds, "epsk", Float64, ("x", "z", "t"))
haskey(ds, "epsp")      || defVar(ds, "epsp", Float64, ("x", "z", "t"))
haskey(ds, "epsp_tur")  || defVar(ds, "epsp_tur", Float64, ("x", "z", "t"))
haskey(ds, "epsp_lam")  || defVar(ds, "epsp_lam", Float64, ("x", "z", "t"))
haskey(ds, "sigmab")    || defVar(ds, "sigmab", Float64, ("x", "z", "t"))
haskey(ds, "jbx")       || defVar(ds, "jbx", Float64, ("X", "z", "t"))
haskey(ds, "jbz")       || defVar(ds, "jbz", Float64, ("x", "Z", "t"))
haskey(ds, "db_x")      || defVar(ds, "db_x", Float64, ("X", "z", "t"))
haskey(ds, "db_z")      || defVar(ds, "db_z", Float64, ("x", "Z", "t"))

# add averaged budget variables
haskey(ds, "jbzref")      || defVar(ds, "jbzref", Float64, ("zeta", "t"));
haskey(ds, "djbzref_zeta")      || defVar(ds, "djbzref_zeta", Float64, ("zeta", "t"));
haskey(ds, "dbref_t")      || defVar(ds, "dbref_t", Float64, ("zeta", "t"));


## SORTED REFERENCE STATE COMPUTATION

# load reference state variables as slices
zeta        = ds["zeta"]
Zeta        = ds["Zeta"]
perm        = ds["perm"]
iperm       = ds["iperm"]
ipermbin    = ds["ipermbin"]

# write zeta coordinates
Lz = model.space.Lz
zeta[:] = range(0.5 * Lz / Nbins, step = Lz / Nbins, length = Nbins)
Zeta[:] = range(0.0, step = Lz / Nbins, length = Nbins + 1)

# load buoyancy
b           = ds[:b]

# load zref array
zref        = ds["zref"]
dzref_x     = ds["dzref_x"]
dzref_z     = ds["dzref_z"]
zrefdif        = ds["zrefdif"]
dzrefdif_x     = ds["dzrefdif_x"]
dzrefdif_z     = ds["dzrefdif_z"]

# load bref array
bref        = ds["bref"]
brefz       = ds["brefz"]
beta        = ds["beta"]
N2          = ds["N2"]
N2ref          = ds["N2ref"]

# load state
m           = ds["m"]
s           = ds["s"]
q           = ds["q"]
u           = ds["u"]
w           = ds["w"]

# load centered variables
uc          = ds["uc"]
wc          = ds["wc"]

# energy
ke          = ds["ke"]
pe          = ds["pe"]

# load energy conversions
bw          = ds["bw"]
betaw       = ds["betaw"]
epsk        = ds["epsk"]
epsp        = ds["epsp"]
epsp_tur    = ds["epsp_tur"]
epsp_lam    = ds["epsp_lam"]
sigmab      = ds["sigmab"]
jbx         = ds["jbx"]
jbz         = ds["jbz"]
db_x         = ds["db_x"]
db_z         = ds["db_z"]

# load averaged budget variables
jbzref = ds["jbzref"]
djbzref_zeta = ds["djbzref_zeta"]
dbref_t = ds["dbref_t"]

# fill state (assuming Boussinesq ρ_R = rho0)
m[:, :, 1, :] = s[:, :, :] * rho0
m[:, :, 2, :] = q[:, :, :] * rho0

include("analysis_functions.jl")

# compute the reference state structure
compute_ref_state!((perm, iperm, ipermbin), (b, zeta, Zeta), model, del=1e-6)

# compute the field z*(x, z, t)
compute_zref!(zref, (iperm, zeta), model)

# compute z* - z
z_ = [zpoint(domain, model.dz, k) for i in 1:xdim(domain), k in 1:zdim(domain)]
for t in axes(zref, 3)
    zrefdif[:, :, t] = zref[:, :, t] .- z_
end

# compute ∇z*
compute_grad!((db_x, db_z), b, model)
compute_grad!((dzref_x, dzref_z), zref, model)
compute_grad!((dzrefdif_x, dzrefdif_z), zrefdif, model)

# compute b*(ζ, t) = ⟨b⟩(ζ, t)
isoaverage!(bref, b, ipermbin, Zeta, model)

# compute N² in reference state
compute_N2!((N2, N2ref), (bref, Zeta, ipermbin), model)

# compute β(x, z, t) = b(x, z, t) - b*(z, t)
# assumption: Nbins = Mz so can use same array for b*(z, t) and b*(ζ, t)
compute_beta!(beta, (b, bref), model)

# interpolate velocity fields onto centre
center_u!(uc, u, model)
center_w!(wc, w, model)

# compute APE <-> KE
betaw[:, :, :] = beta[:, :, :] .* wc[:, :, :];
bw[:, :, :] = b[:, :, :] .* wc[:, :, :];

# compute KE dissipation, buoyancy production and flux (assume no salt flux)
call_diagnostics!((epsk, sigmab, jbx, jbz, ke), (:ρε, :σ_cons, :Jcons_x, :Jcons_z, :ke), (m, u, w), model)
epsk[:, :, :] .*= inv(rho0)
sigmab[:, :, :] .*= space.g * fluid.α_T / rho0
jbx[:, :, :] .*= space.g * fluid.α_T / rho0
jbz[:, :, :] .*= space.g * fluid.α_T / rho0

# compute APE dissipation
compute_epsp!((epsp, epsp_tur, epsp_lam), (sigmab, jbx, jbz, dzrefdif_x, dzrefdif_z, zref), model)

# εₚ_turb = -jb⋅∇z*
isoaverage!(jbzref, -epsp_tur, ipermbin, Zeta, model)

# pe = - bz
compute_pe!(pe, b, model)


dzeta!(djbzref_zeta, jbzref, Zeta, model)

dt!(dbref_t, bref, params.slice_size)


function int_dzeta!(int_f, f, Zeta)
    print("Computing ∫fdζ...\n")

    dζ = Zeta[2] - Zeta[1]

    Nbins = size(f, 2)

    for i in 1:Nbins
        int_f[i, :] = sum(f[1:i, :], dims=1)
    end
    int_f *= dζ
end

# vol integral for array with halo
function volume_integral(domain, arr)
    int_i = xrange_interior(domain)
    int_j = zrange_interior(domain)
    arr_interior = @view arr[int_i, int_j, :]
    arr_integral = vec(sum(arr_interior, dims=(1,2)))
    return arr_integral
end

# cumulative time integral of time series
function time_integral(time_series, slice_size)
    return [sum(time_series[1:t])*slice_size for t in 1:length(time_series)]
end

# time derivative of time series
function time_derivative(time_series, slice_size)
    Nt = length(time_series)
    integral = Vector{Float64}(undef, Nt)
    for i in 1:Nt-1
        integral[i] = (time_series[i+1] - time_series[i])/slice_size
    end
    integral[Nt] = integral[Nt-1] 
    return integral
end

## integral quantities
int_dbref_t = Array{Float64}(undef, Nbins, params.Nslice)
int_dzeta!(int_dbref_t, dbref_t, Zeta)


# integrate βw
betaw_int = volume_integral(domain, betaw)

b_int = volume_integral(domain, b)
ke_int = volume_integral(domain, ke)
pe_int = volume_integral(domain, pe)
ke_int .-= ke_int[1]
pe_int .-= pe_int[1]

epsk_int = volume_integral(domain, epsk)
epsk_int_tint = time_integral(epsk_int, params.slice_size)

bw_int = volume_integral(domain, bw)

ke_int_t = time_derivative(ke_int, params.slice_size)

# next: 
# - prime of a field
# - energy diagnostics
# - computation of b*(z, t) - with Nbins = Mz should be the same as b*(zeta, t) but might want to generalise

##
# quick heatmap
t = params.Nslice
slice = interior(domain, ds[:pe][:,:,t])
nx, ny = size(slice)
fig = Figure(size=(300,600))
ax = Axis(fig[1, 1])
hm = Makie.heatmap!(ax, slice)
Colorbar(fig[1, 2], hm, width = 20)
colsize!(fig.layout, 1, Aspect(1, nx/ny))
resize_to_layout!(fig)
fig


# average plot in time
Xa = ds[:jbzref][:, :]
# Xa = int_dbref_t
fig = Figure(size=(600,600))
ax = Axis(fig[1, 1])
hm = Makie.heatmap!(ax, Xa')
Colorbar(fig[1, 2], hm, width = 20)
resize_to_layout!(fig)
fig

# quick lineplot against zeta
t = params.Nslice
line = ds[:bref][:, t]
zetavec = ds[:zeta][:]
fig = Figure(size=(300,600))
ax = Axis(fig[1, 1])
ln = Makie.lines!(ax, line, zetavec)
fig

# quick lineplot against time
tvec = [i for i in 1:params.Nslice]
tline = bw_int
fig = Figure(size=(600,400))
ax = Axis(fig[1, 1])
ln = Makie.lines!(ax, tvec, tline)
fig


# plot multiple lines
tvec = [i for i in 1:params.Nslice]
tlines = (ke_int_t, bw_int, epsk_int, ke_int_t .- bw_int .+epsk_int)
fig = Figure(size=(600,400))
ax = Axis(fig[1, 1])
for line in tlines
    Makie.lines!(ax, tvec, line)
end
fig


##

# # construct m from (s, q, u, w)
# ds, keys = construct_state!(ds, keys, model);



# CFAnelastic.available_energetics!(ds, keys, model, ref_state_type, (; params..., ae_variables))
# ## PLOTS
# (; Nslice, slice_size) = params
# tgrid = [it for it in 1:Nslice, j in 1:nbins]

# zeta_c = ds[:zeta_c][:,:]
# b_star = ds[:b_star][:,:]
# tke = ds[:tke][:,:]
# rke = ds[:rke][:,:]
# bw = ds[:bw][:,:]
# wa = ds[:wa][:,:]
# eps_k = ds[:eps_k][:,:]
# grad_z_star_a = ds[:grad_z_star_a][:,:]

# function flat_surface(field; title = "")
#     f = Figure()
#     ax = Axis(f[1, 1], title = title)

#     su = surface!(tgrid, zeta_c', zeros(size(field')), color=field', shading=NoShading)

#     tightlimits!(ax)

#     Colorbar(f[1, 2], su)
#     f
# end
# # function time_deriv(arr, dt)
# #     arr_dt = similar(arr)
# #     arr_dt[:,1] = (arr[:,2] .- arr[:,1]) / dt
# #     for i in 2:size(arr, 2)-1
# #         arr_dt[:,i] = (arr[:,i+1] .- arr[:,i-1]) / (2dt)
# #     end
# #     arr_dt[:,end] = (arr[:,end] .- arr[:,end-1]) / dt
# #     return arr_dt
# # end

# # drke_dt = time_deriv(rke, slice_size) 
# # dtke_dt = time_deriv(tke, slice_size) 

# flat_surface(rke, title="RKE")

# flat_surface(wa, title="<w>")
# flat_surface(b_star, title="b_star")
# flat_surface(bw, title="b*<w>")
# flat_surface(eps_k, title="εₖ")
# flat_surface(grad_z_star_a, title="|∇z*|")
# fig = Figure()
# ax = Axis(fig[1, 1])
# su = surface!(tgrid, zeta_c[64:196,:]', zeros(size(grad_z_star_a[64:196,:]')), color=grad_z_star_a[64:196,:]', shading=NoShading)
# tightlimits!(ax)
# Colorbar(fig[1, 2], su)
# fig

# lines(ds[:s][1,200:end,1])

# fig, ax, hm = heatmap(ds[:b][2:end-1,:,100])
# Colorbar(fig[1,2], hm)
# fig
# b = ds[:b][2:end-1,:,100]

# db = b[:,2:end] .- b[:,1:end-1]

# fig, ax, hm = heatmap(db)
# Colorbar(fig[1,2], hm)
# fig




# lines(ds[:b][1,200:end-1,10])
# lines(ds[:z_star][1,200:end-1,10])
# lines(ds[:grad_z_star][1,2:end-1,10])

# lines(ds[:w][100,:,1])
# xvec = [i*dx for i in 1:Mx, j in 1:Mz]
# zvec = [j*dz for i in 1:Mx, j in 1:Mz]

# Makie.heatmap()
# s_arr = θ_init.(xvec, zvec)
# lines(s_arr[2,1:50])

# lines(ds[:grad_z_star_a][50:150,1])

# lines(ds[:z_star][1,2:end-1,100])

# interior(domain, ds[:grad_z_star][:,:,100])

# ## TRADITIONAL ENERGETICS

# # choose which energetic variables to record
# energetics_variables = (
#     :ke, :pe, :te,                                          # kinetic, potential, total energy
#     :viscous_diss, :entropy_prod,                           # irreversible production terms
#     :conjdivJcons, :chemdivJq, :Jconsgradconj, :Jqgradchem, # flux divergence and gradient terms
#     :buoy_flux_Σ_ϕ, :buoy_flux_B,                           # buoyancy flux terms
#     :pdivu,                                                 # pressure work term
#     :divru, :conjdivrsu, :chemdivrqu                        # thermodynamic budget terms
#     );

# # run loop to compute and record energetics
# CFAnelastic.energetics(ds, keys, model, (; params..., energetics_variables));

# # entropy integral
# s_int = rintegral(ds, :s, model)

# # energy integrals
# ke_int = dintegral(ds, :ke, model)
# pe_int = dintegral(ds, :pe, model)
# te_int = dintegral(ds, :te, model)

# # energy derivatives
# function time_deriv(vec, dt)
#     vec_dt = similar(vec)
#     vec_dt[1] = (vec[2] - vec[1]) / dt
#     for i in 2:length(vec)-1
#         vec_dt[i] = (vec[i+1] - vec[i-1]) / (2dt)
#     end
#     vec_dt[end] = (vec[end] - vec[end-1]) / dt
#     return vec_dt
# end
# ∂ke_∂t_int = time_deriv(ke_int, ds.attrib["slice_size"])
# ∂pe_∂t_int = time_deriv(pe_int, ds.attrib["slice_size"])
# ∂te_∂t_int = time_deriv(te_int, ds.attrib["slice_size"])

# # budget integrals
# viscous_diss_int    = integral(ds, :viscous_diss, model)
# entropy_prod_int    = integral(ds, :entropy_prod, model)
# conjdivJcons_int    = integral(ds, :conjdivJcons, model)
# chemdivJq_int       = integral(ds, :chemdivJq, model)
# Jconsgradconj_int   = integral(ds, :Jconsgradconj, model)
# Jqgradchem_int      = integral(ds, :Jqgradchem, model)
# buoy_flux_Σ_ϕ_int   = integral(ds, :buoy_flux_Σ_ϕ, model)
# buoy_flux_B_int     = integral(ds, :buoy_flux_B, model)
# pdivu_int           = integral(ds, :pdivu, model)
# divru_int           = integral(ds, :divru, model)
# conjdivrsu_int      = integral(ds, :conjdivrsu, model)
# chemdivrqu_int      = integral(ds, :chemdivrqu, model)


# ## PLOTTING

# # animation
# animate(ds, :s, domain; fixed_clims=true)

# # energy integrals
# Plots.plot(ke_int,  label="KE")
# Plots.plot!(pe_int, label="PE") 
# Plots.plot(te_int,  label="TE")

# # energy derivatives
# Plots.plot(∂ke_∂t_int,  label="∂KE/∂t")
# Plots.plot!(∂pe_∂t_int, label="∂PE/∂t") 
# Plots.plot!(∂te_∂t_int,  label="∂TE/∂t")

# # viscous dissipation and entropy production
# Plots.plot!(viscous_diss_int, label="ρε")
# Plots.plot!(entropy_prod_int, label="πσ_s")

# # pressure work
# Plots.plot(pdivu_int, label="p∇.u")

# # flux terms
# Plots.plot(-conjdivJcons_int, label="π∇.Js")
# Plots.plot!(Jconsgradconj_int, label="Js.∇π")
# Plots.plot!(chemdivJq_int, label="μ∇.Jq")
# Plots.plot!(Jqgradchem_int, label="Jq.∇μ")

# Plots.plot(buoy_flux_Σ_ϕ_int, label="ρ_R.Σ_ϕ.g.w")
# Plots.plot!(-buoy_flux_B_int, label="-ρ_R.B.g.w")

# # thermodynamic budget terms
# Plots.plot(divru_int, label="∇.(ρu)")
# Plots.plot(conjdivrsu_int, label="π∇.(ρsu)")
# Plots.plot(chemdivrqu_int, label="μ∇.(ρqu)")

# ## AVAILABLE ENERGETICS
# CFAnelastic.available_energetics(ds, keys, model, SortedLorenzState())


# ## Test

# Plots.heatmap(interior(domain, ds["s"][:, :, 2])')

# ds["s"][:, :, 2]