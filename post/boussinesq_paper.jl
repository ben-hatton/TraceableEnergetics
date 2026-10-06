include("../inc/preamble.jl")
cd("$(@__DIR__)")

# ClimFlows & extensions
using ClimFluids
using CFAnelastic
using CFBoxes
using CFPlanets: Tank2D
using CFDiffusionSchemes
using CFTimeSchemes

# Data handling
using NCDatasets

# Plot tools
using Plots
include("plots.jl")
include("integrals.jl")

## Reconstruct model

Pr = 10.                                        # Prandtl number = ν / k_T
Le = 100.                                       # Lewis number = k_T / k_q

consvar     = :potential_temperature            # choice of prognostic conservative variable
rho0        = 1000.                             # Boussinesq reference density [kg.m⁻³]
kin_visc    = 1e-5                              # kinematic viscosity [m²s⁻¹]

dT = 5.
Tc = 273. +5.                                   # cool temperature
Th = Tc + dT                                    # hot temparature
T0 = Tc + dT/2                                  # background temperature
q0 = 30e-3

domain = Box2D(Mx = 256, Mz = 128, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))

space  = Tank2D(Lx = 10.0, Lz = 2.5, g = 10.)    

viscosity_scheme = ViscosityScheme(dyn_visc = rho0 * kin_visc, bulk_visc = 0.)

heatflux_scheme_new  = HeatFluxConsistent(k_T = kin_visc * Pr, k_q = 0.)
heatflux_scheme_old  = HeatFluxSimple(k_T = kin_visc * Pr, k_q = 0.)

buoyancy_scheme_new  = BuoyDynNew()
buoyancy_scheme_old  = BuoyDynOld()

advection_scheme = AdvEnergyCons()

fluid = NonlinearBinaryFluid( consvar, (; 
    p0      = 1e5,                              # reference pressure [Pa]
    T0      = T0,                            # reference temperature [K]
    q0      = q0,                          # reference composition concentration [kg/kg !!!]
    Cp      = 4000.0,                           # specific heat capacity [Jkg⁻¹K⁻¹]
    v0      = 0.001,                            # reference specific volume [m³kg⁻¹]
    α_T     = 0.001,                        # (first) thermal expansion coefficient [K⁻¹]
    α_q     = 0.0,                              # haline contraction cooefficient [(kg/kg)⁻¹] 0.001
    α_p     = 0.,                           # compressibility coefficient ( = v0 / cₛ₀² ) [ms²kg⁻¹] 1e-8
    M_s     = 3.14038218e-2,                    # mole-weighted average atomic weight [kg.mol⁻¹]
    R       = 8.31446261815324,                 # gas constant [J.K⁻¹.mol⁻¹]
    α_TT    = 0.,                               # second thermal expansion coefficient [K⁻²] 1.0e-4
    γ       = 0.,                               # thermobaric parameter [Pa⁻¹] 1.0e-7
))

# Anelastic reference state (top is z = 0)
anelastic_reference = AnelasticReference(;
    domain, 
    space, 
    density_profile = z -> rho0, 
    ptop = 1e5
)

# Boundary conditions for (u, w, s, q)
u_bc    = (; bottom = NeumannBC(0.), top = NeumannBC(0.))
w_bc    = (; bottom = DirichletBC(0.), top = DirichletBC(0.))
s_bc    = (; bottom = NeumannBC(0.), top = DirichletBC(x -> Tc + 0.5 * (Th - Tc) * ( 1 + (cos(π * x / space.Lx))^2)))
q_bc    = (; bottom = NeumannBC(0.), top = NeumannBC(0.))

boundary_conditions = BoundaryConditions2D(
    domain, 
    (;  u = u_bc, 
        w = w_bc,
        consvar = s_bc, 
        q = q_bc )
)
    
# Build model
model_new = AN2D(
    (; mgr = tSIMD(), 
    domain, 
    space, 
    fluid, 
    advection_scheme, 
    buoyancy_scheme = buoyancy_scheme_new, 
    heatflux_scheme = heatflux_scheme_new, 
    viscosity_scheme, 
    anelastic_reference,
    boundary_conditions )
)

model_old = AN2D(
    (; mgr = tSIMD(), 
    domain, 
    space, 
    fluid, 
    advection_scheme, 
    buoyancy_scheme = buoyancy_scheme_old, 
    heatflux_scheme = heatflux_scheme_old, 
    viscosity_scheme, 
    anelastic_reference,
    boundary_conditions )
)

# Time integration
u_max = sqrt(space.g * fluid.α_T * (Th - Tc) * space.Lx)
dt_max = max(model_new.dx, model_new.dz) / u_max

time_scheme  = CFTimeSchemes.RungeKutta4(model_new)
time_scheme  = CFTimeSchemes.RungeKutta4(model_new)
time_parameters = (; 
    Nslice = 1000,
    slice_size = 1.0,
    cfl = 0.5, 
    dt_max = dt_max, 
)

## LOAD DATA
exp_dir = "../../experiments/BQ"
exp_name_new = "HC_128_new"
exp_name_old = "HC_256_param_tweak_old"
data_file_new = "$exp_dir/$exp_name_new/data/fields.nc"
data_file_old = "$exp_dir/$exp_name_old/data/fields.nc"

ds_new = NCDatasets.NCDataset(data_file_new, "a")
ds_old = NCDatasets.NCDataset(data_file_old, "a")
keys = (; s = :s, q = :q, u = :u, w = :w)

# construct m from (s, q, u, w)
ds_new, keys = construct_state!(ds_new, keys, model_new);
ds_old, keys = construct_state!(ds_old, keys, model_old);

## TRADITIONAL ENERGETICS

# choose which energetic variables to record
energetics_variables = (
    :ke, :pe, :te,          # kinetic, potential, total energy
    :gpe, :ie, :gpe_bz,            # Boussinesq energies as per Tailleux et al. (2026)
    :Jcons_x, :Jcons_z
    );

# run loop to compute and record energetics
CFAnelastic.energetics(ds_new, keys, model_new, (; time_parameters..., energetics_variables));
CFAnelastic.energetics(ds_old, keys, model_old, (; time_parameters..., energetics_variables));

# entropy integral
s_int_new = rintegral(ds_new, :s, model_new)
s_int_old = rintegral(ds_old, :s, model_old)

# energy integrals
ke_int_new = dintegral(ds_new, :ke, model_new)
ke_int_old = dintegral(ds_old, :ke, model_old)
pe_int_new = dintegral(ds_new, :pe, model_new)
pe_int_old = dintegral(ds_old, :pe, model_old)
te_int_new = dintegral(ds_new, :te, model_new)
te_int_old = dintegral(ds_old, :te, model_old)
gpe_int_new = dintegral(ds_new, :gpe, model_new)
gpe_int_old = dintegral(ds_old, :gpe, model_old)
gpe_bz_int_new = dintegral(ds_new, :gpe_bz, model_new)
gpe_bz_int_old = dintegral(ds_old, :gpe_bz, model_old)
ie_int_new = dintegral(ds_new, :ie, model_new)
ie_int_old = dintegral(ds_old, :ie, model_old)


## PLOTTING

# animation
animate(ds_new, :s, domain; fixed_clims=true, t_step = 20)
animate(ds_old, :s, domain; fixed_clims=true, t_step = 20)

# conservative variable integral
Plots.plot(s_int_new,  label="Consvar")
Plots.plot(s_int_old,  label="Consvar")

# energy integrals
Plots.plot(ke_int_new,  label="KE")
Plots.plot!(ke_int_old,  label="KE")
Plots.plot(gpe_bz_int_new, label="PE") 
Plots.plot!(gpe_bz_int_old, label="PE") 
Plots.plot(te_int,  label="TE")

Plots.plot(pe_int, label="PE") 
Plots.plot(gpe_int, label="GPE")
Plots.plot(gpe_bz_int_new, label="GPE")
Plots.plot(gpe_bz_int_old, label="GPE")
Plots.plot(ie_int, label="IE")