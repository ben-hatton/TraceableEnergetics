include("../inc/preamble.jl")
cd("$(@__DIR__)")

# ClimFlows modules
using CFTimeSchemes
using ClimFluids
using CFPlanets: Tank2D
using CFBoxes
using CFDiffusionSchemes

# Local modules
using CFAnelastic

# Other modules
using JLD2
using BenchmarkTools: @btime


# Data/plot handling
include("../inc/data_processing.jl")

# Defining parameters
Pr = 10.                                        # Prandtl number = ν / k_T
Le = 100.                                       # Lewis number = k_T / k_q

consvar     = :potential_temperature            # choice of prognostic conservative variable
rho0        = 1000.                             # Boussinesq reference density [kg.m⁻³]
kin_visc    = 1e-4                              # kinematic viscosity [m²s⁻¹]

dT = 5.
Tc = 273. +5.                                   # cool temperature
Th = Tc + dT                                    # hot temparature
T0 = Tc + dT/2                                  # background temperature
q0 = 30e-3

    
# Ra = space.g * fluid.α_T * space.Lx^3 / (kin_visc * heatflux_scheme.k_T)


# Numerical choices
mgr             = tSIMD()                       # multi-threaded SIMD manager
TimeScheme      = CFTimeSchemes.RungeKutta4     # time integration scheme: RungeKutta4
BuoyancyScheme  = BuoyDynNew                    # buoyancy scheme: BuoyThermo, BuoyDynNew, BuoyDynOld
AdvectionScheme = AdvEnergyCons                 # advection scheme: AdvEnergyCons, AdvEnstrophyCons

# Physical choices
Model           = AN2D                          # model: AN2D
Fluid           = NonlinearBinaryFluid          # fluid model: NonlinearBinaryFluid
HeatFluxScheme  = HeatFluxConsistent            # heat flux closure: HeatFluxConsistent, HeatFluxSimple

# Numerical domain
domain = Box2D(
    Mx = 128, Mz = 64,                          # grid size 
    Hx = 1, Hz = 1,                             # halo size
    boundary = (Periodic, Bounded)              # boundary topology [only (Periodic, Bounded) currently supported]
)

# Physical space
space  = Tank2D(
    Lx = 10.0, Lz = 2.5,                       # spatial extent 
    g = 10.                                     # gravitational acceleration
)    

# Viscosity scheme
viscosity_scheme = ViscosityScheme( 
    dyn_visc = rho0 * kin_visc,                 # dynamic viscosity
    bulk_visc = 0.                              # bulk viscosity
)

# Heat flux scheme
heatflux_scheme  = HeatFluxScheme( 
    k_T = kin_visc * Pr,                        # thermal diffusivity
    k_q = 0. #kinematic_visc / Le,              # compositional diffusivity
)

# Numerical schemes
buoyancy_scheme  = BuoyancyScheme()
advection_scheme = AdvectionScheme()

# Fluid
fluid = Fluid( consvar, (; 
    p0      = 1e5,                              # reference pressure [Pa]
    T0      = T0,                            # reference temperature [K]
    q0      = q0,                          # reference composition concentration [kg/kg !!!]
    Cp0     = 4000.0,                           # specific heat capacity at p0 [Jkg⁻¹K⁻¹]
    v0      = 0.001,                            # reference specific volume [m³kg⁻¹]
    α_T     = 0.002,                        # (first) thermal expansion coefficient [K⁻¹]
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
u_bc    = (; bottom = NeumannBC(0.),
             top = NeumannBC(0.),
)
w_bc    = (; bottom = DirichletBC(0.),
             top = DirichletBC(0.),
)
s_bc    = (; bottom = NeumannBC(0.),
            #  top = DirichletBC(x -> Tc + 0.5 * (Th - Tc) * ( 1 + (cos(π * x / space.Lx))^2)),
             top = DirichletBC(x -> T0 - 0.5 * dT * cos(2π * x / space.Lx)),
)
q_bc    = (; bottom = NeumannBC(0.),
             top = NeumannBC(0.),
)

boundary_conditions = BoundaryConditions2D(
    domain, 
    (;  u = u_bc, 
        w = w_bc,
        consvar = s_bc, 
        q = q_bc )
)
    
# Build model
model = Model(
    (; mgr, 
    domain, 
    space, 
    fluid, 
    advection_scheme, 
    buoyancy_scheme, 
    heatflux_scheme, 
    viscosity_scheme, 
    anelastic_reference,
    boundary_conditions )
)

# Initial conditions
initial_conditions = (; 
    u       = (x, z) -> 0.,
    w       = (x, z) -> 0.,
    consvar = (x, z) -> T0,
    q       = (x, z) -> q0
)

# Time integration
u_max = sqrt(space.g * fluid.α_T * (Th - Tc) * space.Lx)
dt_max = max(model.dx, model.dz) / u_max

time_scheme  = TimeScheme(model)
time_parameters = (; 
    Nslice = 1000,
    slice_size = 10.0,
    cfl = 0.5, 
    dt_max = dt_max, 
)

# Data parameters
exp_name = "HC_128_new"
exp_model = "BQ"
exp_dir  = "../../experiments/$exp_model"
data_file = "$exp_dir/$exp_name/data/fields.nc"
data_parameters = (; 
    exp_name,                                       # experiment name,
    exp_model,                                      # experiment model,
    exp_dir,                                        # experiment directory
    data_file,                                      # file to save data
    saved_variables     = (:s, :q, :u, :w),         # variables to save
    plotted_variable    = :s,                       # variable to plot
)
params = (; time_parameters..., data_parameters...)

# Save parameters
# save("$exp_dir/$exp_name/data/vars.jld2", Dict("model" => model, "params" => params));

# Prepare data objects
write_func  = write_func                            # function to write data
plot_func   = plot_func                             # function to plot data
write_obj   = prepare_ds(params, model, Float64)    # prepare dataset
plot_obj    = void                                  # prepare plot object

# Run simulation
CFAnelastic.loop(
    model,
    initial_conditions,
    time_scheme,
    params;
    # plot_func   = plot_func,
    # plot_obj    = plot_obj,
    write_func  = write_func,
    write_obj   = write_obj,
)