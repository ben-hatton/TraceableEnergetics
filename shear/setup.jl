include("../tools/preamble.jl")
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
using SpecialFunctions: erf
using Statistics
using LinearAlgebra


# Data/plot handling
include("../tools/data_processing.jl")

"""
2D Kelvin-Helmholtz instablity
"""

exp_name = "test_512_1000"    # file name
Nslice  = 500                   # number of time slices


## Params

# NONDIMENSIONAL CHOICES

Re = 2097.          # = U.L/ν                   Reynolds number
Ri = 0.1            # = gαΔθL/U²                bulk Richardson number
Pr = 1.             # = ν/k                     Prandtl number
Ec = 2e-5           # = U^2 / cₚΔθ              Eckert number

δ_xz     = 25/14.   # = Lz / Lx                 aspect ratio
δ_xshear = 14       # = Lx / L_shear            ratio of domain width to shear size

# DIMENSIONAL CHOICES

L_shear = 10.           # shear interface thickness [m]
U       = 1.            # velocity shear [m/s]

g       = 10.           # gravitational acceleration [m.s⁻²]
α       = 0.0001        # thermal expansion coefficient [K⁻¹]

rho0    = 1000.         # Boussinesq reference density [kg.m⁻³]
T0      = 273.          # bottom temperature [K]
q0      = 35.           # salinity (arbitrary)

# DERIVED QUANTITIES

Lx = δ_xshear * L_shear     # domain width [m]
Lz = Lx * δ_xz              # domain height [m]
δ_z =  Lz / L_shear         # ratio of shear layer to domain height

T = L_shear / U             # characteristic time scale [s]
T_adv = Lx / U              # advection time scale [s]

ν   = U * L_shear / Re      # kinematic viscosity [m²/s]
k_T = ν / Pr                # heat diffusivity [m²/s]

ΔT = Ri * U^2 / (g * α * L_shear) # initial temperature step [K]

Cp = U^2 / (Ec * ΔT)          # specific heat capacity [Jkg⁻¹K⁻¹]

# N = g * α * ΔT / L_shear
# c_igw = N * max(Lx, Lz) / (2 * pi)


# NUMERICS

Mz  = 512                   # vertical resolution
Mx  = floor(Int, Mz / δ_xz) # horizontal resolution

dz = Lz / Mz
dx = Lx / Mx

slice_size = T_adv/20          # time step for data recording


# INITIAL CONDITION

k_instab = 2 * pi / Lx
amp = 0.01

f(z) = (1 + erf((z - Lz/2) / (L_shear))) / 2  # erf profile (from 0 to 1)
envelope(z) = exp(-((z - Lz/2) / (L_shear))^2)

u_shear(z) = - U/2 + U * f(z)    # initial velocity [m/s]
θ_init(x, z) = T0 + ΔT * f(z)       # initial potential temperature [K]

# STABILITY ANALYSIS
# include("./stability/stab_funcs.jl")
# check_stability(Lx, Lz, dz, ν, k_T, z -> g * α * (θ_init(0, z) - T0), z -> u_shear(z), 2, 30; plot=false)

## Build model & run

# Numerical choices
consvar         = :potential_temperature        # choice of prognostic conservative variable
mgr             = tSIMD()                       # multi-threaded SIMD manager
TimeScheme      = CFTimeSchemes.RungeKutta4     # time integration scheme: RungeKutta4
BuoyancyScheme  = BuoyDynNew                    # buoyancy scheme: BuoyThermo, BuoyDynNew, BuoyDynOld
AdvectionScheme = AdvEnergyCons                 # advection scheme: AdvEnergyCons, AdvEnstrophyCons

# Physical choices
Model           = AN2D                          # model: AN2D
Fluid           = NonlinearBinaryFluid          # fluid model: NonlinearBinaryFluid
HeatFluxScheme  = HeatFluxSimple            # heat flux closure: HeatFluxConsistent, HeatFluxSimple

# Numerical domain
domain = Box2D(
    Mx = Mx, Mz = Mz,                           # grid size 
    Hx = 1, Hz = 1,                             # halo size
    boundary = (Periodic, Bounded)              # boundary topology [only (Periodic, Bounded) currently supported]
)

# Physical space
space  = Tank2D(
    Lx = Lx, Lz = Lz,                           # spatial extent 
    g = g                                       # gravitational acceleration
)    

# Viscosity scheme
viscosity_scheme = ViscosityScheme( 
    dyn_visc = rho0 * ν,                        # dynamic viscosity
    bulk_visc = 0.                              # bulk viscosity
)

# Heat flux scheme
heatflux_scheme  = HeatFluxScheme( 
    k_T = k_T,                                  # thermal diffusivity
    k_q = 0.,                                   # compositional diffusivity
)

# Numerical schemes
buoyancy_scheme  = BuoyancyScheme()
advection_scheme = AdvectionScheme()

# Fluid
fluid = Fluid( consvar, (; 
    p0      = 1e5,                              # reference pressure [Pa]
    T0      = T0,                               # reference temperature [K]
    q0      = q0,                               # reference composition concentration [kg/kg !!!]
    Cp      = Cp,                           # specific heat capacity [Jkg⁻¹K⁻¹]
    v0      = 1/rho0,                           # reference specific volume [m³kg⁻¹]
    α_T     = α,                                # thermal expansion coefficient [K⁻¹]
    α_q     = 0.,                                # haline contraction cooefficient [(kg/kg)⁻¹] 0.001
    α_p     = 0.,                               # compressibility coefficient ( = v0 / cₛ₀² ) [ms²kg⁻¹] 1e-8
    M_s     = 3.14038218e-2,                    # mole-weighted average atomic weight [kg.mol⁻¹]
    R       = 8.31446261815324,                 # gas constant [J.K⁻¹.mol⁻¹]
    α_TT    = 0.,                                # second thermal expansion coefficient [K⁻²] 1.0e-4
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
             top = NeumannBC(0.),
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

stream = alloc_XZ(Float64, domain)
u_arr, w_arr = alloc_Xz(Float64, domain), alloc_xZ(Float64, domain)
for I in axes(stream, 1), J in axes(stream, 2)
    X, Z = Xpoint(domain, dx, I), Zpoint(domain, dz, J)
    stream[I, J] = amp * U * envelope(Z) * cos(k_instab * X) / k_instab
end
for I in Xrange(domain), j in zrange(domain)
    z = zpoint(domain, dz, j)
    u_arr[I, j] = dif_z(stream, I, j) * inv(dz) + u_shear(z)
end
for i in xrange(domain), J in Zrange(domain)
    w_arr[i, J] = - dif_x(stream, i, J) * inv(dx)
end

initial_conditions = (; 
    u       = u_arr,
    w       = w_arr,
    consvar = θ_init,
    q       = (x, z) -> q0
)

# Time integration
dt_max = max(model.dx, model.dz) / U
time_scheme  = TimeScheme(model)
time_parameters = (; 
    Nslice = Nslice,
    slice_size = slice_size,
    cfl = 0.05, 
    dt_max = dt_max, 
)

# Data parameters
exp_model = "simple Boussinesq"
exp_dir  = "./data/$exp_name"
data_file = "$exp_dir/output.nc"
data_parameters = (; 
    exp_name,                                       # experiment name,
    exp_model,                                      # experiment model,
    exp_dir,                                        # experiment directory
    data_file,                                      # file to save data
    saved_variables     = (:s, :q, :b, :u, :w),         # variables to save
    plotted_variable    = :b,                       # variable to plot
)
params = (; time_parameters..., data_parameters...)

# Save parameters
save_dict = Dict(   "model_parts" => (; mgr, domain, space, fluid, advection_scheme, buoyancy_scheme, heatflux_scheme, viscosity_scheme, rho0),
        "params" => params)
save("$exp_dir/params.jld2", save_dict);



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