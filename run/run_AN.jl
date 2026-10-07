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


    ## Numerical choices
    mgr             = tSIMD()                    # multi-threaded SIMD manager
    TimeScheme      = CFTimeSchemes.RungeKutta4  # time integration scheme: RungeKutta4
    BuoyancyScheme  = BuoyThermo                 # buoyancy scheme: BuoyThermo, BuoyDynNew, BuoyDynOld
    AdvectionScheme = AdvEnergyCons              # advection scheme: AdvEnergyCons, AdvEnstrophyCons


    ## Physical choices
    Model           = AN2D                        # model: AN
    Fluid           = NonlinearBinaryFluid      # fluid model: NonlinearBinaryFluid
    HeatFluxScheme  = HeatFluxConsistent        # heat flux closure: HeatFluxConsistent, HeatFluxSimple


    ## Numerical domain
    domain = Box2D(
        Mx = 128, Mz = 128,               # grid size 
        Hx = 1, Hz = 1,                 # halo size
        boundary = (Periodic, Bounded)  # boundary topology [only (Periodic, Bounded) currently supported]
    )


    ## Physical space
    space  = Tank2D(
        Lx = 10., Lz = 10.,     # spatial extent 
        g = 10.                 # gravitational acceleration
    )    


    ## Heat flux scheme
    heatflux_scheme  = HeatFluxScheme( 
        k_T = 0.001,     # thermal diffusivity
        k_q = 0.0       # compositional diffusivity
    )


    ## Viscosity scheme
    viscosity_scheme = ViscosityScheme( 
        dyn_visc = 1.0, # dynamic viscosity
        bulk_visc = 0.  # bulk viscosity
    )


    ## Fluid
    consvar = :potential_temperature    # choice of conservative variable
    fluid = Fluid( consvar, (; 
        p0      = 100000.0,             # reference pressure [Pa]
        T0      = 300.0,                # reference temperature [K]
        q0      = 30.0e-3,              # reference composition concentration [g/kg !!!]
        Cp0     = 4000.0,               # specific heat capacity at p0 [Jkg⁻¹K⁻¹]
        v0      = 0.001,                # reference specific volume [m³kg⁻¹]
        α_T     = 0.0001,               # (first) thermal expansion coefficient [K⁻¹]
        α_q     = 0.0,                  # haline contraction cooefficient [(g/kg)⁻¹] 0.001
        α_p     = 1e-8,                  # compressibility coefficient ( = v0 / cₛ₀² ) [ms²kg⁻¹] 1e-8
        M_s     = 3.14038218e-2,        # mole-weighted average atomic weight [kg.mol⁻¹]
        R       = 8.31446261815324,     # gas constant [J.K⁻¹.mol⁻¹]
        α_TT    = 0.,                   # second thermal expansion coefficient [K⁻²] 1.0e-4
        γ       = 0.,                   # thermobaric parameter [Pa⁻¹] 1.0e-7
    ))


    ## Anelastic reference state (top => z = 0)
    include("../inc/profiles.jl")
    density_profile(z)  = exp_profile(z, space, fluid)
    anelastic_reference = AnelasticReference(
        domain, 
        space, 
        density_profile, 
        (; ρtop = 1e3, ptop = 1e5))

    ## Numerical schemes
    buoyancy_scheme  = BuoyancyScheme()
    advection_scheme = AdvectionScheme()


    ## Boundary conditions for (u, w, s, q) or (u, w, T, q)
    u_bc    = (; bottom = NeumannBC(0.),
                top = NeumannBC(0.),
    )
    w_bc    = (; bottom = DirichletBC(0.),
                top = DirichletBC(0.),
    )
    T_bc    = (; bottom = NeumannBC(0.),
                top = NeumannBC(0.),
    )
    q_bc    = (; bottom = NeumannBC(0.),
                top = NeumannBC(0.),
    )

    boundary_conditions = BoundaryConditions2D(
        domain, 
        (;  u = u_bc, 
            w = w_bc,
            T = T_bc, 
            q = q_bc )
    )
        
    # to do: implement initial conditions properly
    ## Initial conditions
    # initial_conditions = KelvinHelmholtz(domain, space)


    ## Build model
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


    ## Initial conditions (needs improvement)
    initial_parameters = (;
        experiment      = "jet_sin",        # experiment choice: shear_sin, shear_rand, jet_sin, jet_rand, random
        rest_variable   = :θ,           # rest state temperature variable: :θ, ...
        rest_profile    = "tanh",       # profile of variable: constant, linear, tanh, sin
        # background rest state
        T₀              = 300.0,        # Initial temperature T(z₀) at z₀
        ΔT              = 1.0,          # ≈ T(top) - T(bottom)
        δ_T             = 1.,           # shear layer thickness
        q₀              = 30.0e-3,
        Δq              = 0., # -0.05,
        δ_q             = 1.0,          # shear layer thickness
        # shear flow experiment
        Δu              = 1.0,          # ≈ u(top) - u(bottom)     
        δ_u             = 1.,           # shear layer thickness
        # perturbation
        kick_size       = 0.01,         # perturbation amplitude (proportional to background Δ)
        λ_pert          = space.Lx      # wavelength of velocity perturbation
    )


    ## Time integration
    time_scheme  = TimeScheme(model)
    time_parameters = (; 
        Nslice = 50,
        slice_size = 1.,
        cfl = 0.8, 
        dt_max = 0.1, 
    )

    ## Data parameters
    exp_name = "retest"
    exp_model = "AN"
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
    params = (; time_parameters..., initial_parameters..., data_parameters...)

    ## Save parameters
    # save("$exp_dir/$exp_name/data/vars.jld2", Dict("model" => model, "params" => params));

    ## Prepare data objects
    include("../inc/data_processing.jl")

    write_func  = write_func                            # function to write data
    plot_func   = plot_func                             # function to plot data
    write_obj   = prepare_ds(params, model, Float64)             # prepare dataset
    plot_obj    = void                                  # prepare plot object

    ## Run simulation
    CFAnelastic.loop(
        model,
        time_scheme,
        params;
        plot_func   = plot_func,
        plot_obj    = plot_obj,
        write_func  = write_func,
        write_obj   = write_obj,
    )