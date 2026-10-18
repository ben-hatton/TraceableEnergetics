# Post-processing of AN2D output stored in NetCDF files (moved out of CFAnelastics, which no longer depends on
# NCDatasets). Load after the model packages, e.g. from an experiment's analysis script:
#     include("<path to>/energetics/energetics_io.jl")
using NCDatasets
using CFBoxes
using CFAnelastics: AN2D
include("energetics.jl")
using CFTimeSchemes: CFTimeSchemes
using ManagedLoops: @vec, @with
using MutatingOrNot: Void, void, similar!

function energetics(ds, keys, model::AN2D{F}, params) where {F}
    (; Nslice, slice_size, energetics_variables) = params

    # define variables from energetics_variables
    # assume coordinates x, z, t already defined
    for sym in energetics_variables
        ~(string(sym) in NCDatasets.keys(ds)) && defVar(ds, string(sym), F, ("x", "z", "t"))
    end

    # initialise energetics state (; s, q, u, w ) from simulation dataset
    energetics_state0 = initialise_energetics_state(ds, keys)
    energetics_state = deepcopy(energetics_state0)

    # initialise dynamics scratch space
    dynamics_dstate, dynamics_scratch = CFTimeSchemes.tendencies!(void, void, model, energetics_state0, nothing)

    # initialise energetics scratch space
    energetics_scratch = energetics_tendencies!(void, model, energetics_state0, dynamics_scratch)

    # run loop
    for t_iter = 1:Nslice
        # update energetics state from dataset
        update_energetics_state!(energetics_state, ds, keys, t_iter)

        # compute dynamics scratch space
        dynamics_dstate, dynamics_scratch = CFTimeSchemes.tendencies!(dynamics_dstate, dynamics_scratch, model, energetics_state, t_iter * slice_size)

        # compute energetics
        energetics_tendencies!(energetics_scratch, model, energetics_state, dynamics_scratch)

        # write data from energetics diagnostics
        session = open(energetics_diagnostics(); energetics_scratch, model)
        let data = CookBooks.get(session, energetics_variables)
            for sym in energetics_variables
                ds[sym][:, :, t_iter] = data[sym]
            end
        end
        close(session);

        # advance progress meter
        progress(t_iter, slice_size, Nslice * slice_size)
    end
end

## TOOLS

# geopotential at cell centre
function initialise_energetics_state(ds, keys)
    energetics_state = (;
        m = ds[keys.m][:, :, :, 1],
        u = ds[keys.u][:, :, 1],
        w = ds[keys.w][:, :, 1]
        )
    return energetics_state
end

# update
function update_energetics_state!(energetics_state, ds, keys, t_iter)
    (; m, u, w) = energetics_state
    m[:, :, :] .= ds[keys.m][:, :, :, t_iter]
    u[:, :] .= ds[keys.u][:, :, t_iter]
    w[:, :] .= ds[keys.w][:, :, t_iter]
end

# construct state from (s, q, u, w)
function construct_state!(ds, keys, model::AN2D{F}) where {F}
    ~("mi" in NCDatasets.keys(ds.dim)) && defDim(ds, "mi", 2)
    ~("m" in NCDatasets.keys(ds)) && defVar(ds, "m", F, ("x", "z", "mi", "t"))
    
    (; mgr, domain, reference) = model
    (; ρ_R) = reference

    Nslice = ds.attrib["Nslice"]

    dsm1 = @views ds["m"][:, :, 1, :]
    dsm2 = @views ds["m"][:, :, 2, :]

    let jrange = cells(domain, 2)
        for j in jrange
            dsm1[:, j, :] = ds[keys.s][:, j, :] .* ρ_R[j]
            dsm2[:, j, :] = ds[keys.q][:, j, :] .* ρ_R[j]
        end
    end
    keys = (; m = :m, keys...)
    return ds, keys
end

## DIAGNOSTICS

# diagnostics for energetics
energetics_diagnostics() = CookBook(;
    ke = energetics_scratch -> energetics_scratch.energies.ke,
    pe = energetics_scratch -> energetics_scratch.energies.pe,
    te = energetics_scratch -> energetics_scratch.energies.te,
    gpe = energetics_scratch -> energetics_scratch.bq_energies.gpe,
    ie = energetics_scratch -> energetics_scratch.bq_energies.ie,
    gpe_bz = energetics_scratch -> energetics_scratch.bq_energies.gpe_bz,
    viscous_diss = energetics_scratch -> energetics_scratch.irreversible.viscous_diss,
    entropy_prod = energetics_scratch -> energetics_scratch.irreversible.entropy_prod,
    pdivu = energetics_scratch -> energetics_scratch.dynamic.pdivu,
    consflux_x = energetics_scratch -> energetics_scratch.irreversible.consflux_x,
    consflux_z = energetics_scratch -> energetics_scratch.irreversible.consflux_z,
    conjdivconsflux = energetics_scratch -> energetics_scratch.irreversible.conjdivconsflux,
    chemdivqflux = energetics_scratch -> energetics_scratch.irreversible.chemdivqflux,
    consfluxgradconj = energetics_scratch -> energetics_scratch.irreversible.consfluxgradconj,
    qfluxgradchem = energetics_scratch -> energetics_scratch.irreversible.qfluxgradchem,
    buoy_flux_Σ_ϕ = energetics_scratch -> energetics_scratch.dynamic.buoy_flux_Σ_ϕ,
    buoy_flux_B = energetics_scratch -> energetics_scratch.dynamic.buoy_flux_B,
    divru = energetics_scratch -> energetics_scratch.thermodynamic.divru,
    conjdivrsu = energetics_scratch -> energetics_scratch.thermodynamic.conjdivrsu,
    chemdivrqu = energetics_scratch -> energetics_scratch.thermodynamic.chemdivrqu
)