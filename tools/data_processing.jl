using NCDatasets
using Plots; gr()
using MutatingOrNot: Void, void
import CookBooks: CookBooks, CookBook, open, close
using GLMakie  # or CairoMakie if you want static output

function plot_func(model, state, dstate, scratch, cookbook, plot_obj, t, params) 
    (; plotted_variable) = params
    (; domain) = model
    
    session = open(cookbook; state, dstate, scratch, model)
    field = CookBooks.get(session, plotted_variable)
    plot_obj = update_obs!(plot_obj, interior(domain, field))
    close(session)
    return plot_obj
end

function update_obs!(plot_obj::Observable, field)
    plot_obj[] = field
    return plot_obj
end

function update_obs!(plot_obj_::Void, field)
    plot_obj = Observable(field)
    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1])
    hm = GLMakie.heatmap!(ax, plot_obj, colormap = :viridis, interpolate = false)
    display(fig)
    return plot_obj
end

function prepare_ds(params, model, F)
    (; data_file, Nslice, saved_variables, exp_model, exp_name, slice_size) = params
    (; domain, space, dx, dz) = model
    (; Lx, Lz) = space

    Mx, Mz = dims(domain)
    Hx, Hz = halo_size(domain)

    # remove existing data file    
    isfile(data_file) && rm(data_file)
    ~isfile(data_file) && mkpath(dirname(data_file))
    
    # create netcdf dataset
    ds = NCDataset(data_file, "c")

    # define dimensions of staggered arrays
    defDim(ds, "t", Nslice)
    defDim(ds, "x", xdim(domain))
    defDim(ds, "z", zdim(domain))
    defDim(ds, "X", Xdim(domain))
    defDim(ds, "Z", Zdim(domain))

    # define global attributes
    ds.attrib["title"]      = "Simulation data"
    ds.attrib["model"]      = exp_model
    ds.attrib["experiment"] = exp_name
    ds.attrib["Mx"]         = Mx
    ds.attrib["Mz"]         = Mz
    ds.attrib["halo_x"]     = Hx
    ds.attrib["halo_z"]     = Hz
    ds.attrib["Lx"]         = Lx
    ds.attrib["Lz"]         = Lz
    ds.attrib["dx"]         = dx
    ds.attrib["dz"]         = dz
    ds.attrib["Nslice"]     = Nslice
    ds.attrib["slice_size"] = slice_size

    # define variables from save_variables, with the long name and units of each diagnostic
    info = diagnostic_info(model)
    for sym in saved_variables
        name = string(sym)
        attrib = ["long_name" => info[sym].long_name, "units" => info[sym].units]
        if sym in (:u, :du, :Jcons_x, :Jq_x)
            defVar(ds, name, F, ("X", "z", "t"); attrib)
        elseif sym in (:w, :dw, :Jcons_z, :Jq_z)
            defVar(ds, name, F, ("x", "Z", "t"); attrib)
        elseif sym in (:vorticity, :∂u_z, :∂w_x)
            defVar(ds, name, F, ("X", "Z", "t"); attrib)
        else
            defVar(ds, name, F, ("x", "z", "t"); attrib)
        end
    end
    return ds
end

function write_func(model, state, dstate, scratch, cookbook, ds::NCDataset, t, params)
    (; saved_variables) = params

    # open diagnostics
    session = open(cookbook; state, dstate, scratch, model)

    # write data from diagnostics
    let data = CookBooks.get(session, saved_variables)
        for sym in saved_variables
            ds[sym][:, :, t] = data[sym]
        end
    end

    close(session)
    return ds
end

function write_func(model, state, dstate, scratch, cookbook, ds::Nothing, t, params) end

# long names and units of the diagnostics of a model, from the package that defines it (CFAnelastics, CFCompressible)
diagnostic_info(model) = parentmodule(typeof(model)).diagnostic_info(model)
