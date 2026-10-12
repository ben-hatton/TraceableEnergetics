using NCDatasets
using Plots; gr()
using MutatingOrNot: Void, void
import CookBooks: CookBooks, CookBook, open, close

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

    # define variables from save_variables, with the long name, units and grid location of each diagnostic
    info = diagnostic_info(model)
    for sym in saved_variables
        name = string(sym)
        attrib = ["long_name" => info[sym].long_name, "units" => info[sym].units]
        # dimensions from the grid location, e.g. :Xz ↦ ("X", "z")
        defVar(ds, name, F, (map(string, collect(string(info[sym].location)))..., "t"); attrib)
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
