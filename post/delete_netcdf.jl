using NCDatasets

# Original and new NetCDF file paths
src_file = "../shear/data/test_256/fields.nc"
tmp_file = "../shear/data/test_256/fields_clean.nc"

# vars to delete
del_vars = ("b_star", "z_star", "zeta", "zeta_c")

# Open original dataset
ds_src = Dataset(src_file, "r")

# Create new dataset
ds_new = Dataset(tmp_file, "c")

# Copy dimensions
for (name, length) in ds_src.dim
    defDim(ds_new, name, length)
end

# Copy global attributes
for (attname, attval) in ds_src.attrib
    ds_new.attrib[attname] = attval
end

# Copy variables (excluding those in del_var)
for var in NCDatasets.keys(ds_src)
    if var in del_vars
        println("Skipping variable $var...")
        continue
    end
    # Define variable in new file
    newvar = defVar(ds_new, var, eltype(ds_src[var]), dimnames(ds_src[var]), attrib = ds_src[var].attrib)

    # Copy data
    newvar[:] = ds_src[var][:]
end

# Close both datasets
close(ds_src)
close(ds_new)



rm(src_file; force=true)
mv(tmp_file, src_file)

