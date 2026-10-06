
# # reset netcdf after changing reference state
# function update_ref_dims!(ds, filename, ref_state, del_vars)
#     nbins = ref_state.nbins

#     # Original and new NetCDF file paths
#     src_file = filename
#     tmp_file = "$(src_file)_update.nc"

#     # dims and vars to update
#     # del_vars = ("b_star", "z_star", "zeta", "zeta_c")

#     # Open original dataset
#     ds_src = Dataset(src_file, "r")

#     # Create new dataset
#     ds_new = Dataset(tmp_file, "c")

#     # Copy dimensions except ref state dimensions
#     for (name, length) in ds_src.dim 
#         if name == "zeta"
#             defDim(ds_new, name, nbins+1)
#         elseif name == "zeta_c"
#             defDim(ds_new, name, nbins)
#         else
#             defDim(ds_new, name, length)
#         end
#     end

#     # Copy global attributes
#     for (attname, attval) in ds_src.attrib
#         ds_new.attrib[attname] = attval
#     end

#     # Copy variables (excluding those in del_var)
#     for var in NCDatasets.keys(ds_src)
#         if var in del_vars
#             println("Skipping variable $var...")
#             continue
#         end
#         # Define variable in new file
#         newvar = defVar(ds_new, var, eltype(ds_src[var]), dimnames(ds_src[var]), attrib = ds_src[var].attrib)

#         # Copy data
#         newvar[:] = ds_src[var][:]
#     end

#     # Close both datasets
#     close(ds_src)
#     close(ds_new)

#     rm(src_file; force=true)
#     mv(tmp_file, src_file)
# end


# function copy_ds!(ds, filename, copy_dims, copy_vars)
#     # original and new NetCDF file paths
#     src_file = filename
#     tmp_file = "$(src_file)_update.nc"

#     # open original dataset
#     ds_src = Dataset(src_file, "r")

#     # create new dataset
#     ds_new = Dataset(tmp_file, "c")

#     # copy dimensions except those in del_dim
#     for (name, length) in ds_src.dim 
#         if name in copy_dims
#             println("Copying dimension $name...")
#             defDim(ds_new, name, length)
#         end
#     end

#     # copy global attributes
#     for (attname, attval) in ds_src.attrib
#         ds_new.attrib[attname] = attval
#     end

#     # copy variables except those in del_var
#     for var in NCDatasets.keys(ds_src)
#         if var in copy_vars
#             println("Copying variable $var...")
#             defVar(ds, ds_src[var])
#         end    
#     end

#     # close both datasets
#     close(ds_src)
#     close(ds_new)

#     # delete source file
#     rm(src_file; force=true)

#     # move temp to source fil
#     mv(tmp_file, src_file)
# end



# # reset netcdf after changing reference state
# function del_vars_dims!(ds, filename, del_dims, del_vars)
#     # original and new NetCDF file paths
#     src_file = filename
#     tmp_file = "$(src_file)_update.nc"

#     # open original dataset
#     ds_src = Dataset(src_file, "r")

#     # create new dataset
#     ds_new = Dataset(tmp_file, "c")

#     # copy dimensions except those in del_dim
#     for (name, length) in ds_src.dim 
#         if name in del_dims
#             println("Deleting dim $name...")
#             continue
#         else
#             defDim(ds_new, name, length)
#         end
#     end

#     # copy global attributes
#     for (attname, attval) in ds_src.attrib
#         ds_new.attrib[attname] = attval
#     end

#     # copy variables except those in del_var
#     for var in NCDatasets.keys(ds_src)
#         if var in del_vars
#             println("Skipping variable $var...")
#             continue
#         end    
#         defVar(ds, ds_src[var])
#     end

#     # close both datasets
#     close(ds_src)
#     close(ds_new)

#     # delete source file
#     rm(src_file; force=true)

#     # move temp to source fil
#     mv(tmp_file, src_file)
# end

