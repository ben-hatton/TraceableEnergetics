# Boundary conditions for CFAnelastics model

"""
    state_bc!(model::AN2D, state, scratch) -> NamedTuple

Apply physical boundary conditions in-place to the prognostic **state variables**
of a 2D anelastic model.

This routine enforces boundary conditions on the velocity components
`u`, `w`, and on either the conservative thermodynamic variable `consvar`
*or* the temperature `T`, depending on the boundary-condition configuration.
The remaining thermodynamic quantity is reconstructed consistently.

# Arguments
- `model::AN2D`
- `(; m, u, w)`
- `(consvar, comp, T)`:

# Boundary-condition logic
Exactly one of the following must be specified:
- Boundary conditions on `(consvar, q)`
- Boundary conditions on `(T, q)`

Specifying both `consvar` and `T` boundary conditions is an error.

# Returns
A named tuple `(m = m, u = u, w = w)` with boundary conditions applied.
"""
function state_bc!(model::AN2D, (; m, u, w)::NamedTuple, (consvar, comp, T)::Tuple)
    (; boundary, reference, fluid) = model
    (; p_R, ρ_R) = reference

    bcs = boundary.conditions
    bc_keys = boundary.keys # (:u, :w, :consvar, :q) or (:u, :w, :T, :q)

    if (:consvar in bc_keys) && ~(:T in bc_keys)
        # apply boundary conditions to (u, w, consvar, q)
        apply_bc_xz!(model, bcs.consvar, consvar)
        apply_bc_xz!(model, bcs.q, comp)
        apply_bc_Xz!(model, bcs.u, u)                                                                                                                                                                                                                                      
        apply_bc_xZ!(model, bcs.w, w)

        # construct m = (ρ_R * consvar, ρ_R * comp)
        @with model.mgr, let (irange, jrange) = (axes(m, 1), axes(m, 2))
            @vec for i in irange, j in jrange
                T[i, j] = fluid(:p, :consvar, :q).temperature(p_R[j], consvar[i, j], comp[i, j])
                m[i, j, 1] = consvar[i, j] * ρ_R[j]
                m[i, j, 2] = comp[i, j] * ρ_R[j]
            end
        end
    elseif (:T in bc_keys) && ~(:consvar in bc_keys)                                                                                                                                                
        # apply boundary conditions to (u, w, T, q)
        apply_bc_xz!(model, bcs.T, T)
        apply_bc_xz!(model, bcs.q, comp)
        apply_bc_Xz!(model, bcs.u, u)
        apply_bc_xZ!(model, bcs.w, w)

        # construct m = (ρ_R * consvar, ρ_R * comp)
        @with model.mgr, let (irange, jrange) = (axes(m, 1), axes(m, 2))
            @vec for i in irange, j in jrange
                consvar[i, j] = fluid(:p, :T, :q).conservative_variable(p_R[j], T[i, j], comp[i, j])
                m[i, j, 1] = consvar[i, j] * ρ_R[j]
                m[i, j, 2] = comp[i, j] * ρ_R[j]
            end
        end
    elseif (:consvar in bc_keys) && (:T in bc_keys)
        error("Boundary conditions cannot be specified for both consvar and T.")
    else 
        error("Boundary conditions must be specified for either (consvar, q) or (T, q).")
    end
    return (; m = m, u = u, w = w)
end