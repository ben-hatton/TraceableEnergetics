"""
    diagnostics(model::AN2D)
Return a `CookBook` of diagnostic functions for the `AN2D` model.
"""
diagnostics(::AN2D) = diagnostics_AN()

diagnostics_AN() = CookBook(;
    s           = (state, model) -> mult_v(state.m[:,:,1], model),
    q           = (state, model) -> mult_v(state.m[:,:,2], model),
    u           = state -> state.u,
    w           = state -> state.w,
    b           = scratch -> scratch.thermo.b,
    p           = p_diag,
    T           = scratch -> scratch.thermo.T,
    vorticity   = scratch -> scratch.derivatives.∂u_z - scratch.derivatives.∂w_x,
    divu        = scratch -> scratch.derivatives.∂u_x + scratch.derivatives.∂w_z,
    conjvar     = scratch -> scratch.thermo.conjvar,
    chempot     = scratch -> scratch.thermo.chempot,
    rs          = state -> state.m[:,:,1],
    rq          = state -> state.m[:,:,2],
    ωw          = scratch -> scratch.advection.ωw,
    ωu          = scratch -> scratch.advection.ωu, 
    Jcons_x     = scratch -> scratch.irreversible.Jcons_x,
    Jcons_z     = scratch -> scratch.irreversible.Jcons_z,
    Jq_x        = scratch -> scratch.irreversible.Jq_x,
    Jq_z        = scratch -> scratch.irreversible.Jq_z,
    ρε          = scratch -> scratch.irreversible.ρε,
    σ_cons      = scratch -> scratch.irreversible.σ_cons,
    div_Jcons   = (scratch, model) -> div_diag(scratch.irreversible.Jcons_x, scratch.irreversible.Jcons_z, scratch, model),
    div_Jq      = (scratch, model) -> div_diag(scratch.irreversible.Jq_x, scratch.irreversible.Jq_z, scratch, model),
    div_rsu     = (scratch, model) -> div_diag(scratch.advection.mu[:,:,1], scratch.advection.mw[:,:,1], scratch, model),
    div_rqu     = (scratch, model) -> div_diag(scratch.advection.mu[:,:,2], scratch.advection.mw[:,:,2], scratch, model),
    ∂u_x        = scratch -> scratch.derivatives.∂u_x,
    ∂w_z        = scratch -> scratch.derivatives.∂w_z,
    ∂u_z        = scratch -> scratch.derivatives.∂u_z,
    ∂w_x        = scratch -> scratch.derivatives.∂w_x,
    drs         = dstate -> dstate.m[:,:,1],
    drq         = dstate -> dstate.m[:,:,2],
    du          = dstate -> dstate.u,
    dw          = dstate -> dstate.w,
    φ           = scratch -> scratch.momentum.φ,
    su          = scratch -> scratch.advection.mu[:,:,1],
    sw          = scratch -> scratch.advection.mw[:,:,1],
    qu          = scratch -> scratch.advection.mu[:,:,2],
    qw          = scratch -> scratch.advection.mw[:,:,2],
    ke          = scratch -> scratch.advection.K,
    # u_c         = (state, model) -> centre_Xz(state.u, scratch, model),
    # w_c         = (state, model) -> centre_xZ(state.w, scratch, model),
)

function mult_v(arr, model)
    (; mgr, reference) = model
    (; v_R) = reference
    @with mgr, let (irange, jrange) = (axes(arr, 1), axes(arr, 2))
        @vec for i in irange, j in jrange
            arr[i, j] *= v_R[j]
        end
    end
    return arr
end

function p_diag(scratch, model)
    (; mgr, reference) = model
    (; ρ_R, p_R) = reference
    p = similar(scratch.momentum.φ)
    @with mgr, let (irange, jrange) = (axes(p, 1), axes(p, 2))
        @vec for i in irange, j in jrange
            p[i, j] = p[i, j] * ρ_R[j] + p_R[j]
        end
    end
    return p
end

function div_diag(Ax, Az, scratch, model)
    (; mgr, domain, inv_dx, inv_dz) = model
    div_A = similar(scratch.derivatives.∂u_x)
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange                      
            div_A[i, j] = inv_dx *  dif_x(Ax, i, j) + inv_dz * dif_z(Az, i, j)
        end       
    end
    periodize!(model, div_A)
    return div_A
end