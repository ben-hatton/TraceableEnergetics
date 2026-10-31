# Check of the tendency formula for d/dt P_*(z,t) at fixed z (FC2D, sorted plan, separable EOS).
import numpy as np
rng = np.random.default_rng(3)
v0, aT, aq, aP, gam, aTT, cp0 = 1e-3, 2e-4, 0.8, 1e-6, 0.0, 1e-5, 4000.0
P0, T0, q0, g, Lx, Lz = 1e5, 293.0, 0.01, 10.0, 1.0, 0.5
def v(P, th, q): return v0*(1 + aT*(1+gam*(P-P0))*(th-T0) - aq*(q-q0) - aP*(P-P0) + 0.5*aTT*(th-T0)**2)
def v_th(P, th, q): return v0*(aT*(1+gam*(P-P0)) + aTT*(th-T0))
def v_q(P, th, q): return -v0*aq + 0*P
def v_P(P, th, q): return v0*(-aP + aT*gam*(th-T0))
N, K = 400, 40
m = 1.0*(1 + 0.3*rng.random(N)); th = T0 + 2*rng.standard_normal(N); q = q0 + 2e-3*rng.standard_normal(N)
thdot, qdot = rng.standard_normal(N), 1e-3*rng.standard_normal(N)      # material rates (diabatic only)
M = m.sum(); Mk = M/K
def plan(th, q):                        # sorted fill from the top, split at slot boundaries
    order = np.argsort(-v(P0, th, q)); x = np.zeros((N, K)); k, room = 0, Mk
    for n in order:
        left = m[n]
        while left > 1e-14*M:
            d = min(left, room); x[n, k] += d; left -= d; room -= d
            if room <= 1e-14*M and k < K-1: k, room = k+1, Mk
    return x
def state(th, q):
    x = plan(th, q); Ptop = P0
    for it in range(50):
        Pk = Ptop + g/Lx*(np.arange(K)+0.5)*Mk
        R = np.sum(x*v(Pk, th[:, None], q[:, None])) - Lx*Lz
        if abs(R) < 1e-15: break
        Ptop -= R/np.sum(x*v_P(Pk, th[:, None], q[:, None]))
    Vk = np.sum(x*v(Pk, th[:, None], q[:, None]), axis=0)
    zedge = np.r_[np.cumsum(Vk[::-1])[::-1]/Lx, 0.0]                  # heights of the edges, top edge first (z = Lz)
    Pedge = Ptop + g/Lx*np.arange(K+1)*Mk
    return x, Ptop, Pk, Vk, zedge, Pedge
x, Ptop, Pk, Vk, zedge, Pedge = state(th, q)
# tendency formula
B = v_th(Pk, th[:, None], q[:, None])*thdot[:, None] + v_q(Pk, th[:, None], q[:, None])*qdot[:, None]
vP = v_P(Pk, th[:, None], q[:, None])
dPtop = -np.sum(x*B)/np.sum(x*vP)
dvk = np.sum(x*(B + vP*dPtop), axis=0)/Mk                              # d/dt v_* at fixed mass coordinate (slot means)
dz_edge = np.r_[np.cumsum((Mk*dvk)[::-1])[::-1]/Lx, 0.0]               # d/dt z at fixed mass, edges from the top
rho_k = Mk/Vk                                                          # slot density; edge k lies between slots k-1 and k
# d/dt P_* at fixed z = the current interior edge heights: dPtop + g rho_* dz|_mu (rho_* one-sided: use slot below edge)
# finite difference: P_* at fixed z from the perturbed state by linear interpolation of edge pressure in z
eps = 1e-6
_, Ptop2, _, _, zedge2, Pedge2 = state(th + eps*thdot, q + eps*qdot)
fd_Ptop = (Ptop2 - Ptop)/eps
Pz2 = np.interp(zedge[1:-1], zedge2[::-1], Pedge2[::-1])
fd = (Pz2 - Pedge[1:-1])/eps
formula_below = dPtop + g*rho_k[1:]*dz_edge[1:-1]     # density of the slot below the edge (dz>0 moves the edge up into it... check both)
formula_above = dPtop + g*rho_k[:-1]*dz_edge[1:-1]
print("dPtop formula/fd:", dPtop, fd_Ptop)
print("max rel err (rho below):", np.max(np.abs(formula_below-fd))/np.max(np.abs(fd)))
print("max rel err (rho above):", np.max(np.abs(formula_above-fd))/np.max(np.abs(fd)))
print("sample fd, formula:", fd[:4], formula_above[:4])
