# Numerical check of the transport-dual <-> reference-pressure links claimed in the revised note.
import numpy as np
from scipy.optimize import linprog
rng = np.random.default_rng(1)
v0, aT, aq, aP, gam, aTT, cp0 = 1e-3, 2e-4, 0.8, 1e-6, 3e-5, 1e-5, 4000.0
P0, T0, q0, g = 1e5, 293.0, 0.01, 10.0
def v(P, th, q): return v0*(1 + aT*(1+gam*(P-P0))*(th-T0) - aq*(q-q0) - aP*(P-P0) + 0.5*aTT*(th-T0)**2)
def h(P, th, q): return cp0*(th-T0) + v0*(1 + aT*(1+0.5*gam*(P-P0))*(th-T0) - aq*(q-q0) - 0.5*aP*(P-P0) + 0.5*aTT*(th-T0)**2)*(P-P0)
def solve_tp(C, m, M):
    N, K = C.shape
    Aeq = np.zeros((N+K, N*K))
    for n in range(N): Aeq[n, n*K:(n+1)*K] = 1
    for k in range(K): Aeq[N+k, k::K] = 1
    r = linprog(C.ravel(), A_eq=Aeq, b_eq=np.r_[m, M], bounds=(0, None), method="highs")
    return r.x.reshape(N, K), r.fun, r.eqlin.marginals[:N], r.eqlin.marginals[N:]

# ---------------- AN2D ----------------
Mx, Mz, Lx, Lz = 5, 8, 1.0, 0.5
dx, dz = Lx/Mx, Lz/Mz
z = (np.arange(Mz)+0.5)*dz
rhoR = 1000*(1 + 0.02*(Lz-z)/Lz)
PR = np.empty(Mz); PR[-1] = P0 + 0.5*dz*g*rhoR[-1]
for k in range(Mz-2, -1, -1): PR[k] = PR[k+1] + g*dz*0.5*(rhoR[k]+rhoR[k+1])
th = T0 + 2*rng.standard_normal((Mx, Mz)); qq = q0 + 2e-3*rng.standard_normal((Mx, Mz))
row = np.tile(np.arange(Mz), (Mx, 1)).ravel(); th, qq = th.ravel(), qq.ravel()
m = rhoR[row]*dx*dz; M = Mx*rhoR*dx*dz
C = h(PR[None, :], th[:, None], qq[:, None]) + g*z[None, :]
x, Estar, u, psi = solve_tp(C, m, M)
EP = np.sum(m*C[np.arange(len(m)), row])
pi = -psi                                     # claimed: pi_* = (P_* - P_R)/rho_R = -psi (+ const)
S = C + pi[None, :]
A = S[np.arange(len(m)), row] - S.min(axis=1)
print("AN: E_P - E_* =", EP-Estar, "  sum m A (pi=-psi, z_* = argmin over rows) =", np.sum(m*A))
print("AN: A >= 0:", A.min() >= -1e-9, "  argmin S on support of x:",
      all(np.isclose(S[n, k], S[n].min()) for n, k in zip(*np.nonzero(x > 1e-12))))
# split parcels: step of pi across rows equals the enthalpy+geopotential step of the split parcel
for n in range(len(m)):
    ks = np.nonzero(x[n] > 1e-12)[0]
    if len(ks) > 1:
        k1, k2 = ks[0], ks[1]
        print("  split parcel", n, "rows", k1, k2, ": pi step", pi[k2]-pi[k1], " c step", C[n, k1]-C[n, k2])
        break
# horizontal-mean profile (row-mean v at P_R, pi hydrostatic with the model's face average): weak duality
vbar = np.array([np.mean(v(PR[k], th[row == k], qq[row == k])) for k in range(Mz)])
b = g*(rhoR*vbar - 1)
pim = np.r_[0, np.cumsum(dz*0.5*(b[1:]+b[:-1]))]
Sm = C + pim[None, :]
Am = Sm[np.arange(len(m)), row] - Sm.min(axis=1)
print("AN mean profile: sum m A =", np.sum(m*Am), ">= E_P - E_* :", np.sum(m*Am) >= EP-Estar-1e-9)

# ---------------- FC2D: slots of equal mass, outer loop on P_top ----------------
K = Mz; Mtot = m.sum(); Mk = np.full(K, Mtot/K)
def inner(Ptop):
    Pk = Ptop + g/Lx*(np.arange(K)+0.5)*Mtot/K          # slot k from the top
    Cf = h(Pk[None, :], th[:, None], qq[:, None])
    x, f, u, psi = solve_tp(Cf, m, Mk)
    vol = np.sum(x*v(Pk[None, :], th[:, None], qq[:, None]))
    return x, f, psi, Pk, vol
Ptop = P0
for it in range(30):
    x, f, psi, Pk, vol = inner(Ptop)
    R = vol - Lx*Lz
    dR = -np.sum(x*v0*aP)   # v_P ~ -v0 aP (plus small thermobaric part), enough for Newton
    if abs(R)/(Lx*Lz) < 1e-13: break
    Ptop -= R/dR
Vk = np.sum(x*v(Pk[None, :], th[:, None], qq[:, None]), axis=0)
edges = np.r_[0, np.cumsum(Vk[::-1]/Lx)]            # heights from the bottom
zc = (0.5*(edges[1:]+edges[:-1]))[::-1]             # slot centres, slot 0 at the top
print("FC: |R|/V =", abs(R)/(Lx*Lz), " column height =", edges[-1])
dpsi = psi[:-1]-psi[1:]                              # psi_k - psi_{k+1}, k above k+1
dPhi = g*(zc[:-1]-zc[1:])
print("FC: psi_k - psi_{k+1} =", np.round(dpsi, 5))
print("FC: -(Phi_k - Phi_{k+1})=", np.round(-dPhi, 5))
