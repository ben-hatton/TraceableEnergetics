# Does d/dt v_* follow the 1/N^2-weighted mean of vdot over the parcels at a level (AN, exact LP, strong thermobaricity)?
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
v0, aT, aq, aP, gam, cp0 = 1e-3, 2e-4, 0.8, 1e-6, 1.3e-4, 4000.0
P0, T0, q0, g, rho, Lx, Lz = 1e5, 293.0, 0.01, 10.0, 1000.0, 1.0, 0.5
def v(P, th, q):  return v0*(1 + aT*(1+gam*(P-P0))*(th-T0) - aq*(q-q0) - aP*(P-P0))
def h(P, th, q):  return cp0*(th-T0) + v0*(1 + aT*(1+0.5*gam*(P-P0))*(th-T0) - aq*(q-q0) - 0.5*aP*(P-P0))*(P-P0)
def vth(P, th, q): return v0*aT*(1+gam*(P-P0))
def vP(P, th, q):  return v0*(-aP + aT*gam*(th-T0))
K, per = 80, 10                          # slots, parcels of each family per slot
n1 = K*per; s = (np.arange(n1)+0.5)/n1
thA, qA = T0 + 4*(1-s), q0 + 0*s         # family A: warm at the top
thB = thA + 3.0                          # family B: warmer, compensated by q at mid-depth pressure
Pmid = P0 + rho*g*Lz/2
qB = np.full(n1, q0 + aT*(1+gam*(Pmid-P0))*3.0/aq)
th0, q = np.r_[thA, thB], np.r_[qA, qB]; N = 2*n1
dz = Lz/K; zk = (K-np.arange(K)-0.5)*dz  # slot 0 at the top
PR = P0 + rho*g*(Lz-zk)
m = np.full(N, rho*Lx*dz/(2*per)); Mk = np.full(K, rho*Lx*dz)
rows = np.r_[np.repeat(np.arange(N), K), N + np.tile(np.arange(K), N)]
cols = np.r_[np.arange(N*K), np.arange(N*K)]
Aeq = sp.csr_matrix((np.ones(2*N*K), (rows, cols)), shape=(N+K, N*K))
def solve(th):
    C = h(PR[None, :], th[:, None], q[:, None]) + g*zk[None, :]
    r = linprog(C.ravel(), A_eq=Aeq, b_eq=np.r_[m, Mk], bounds=(0, None), method="highs")
    x = r.x.reshape(N, K)
    return x, np.sum(x*v(PR[None, :], th[:, None], q[:, None]), axis=0)/Mk
thdot = np.r_[np.ones(n1), np.zeros(n1)]  # heat family A only
x, vs = solve(th0)
eps = 0.02
_, vp = solve(th0 + eps*thdot); _, vm = solve(th0 - eps*thdot)
fd = (vp - vm)/(2*eps)
dvz = -np.gradient(vs, dz)               # d v_*/dz (slots ordered top to bottom)
vdot = vth(PR[None, :], th0[:, None], q[:, None])*thdot[:, None]
N2 = g*rho*(dvz[None, :] + g*rho*vP(PR[None, :], th0[:, None], q[:, None]))
plain = np.sum(x*vdot, axis=0)/Mk
w = x/N2; weighted = np.sum(w*vdot, axis=0)/np.sum(w, axis=0)
inner = slice(10, K-10)
print("N2 family A / B (mid):", N2[n1//2, K//2] if x[n1//2, K//2] else np.nan, N2[n1+n1//2, K//2])
print("mean over interior slots:  fd %.4e  plain %.4e  weighted %.4e" % (fd[inner].mean(), plain[inner].mean(), weighted[inner].mean()))
print("rms rel. error  plain %.3f  weighted %.3f" % (np.sqrt(np.mean((plain-fd)[inner]**2))/np.abs(fd[inner]).mean(),
                                                   np.sqrt(np.mean((weighted-fd)[inner]**2))/np.abs(fd[inner]).mean()))
