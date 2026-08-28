import numpy as np
from scipy.integrate import solve_ivp

def rhs(t, x, u, risk_signal):
    # Bounded illustrative state dynamics for M,Q,R,S,C in [0,1].
    M,Q,R,S,C = np.clip(x, 0.0, 1.0)
    effort = float(np.clip(u, 0.5, 1.8))
    dM = 0.035*effort*(1-M) - 0.008*S
    dQ = 0.025*M*(1-Q) - 0.020*R*Q - 0.008*S*Q
    dR = 0.030*S*(1-R) + 0.015*(1-Q) + 0.010*risk_signal*(1-R) - 0.045*effort*R
    dS = 0.020*R*(1-S) + 0.015*(1-C) - 0.030*effort*S
    dC = 0.020*(1-C) - 0.005*effort*C
    return [dM,dQ,dR,dS,dC]

def simulate(x0, u, risk_signal=0.5, horizon=20.0, steps=200):
    t=np.linspace(0,horizon,steps)
    x0=np.clip(np.asarray(x0,dtype=float),0.0,1.0)
    sol=solve_ivp(lambda tt,xx: rhs(tt,xx,u,risk_signal),(0,horizon),x0,t_eval=t)
    traj=np.clip(sol.y.T,0.0,1.0)
    return sol.t, traj
