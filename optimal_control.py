from scipy.optimize import minimize_scalar
import numpy as np
from .system_dynamics import simulate

class OptimalControlEngine:
    def __init__(self, weights=(2.0,1.0,0.5), bounds=(0.5,1.8)):
        self.w_r,self.w_s,self.w_u=weights
        self.bounds=bounds

    def objective(self,u,x0,risk_signal):
        t,traj=simulate(x0,u,risk_signal)
        R=traj[:,2]; S=traj[:,3]
        running=self.w_r*np.mean(R)+self.w_s*np.mean(S)+self.w_u*(u-1.0)**2
        terminal=self.w_r*R[-1]+self.w_s*S[-1]
        return float(running+terminal)

    def optimize(self,x0,risk_signal):
        res=minimize_scalar(lambda u:self.objective(u,x0,risk_signal),
                            bounds=self.bounds,method="bounded")
        return float(res.x),float(res.fun)
