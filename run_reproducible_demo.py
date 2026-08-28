import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from src.data_generator import make_data
from src.deep_learning import train_predict
from src.system_dynamics import simulate
from src.optimal_control import OptimalControlEngine
from src.decision_engine import rank_scenarios, gate_recommendation

df=make_data()
model,rmse,pred=train_predict(df)
x0=df[["maturity","quality","risk","schedule_pressure","resource_capacity"]].iloc[-1].values.astype(float)
risk_signal=float(np.clip(pred,0,1))

opt=OptimalControlEngine()
u_star,j_star=opt.optimize(x0,risk_signal)

policies={"continue":1.0,"optimized":u_star,"additional_validation":min(1.8,u_star+0.25)}
rows=[]
for name,u in policies.items():
    t,tr=simulate(x0,u,risk_signal)
    rows.append({"policy":name,"risk_final":tr[-1,2],
                 "schedule_final":tr[-1,3],"quality_final":tr[-1,1]})
ranked=rank_scenarios(rows)
print("Illustrative LSTM RMSE:",round(rmse,4))
print("Predicted risk:",round(risk_signal,4))
print("Optimized control u*:",round(u_star,4))
print("Objective J(u*):",round(j_star,4))
print(ranked)
print("Gate recommendation:",gate_recommendation(ranked))

plt.figure(figsize=(8,5))
for name,u in policies.items():
    t,tr=simulate(x0,u,risk_signal)
    plt.plot(t,tr[:,2],label=name)
plt.xlabel("Time")
plt.ylabel("Risk exposure")
plt.title("Illustrative policy trajectories")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parents[1]/"figures/scenario_trajectories.png",dpi=180)
