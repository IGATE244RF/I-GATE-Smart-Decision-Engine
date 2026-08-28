import numpy as np
import pandas as pd

def make_data(n=240, seed=42):
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    maturity=np.clip(0.45+0.0022*t+rng.normal(0,0.025,n),0,1)
    quality=np.clip(0.60+0.0015*t+rng.normal(0,0.03,n),0,1)
    pressure=np.clip(0.25+0.001*t+rng.normal(0,0.035,n),0,1)
    risk=np.clip(0.58-0.0012*t+0.22*pressure+rng.normal(0,0.025,n),0,1)
    resources=np.clip(0.65-0.0004*t+rng.normal(0,0.025,n),0.2,1)
    return pd.DataFrame({"time":t,"maturity":maturity,"quality":quality,
                         "risk":risk,"schedule_pressure":pressure,"resource_capacity":resources})
