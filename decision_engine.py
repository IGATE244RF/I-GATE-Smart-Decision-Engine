import pandas as pd

def rank_scenarios(rows):
    df=pd.DataFrame(rows)
    # Lower risk and schedule pressure are preferred; higher quality is preferred.
    df["score"]=2.0*df["risk_final"]+1.0*df["schedule_final"]+0.5*(1-df["quality_final"])
    return df.sort_values("score").reset_index(drop=True)

def gate_recommendation(ranked):
    best=ranked.iloc[0]
    return {"recommended_policy":best["policy"],"score":float(best["score"])}
