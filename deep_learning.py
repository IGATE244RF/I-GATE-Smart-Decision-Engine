import numpy as np
import torch
from torch import nn

class RiskLSTM(nn.Module):
    def __init__(self, n_features=4, hidden=16):
        super().__init__()
        self.lstm=nn.LSTM(n_features,hidden,batch_first=True)
        self.fc=nn.Linear(hidden,1)
    def forward(self,x):
        y,_=self.lstm(x)
        return self.fc(y[:,-1,:])

def train_predict(df, window=12, epochs=25, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    cols=["maturity","quality","schedule_pressure","resource_capacity"]
    X=df[cols].values.astype("float32")
    y=df["risk"].values.astype("float32")
    xs=[]; ys=[]
    for i in range(window,len(df)):
        xs.append(X[i-window:i]); ys.append(y[i])
    split=int(0.8*len(xs))
    Xtr=torch.tensor(np.array(xs[:split])); ytr=torch.tensor(np.array(ys[:split])).view(-1,1)
    Xte=torch.tensor(np.array(xs[split:])); yte=torch.tensor(np.array(ys[split:])).view(-1,1)
    model=RiskLSTM(len(cols))
    opt=torch.optim.Adam(model.parameters(),lr=0.01)
    loss_fn=nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); pred=model(Xtr); loss=loss_fn(pred,ytr); loss.backward(); opt.step()
    with torch.no_grad(): pred=model(Xte)
    rmse=float(torch.sqrt(loss_fn(pred,yte)))
    return model, rmse, float(pred[-1])
