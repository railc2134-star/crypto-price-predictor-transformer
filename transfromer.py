import torch
import torch.nn as nn 
import math
import csv
import os
from datetime import datetime
raw_data = []
with open("BTC.csv", "r", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        date_obj = datetime.strptime(row["Date"], "%m/%d/%y")
        raw_data.append({
            "date": date_obj,
            "values": [
                float(row["Open"]), 
                float(row["High"]), 
                float(row["Low"]), 
                float(row["Close"])
            ]
        })
raw_data.sort(key=lambda x: x["date"])
price = [item["values"] for item in raw_data]
first_X=[]
first_Y=[]
for j in range(len(price)-30):
    first_X.append(price[j:j+30])
    first_Y.append(price[j+30][3]) 
X_tensor = torch.tensor(first_X, dtype=torch.float32)
Y_tensor = torch.tensor(first_Y, dtype=torch.float32)
train_sizeX=int(0.8*len(X_tensor))
train_sizeY=int(0.8*len(Y_tensor))
X_train=X_tensor[:train_sizeX]
Y_train=Y_tensor[:train_sizeY]
X_test=X_tensor[train_sizeX:]
Y_test=Y_tensor[train_sizeY:]
prices_mean=X_train.mean()
prices_std=X_train.std()
X_train=(X_train-prices_mean)/prices_std
Y_train=(Y_train-prices_mean)/prices_std
X_test=(X_test-prices_mean)/prices_std
Y_test=(Y_test-prices_mean)/prices_std
Y_train=Y_train.unsqueeze(1)
Y_test=Y_test.unsqueeze(1)
model_path = "brain5.pth"
class TransfromerPrediction(nn.Module):
    def __init__(self, feature_size=128,num_layers=4,num_heads=4):
        super().__init__()
        self.inpute_projection=nn.Linear(4,feature_size)
        self.gps = nn.Parameter(torch.randn(1, 30, feature_size) * 0.01)        
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=num_heads,
            batch_first=True,
            dropout=0.2
        )
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.boss=nn.Linear(feature_size,1)
    def forward(self,x):
        x=self.inpute_projection(x)
        x=x+self.gps
        x=self.transformer(x)
        x=x[:, -1, :]
        x=self.boss(x)
        return x
prediction_on=TransfromerPrediction()
model_path = "brain5.pth"
edit=torch.optim.Adam(prediction_on.parameters(),lr=0.0005)
creation=nn.MSELoss()
if os.path.exists(model_path):
    print("--- Loading existing weights from disk... ---")
    prediction_on.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))))
    do_training = False
else:
    print("--- No saved brain found. Training... ---")
    do_training = True
if do_training:
    for epoch in range(125):
        edit.zero_grad()
        outpute=prediction_on(X_train)
        loss=creation(outpute,Y_train)
        loss.backward()
        edit.step()
        if epoch %25==0 :
            print(f"epoch : {epoch}|| loss : {loss}")
    print(f"last loss is {loss}")
    torch.save(prediction_on.state_dict(),"brain5.pth")
    print("brain saved")
else:
    prediction_on.eval()
    with torch.no_grad() :
        outpute=prediction_on(X_test)
        loss=creation(outpute,Y_test)
        pred_real = outpute * prices_std + prices_mean
        actual_real = Y_test * prices_std + prices_mean
        mae_real = torch.mean(torch.abs(pred_real - actual_real))
        print(f"the diffrent(loss)={loss.item()}")
        print(f"Mean Absolute Error in $ : {mae_real.item():.2f}")
        print("Some Test Predictions vs Actual:")
        for i in range(8):                     
            print(f"Predicted: {pred_real[i].item():.2f}   |   Actual: {actual_real[i].item():.2f}   |   Error: {abs(pred_real[i].item() - actual_real[i].item()):.2f}")
last_info=price[-30:]
last_info=torch.tensor([last_info],dtype=torch.float32)
print(last_info)
last_info=(last_info-prices_mean)/prices_std
outpute=prediction_on(last_info)
pred_reals=outpute*prices_std+prices_mean
print(f"the next prediction{pred_reals.item()}")