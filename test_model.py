import torch
import torch.nn as nn
import numpy as np
import joblib

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sig(self.fc(out[:, -1, :]))

model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pt"))
model.eval()
scaler = joblib.load("scaler.pkl")

seq = [
    [72, 100, 85, 78],
    [130, 85, 102, 88],
    [75, 100, 100, 85],
    [75, 100, 100, 88],
    [130, 85, 102, 88],
]

seq_scaled = scaler.transform(seq)
X = torch.tensor([seq_scaled], dtype=torch.float32)
with torch.no_grad():
    prob = model(X).item()
print(f"Risk: {round(prob*100, 2)}%")
