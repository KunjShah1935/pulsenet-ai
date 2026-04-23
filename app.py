from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib

app = Flask(__name__)

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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sequence = np.array(data["sequence"], dtype=np.float32)  # shape (5, 4)

    # Normalize
    sequence = scaler.transform(sequence)

    X = torch.tensor(np.array([sequence]), dtype=torch.float32)

    with torch.no_grad():
        prob = model(X).item()

    risk = round(prob * 100, 2)

    # Rule-based override for all critically abnormal vitals
    sequence_raw = data["sequence"]
    critical_count = 0
    for reading in sequence_raw:
        hr, bp, temp, spo2 = reading[0], reading[1], reading[2], reading[3]
        if spo2 < 90:             critical_count += 1
        if spo2 < 85:             critical_count += 1  # extra weight
        if hr > 130 or hr < 40:   critical_count += 1
        if hr > 140:              critical_count += 1  # extra weight
        if temp > 103.5:          critical_count += 1
        if temp > 104.5:          critical_count += 1  # extra weight
        if bp < 60:               critical_count += 1  # only very low systolic
        if bp > 180:              critical_count += 1

    last = sequence_raw[-1]
    hr, bp, temp, spo2 = last[0], last[1], last[2], last[3]

    # If latest reading is fully normal, bring risk down
    all_normal = (
        60 <= hr <= 100 and
        90 <= bp <= 140 and
        97 <= temp <= 99.5 and
        spo2 >= 95
    )
    if all_normal:
        risk = min(risk, 25.0)

    # Extreme critical
    elif spo2 < 80 or hr > 145 or hr < 35 or temp > 105 or bp < 50:
        risk = max(risk, 92.0)
    # Very high
    elif spo2 < 85 or hr > 135 or temp > 104 or bp < 60:
        risk = max(risk, 80.0)
    # High — accumulation across sequence
    elif critical_count >= 6:
        risk = max(risk, 75.0)
    elif critical_count >= 4:
        risk = max(risk, 65.0)
    elif critical_count >= 2:
        risk = max(risk, 55.0)

    if risk < 25:
        status = "Normal"
    elif risk < 45:
        status = "Warning"
    else:
        status = "High Risk"

    return jsonify({"risk": risk, "status": status})

if __name__ == "__main__":
    app.run(debug=True)
