import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import joblib

# ── Load real sepsis dataset ─────────────────────────────────────
df = pd.read_csv("dataSepsis.csv", sep=";")

# Keep only needed columns
df = df[["HR", "O2Sat", "Temp", "SBP", "isSepsis"]].copy()
df.rename(columns={"O2Sat": "SpO2", "SBP": "BP"}, inplace=True)

# Drop rows with missing values in key columns
df.dropna(subset=["HR", "SpO2", "Temp", "BP"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Dataset size after cleaning: {len(df)} rows")
print(f"Sepsis cases: {df['isSepsis'].sum()} | Normal: {(df['isSepsis']==0).sum()}")

# ── Balance classes ──────────────────────────────────────────────
sepsis  = df[df["isSepsis"] == 1]
normal  = df[df["isSepsis"] == 0]
min_len = min(len(sepsis), len(normal))
sepsis  = resample(sepsis, n_samples=min_len, random_state=42)
normal  = resample(normal, n_samples=min_len, random_state=42)
df = pd.concat([sepsis, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset: {len(df)} rows")

features = df[["HR", "BP", "Temp", "SpO2"]].values.astype(np.float32)
labels   = df["isSepsis"].values.astype(np.float32)

# ── Normalize ────────────────────────────────────────────────────
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved.")

# ── Build sequences of 5 ─────────────────────────────────────────
SEQ_LEN = 5
X, y = [], []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i+SEQ_LEN])
    y.append([labels[i+SEQ_LEN]])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)
print(f"Sequences: {X.shape}")

# ── Model ─────────────────────────────────────────────────────────
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
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── Train in batches ──────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS     = 20
dataset    = torch.utils.data.TensorDataset(X, y)
loader     = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "lstm_model.pt")
print("✅ Model trained & saved as lstm_model.pt")

model.eval()
with torch.no_grad():
    preds = model(X)
    preds = (preds > 0.5).float()   # convert to 0 or 1

    correct = (preds == y).sum().item()
    total = y.size(0)

    accuracy = correct / total
    print("Accuracy: ", accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set model to evaluation mode
model.eval()

with torch.no_grad():
    preds = model(X)
    preds = (preds > 0.5).int().numpy()   # convert to 0 or 1
    true  = y.int().numpy()

# Compute confusion matrix
cm = confusion_matrix(true, preds)

print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()