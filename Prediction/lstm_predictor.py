import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# -------------------------
# LSTM Model Definition
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take only last time step
        return out

# -------------------------
# Training Function
# -------------------------
def train_lstm_model(csv_path='C:\\Users\\laxmiprasanna\\Desktop\\dcs_project\\traffic\\traffic_visualization\\traffic_data.csv', epochs=10, window_size=24):
    df = pd.read_csv(csv_path)
    data = df['requests'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Build sequences
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X, y = np.array(X), np.array(y)

    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'window_size': window_size
    }, 'models/lstm_model.pth')
    print("✅ LSTM model saved to models/lstm_model.pth")

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss")
    plt.grid(True)
    plt.legend()
    os.makedirs("prediction", exist_ok=True)
    plt.savefig("prediction/training_loss.png")
    plt.show()

# -------------------------
# Prediction Function
# -------------------------
def predict_next(n_steps=5, csv_path='C:\\Users\\laxmiprasanna\\Desktop\\dcs_project\\traffic\\traffic_visualization\\traffic_data.csv'):
    checkpoint = torch.load('C:\\Users\\laxmiprasanna\\Desktop\\dcs_project\\prediction\\models\\lstm_model.pth',weights_only=False)
    scaler = checkpoint['scaler']
    window_size = checkpoint['window_size']

    model = LSTMModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pd.read_csv(csv_path)
    data = df['requests'].values.reshape(-1, 1)
    data_scaled = scaler.transform(data)
    last_seq = torch.Tensor(data_scaled[-window_size:]).unsqueeze(0)  # (1, window, 1)

    preds_scaled = []
    for _ in range(n_steps):
        with torch.no_grad():
            next_val = model(last_seq).item()
        preds_scaled.append(next_val)
        next_seq = last_seq.squeeze(0).tolist()[1:] + [[next_val]]
        last_seq = torch.Tensor([next_seq])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Plot predicted vs actual (last 100 actuals + predicted)
    combined = np.concatenate((data[-100:].flatten(), preds))
    x_vals = np.arange(len(combined))
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals[:100], combined[:100], label="Actual")
    plt.plot(x_vals[100:], combined[100:], label="Predicted", linestyle="--")
    plt.title("LSTM Traffic Forecast (Last 100 + Predicted)")
    plt.xlabel("Time Steps")
    plt.ylabel("Requests")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction/lstm_predicted_vs_actual.png")
    plt.show()

    np.save("prediction/predicted_traffic.npy", preds)
    print("✅ Prediction completed and saved to prediction/predicted_traffic.npy")
    return preds

# ✅ Run for training
if __name__ == '__main__':
    train_lstm_model()
    predict_next(n_steps=20)
