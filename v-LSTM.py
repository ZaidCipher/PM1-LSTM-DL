import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


print("Loading dataset...")
df = pd.read_csv('pm1_mass_concentration_dataset.csv')


target_col = 'PM1_Mass_Conc_ug_m3'
feature_cols = [c for c in df.columns if c.startswith('dN_dlogDp_')]

print(f"Total samples: {len(df)}")
print(f"Number of feature columns (size bins): {len(feature_cols)}")


data = df[feature_cols].values
target = df[target_col].values
data = np.nan_to_num(data, nan=0.0)
target = np.nan_to_num(target, nan=0.0)


scaler_x = StandardScaler()
scaler_y = StandardScaler()

data_scaled = scaler_x.fit_transform(data)
target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()


def create_sequences(features, targets, lookback=12):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : i + lookback])
        y.append(targets[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

LOOKBACK = 12
X, y = create_sequences(data_scaled, target_scaled, lookback=LOOKBACK)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training sequences: {len(X_train)} | Testing sequences: {len(X_test)}")


class SMPSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SMPSDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(SMPSDataset(X_test, y_test), batch_size=32, shuffle=False)


class PM1LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(PM1LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_dim = X.shape[2] # Number of size bins (192)
model = PM1LSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)

# Loss function (MSE) and Adam Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
print("\nStarting LSTM Training...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()       # Clear previous gradients
        outputs = model(batch_X)    # Forward pass
        loss = criterion(outputs, batch_y) # Compute loss
        loss.backward()             # Backward pass
        optimizer.step()            # Update model weights
        train_loss += loss.item() * batch_X.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # Validation Evaluation on Unseen Test Data
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            
    test_loss /= len(test_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

print("\nBaseline LSTM training complete!")
