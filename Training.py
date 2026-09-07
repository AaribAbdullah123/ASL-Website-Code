import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[INFO] Device: {device}")

DATA_DIR = "./asl_dataset"
actions = [
    "hello", "thank", "please", "yes", "no", "help", "good", "bad", "more", "stop",
    "sorry", "name", "want", "like", "love", "eat", "drink", "water", "food", "friend",
    "family", "home", "work", "school", "time", "day", "night", "today", "now", "where",
    "who", "what", "when", "why", "how", "fast", "slow", "big", "small", "hot",
    "cold", "happy", "sad", "open", "close", "read", "write", "learn", "computer", "understand"
]
sequence_length = 90
input_size = 63

def load_data():
    X, y = [], []
    for label_idx, action in enumerate(actions):
        action_dir = os.path.join(DATA_DIR, action)
        if not os.path.exists(action_dir):
            continue
        for file in os.listdir(action_dir):
            if file.endswith(".npy"):
                path = os.path.join(action_dir, file)
                data = np.load(path)
                if data.shape == (sequence_length, input_size):
                    X.append(data)
                    y.append(label_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X, y = load_data()
if len(X) == 0:
    raise ValueError("[ERROR] Dataset missing.")

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

class ASLBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(ASLBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = ASLBiLSTM(input_dim=input_size, hidden_dim=128, output_dim=len(actions)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 60
print("[INFO] Training started...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    epoch_loss = running_loss / train_size

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total if val_total > 0 else 0

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), "asl_bilstm_model.pth")
print("[INFO] Model saved.")
