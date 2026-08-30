import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


# 1. Load Extracted Features
print("Loading extracted features...")
weights_only=False
data = torch.load("fakeddit_features_5k.pt", weights_only=False)

# 2. Prepare Feature Matrix (X) and Labels (y)
X = []
y_2way = []
y_6way = []

for item in data:
    txt = item["text_vector"]
    img = item["image_vector"]
    cos = np.array([item["cosine_similarity"]])
    
    # Concatenate: [Text (512) || Image (512) || Cosine Sim (1)] = 1025 dims
    fused_vector = np.concatenate([txt, img, cos])
    
    X.append(fused_vector)
    y_2way.append(item["label_2way"])
    y_6way.append(item["label_6way"])

X = np.array(X, dtype=np.float32)
y_2way = np.array(y_2way, dtype=np.int64)
y_6way = np.array(y_6way, dtype=np.int64)

# 3. Train/Test Split (80% Train, 20% Evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_6way, test_size=0.2, random_state=42, stratify=y_6way
)

# 4. PyTorch Dataset & DataLoader
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features)
        self.y = torch.tensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=64, shuffle=False)

# 5. Define the Lightweight Multimodal Classifier
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim=1025, num_classes=2):
        super(FakeNewsClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = FakeNewsClassifier(input_dim=1025, num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 6. Training Loop
print("\n--- Training Binary Classifier (6-Way) ---")
epochs = 50

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss:.4f}")

# 7. Evaluation & Metrics
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

print("\n--- EVALUATION RESULTS ---")
print(f"Accuracy: {accuracy_score(all_targets, all_preds) * 100:.2f}%")
print("\nClassification Report (0=True, 1=Satire, 2=Misleading, 3=False Context, 4=Imposter, 5=Manipulated):")
print(classification_report(all_targets, all_preds, target_names=["True", "Satire", "Misleading", "False Context", "Imposter", "Manipulated"]))
torch.save(model.state_dict(), "fakeddit_6way_model.pth")

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["True", "Satire", "Misleading", "False Context", "Imposter", "Manipulated"], yticklabels=["True", "Satire", "Misleading", "False Context", "Imposter", "Manipulated"])
plt.title('6-Way Classification Confusion Matrix', weight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_matrix_6way.png', dpi=300)
print("Saved confusion matrix to plots/confusion_matrix_6way.png")