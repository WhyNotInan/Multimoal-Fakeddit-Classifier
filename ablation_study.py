import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading full dataset into RAM (This happens only once)...")
data = torch.load("fakeddit_features_full.pt", weights_only=False)

# 1. Fast Vectorized Extraction
print("Formatting isolated modalities...")
text_list, image_list, cos_list, y_list = [], [], [], []

for item in data:
    text_list.append(item["text_vector"])
    image_list.append(item["image_vector"])
    cos_list.append([item["cosine_similarity"]])
    y_list.append(item["label_6way"])

X_text = np.array(text_list, dtype=np.float32)
X_image = np.array(image_list, dtype=np.float32)
X_cos = np.array(cos_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

# 2. Define the Test Matrix
ablation_matrix = {
    "Text Only": {"data": X_text, "dim": 512},
    "Image Only": {"data": X_image, "dim": 512},
    "Text + Image (No Cosine)": {"data": np.concatenate([X_text, X_image], axis=1), "dim": 1024},
    "Baseline (Multimodal Full)": {"data": np.concatenate([X_text, X_image, X_cos], axis=1), "dim": 1025}
}

# 3. Dynamic Architecture
class FastFakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FastFakeNewsClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )
    def forward(self, x):
        return self.net(x)

# 4. Automated Evaluation Engine
def run_kfold_evaluation(X_data, input_dim, variant_name):
    print(f"\n================ STARTING: {variant_name} ================")
    K, EPOCHS, BATCH_SIZE = 5, 15, 1024
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y), 1):
        X_train_t = torch.tensor(X_data[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
        X_val_t = torch.tensor(X_data[val_idx], dtype=torch.float32)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        model = FastFakeNewsClassifier(input_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Train Loop
        model.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    outputs = model(bx)
                    loss = criterion(outputs, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Validation Loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    outputs = model(bx)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(by.numpy())

        acc = accuracy_score(val_targets, val_preds) * 100
        fold_accuracies.append(acc)
        print(f"  {variant_name} - Fold {fold}: {acc:.2f}%")

    mean_acc = np.mean(fold_accuracies)
    print(f"--> {variant_name} Final Mean Accuracy: {mean_acc:.2f}%")
    return mean_acc

# 5. Execute the Matrix
results = {}
for name, config in ablation_matrix.items():
    results[name] = run_kfold_evaluation(config["data"], config["dim"], name)

# 6. Generate Comparative Plot
print("\nGenerating Ablation Study Chart...")
plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), color=['#e07a5f', '#f2cc8f', '#81b29a', '#3d405b'])

plt.axhline(results["Baseline (Multimodal Full)"], color='red', linestyle='--', alpha=0.5, label='Multimodal Baseline')
plt.title('Ablation Study: Impact of Modalities on Fakeddit Classification', fontsize=14, fontweight='bold')
plt.ylabel('Mean Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=15, ha='right', fontsize=11)
plt.legend()

# Annotate exact percentages
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/5_ablation_study_comparison.png", dpi=300)
print("\nAblation study complete! Chart saved to 'plots/5_ablation_study_comparison.png'.")