import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# 1. Load Data
print("Loading feature file into RAM...")
data = torch.load("fakeddit_features_full.pt", weights_only=False)

print("Formatting feature matrices...")
X_list, y_list = [], []

for item in data:
    txt = item["text_vector"]
    img = item["image_vector"]
    cos = np.array([item["cosine_similarity"]], dtype=np.float32)
    fused = np.concatenate([txt, img, cos])
    
    X_list.append(fused)
    y_list.append(item["label_6way"])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

CLASS_NAMES = [
    "True",
    "Satire/Parody",
    "Misleading Content",
    "Manipulated Content",
    "False Connection",
    "Completely Fabricated"
]

# 2. Classifier Architecture
class FastFakeNewsClassifier(nn.Module):
    def __init__(self, input_dim=1025, num_classes=6):
        super(FastFakeNewsClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 3. K-Fold Setup
K = 5
EPOCHS = 15
BATCH_SIZE = 256
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device} with {K}-Fold Cross-Validation")

fold_accuracies = []
fold_macro_f1s = []
all_oof_preds = []
all_oof_targets = []

# History containers for learning curves
train_loss_history = np.zeros((K, EPOCHS))
val_loss_history = np.zeros((K, EPOCHS))
train_acc_history = np.zeros((K, EPOCHS))
val_acc_history = np.zeros((K, EPOCHS))

scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

best_overall_acc = 0.0
best_model_path = "best_model.pt"

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n==================== FOLD {fold}/{K} ====================")
    
    X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
    X_val_t = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = FastFakeNewsClassifier(input_dim=1025, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Initialize Early Stopping parameters for the current fold
    patience = 3 
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_fold_model_path = "best_loss_model.pt"

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                outputs = model(bx)
                loss = criterion(outputs, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * bx.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == by).sum().item()
            train_total += by.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        train_loss_history[fold - 1, epoch - 1] = epoch_train_loss
        train_acc_history[fold - 1, epoch - 1] = epoch_train_acc

        # Epoch Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    outputs = model(bx)
                    loss = criterion(outputs, by)

                val_loss += loss.item() * bx.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += by.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        # (Existing code: epoch_val_loss = val_loss / val_total)
        # (Existing code: epoch_val_acc = (val_correct / val_total) * 100)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the model exactly at the lowest loss epoch
            torch.save(model.state_dict(), best_fold_model_path)
            print(f"    -> Val loss dropped to {best_val_loss:.4f}. Saving weights.")
        else:
            epochs_no_improve += 1
            print(f"    -> Val loss did not improve ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"    -> Early Stopping triggered at Epoch {epoch}!")
                break # Halts training for this fold to prevent further overfitting
        val_loss_history[fold - 1, epoch - 1] = epoch_val_loss
        val_acc_history[fold - 1, epoch - 1] = epoch_val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        # Load the weights from the epoch with the lowest validation loss
    model.load_state_dict(torch.load(best_fold_model_path, weights_only=True))

    # Final Fold Validation Extraction
    model.eval()
    fold_val_preds, fold_val_targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                outputs = model(bx)
            preds = torch.argmax(outputs, dim=1)
            fold_val_preds.extend(preds.cpu().numpy())
            fold_val_targets.extend(by.numpy())

    acc = accuracy_score(fold_val_targets, fold_val_preds) * 100
    f1 = f1_score(fold_val_targets, fold_val_preds, average='macro')
    fold_accuracies.append(acc)
    fold_macro_f1s.append(f1)
    
    all_oof_preds.extend(fold_val_preds)
    all_oof_targets.extend(fold_val_targets)

    if acc > best_overall_acc:
        print(f"  --> New Best Model Found in Fold {fold} (Acc: {acc:.2f}%). Saving weights...")
        best_overall_acc = acc
        torch.save(model.state_dict(), best_model_path)
    
    print(f"Fold {fold} Result -> Accuracy: {acc:.2f}% | Macro F1: {f1:.4f}")

print("\n================ FINAL K-FOLD SUMMARY ================")
print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.2f}% (+/- {np.std(fold_accuracies):.2f}%)")
print(f"Mean Macro F1: {np.mean(fold_macro_f1s):.4f} (+/- {np.std(fold_macro_f1s):.4f})")

# ================= HIGH-RESOLUTION EXPORTS =================
print("\nGenerating and saving presentation-ready plots to './plots'...")

epochs_range = np.arange(1, EPOCHS + 1)

# 1. Learning Curves (Loss & Accuracy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss Subplot
mean_train_loss = np.mean(train_loss_history, axis=0)
mean_val_loss = np.mean(val_loss_history, axis=0)
std_val_loss = np.std(val_loss_history, axis=0)

ax1.plot(epochs_range, mean_train_loss, label='Train Loss', color='#1f77b4', lw=2)
ax1.plot(epochs_range, mean_val_loss, label='Val Loss', color='#d62728', lw=2)
ax1.fill_between(epochs_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='#d62728', alpha=0.15)
ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper right')

# Accuracy Subplot
mean_train_acc = np.mean(train_acc_history, axis=0)
mean_val_acc = np.mean(val_acc_history, axis=0)
std_val_acc = np.std(val_acc_history, axis=0)

ax2.plot(epochs_range, mean_train_acc, label='Train Accuracy', color='#1f77b4', lw=2)
ax2.plot(epochs_range, mean_val_acc, label='Val Accuracy', color='#2ca02c', lw=2)
ax2.fill_between(epochs_range, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color='#2ca02c', alpha=0.15)
ax2.set_title('Accuracy vs. Epochs', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig("plots/1_learning_curves.png", dpi=300)
plt.close('all')

# 2. Normalized Confusion Matrix
plt.figure(figsize=(9, 7))
cm = confusion_matrix(all_oof_targets, all_oof_preds, normalize='true') * 100
sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar_kws={'label': 'Percentage (%)'})
plt.title(f'Normalized Confusion Matrix (Overall {K}-Fold OOF)', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=11)
plt.ylabel('True Class', fontsize=11)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/2_normalized_confusion_matrix.png", dpi=300)
plt.close('all')

# 3. Per-Class Precision, Recall, and F1 Breakdown
report_dict = classification_report(all_oof_targets, all_oof_preds, target_names=CLASS_NAMES, output_dict=True)
classes = CLASS_NAMES
p = [report_dict[c]['precision'] for c in classes]
r = [report_dict[c]['recall'] for c in classes]
f = [report_dict[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, p, width, label='Precision', color='#4c72b0')
plt.bar(x, r, width, label='Recall', color='#55a868')
plt.bar(x + width, f, width, label='F1-Score', color='#c44e52')

plt.title('Per-Class Evaluation Metrics (Aggregated 5-Fold Validation)', fontsize=13, fontweight='bold')
plt.xticks(x, classes, rotation=25, ha='right', fontsize=10)
plt.ylabel('Score (0.0 - 1.0)', fontsize=11)
plt.ylim(0.0, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("plots/3_per_class_metrics.png", dpi=300)
plt.close('all')

# 4. Fold Performance Comparison (Accuracy vs. Macro F1)
plt.figure(figsize=(9, 5))
fold_labels = [f"Fold {i}" for i in range(1, K + 1)]
xf = np.arange(len(fold_labels))
wf = 0.35

plt.bar(xf - wf/2, fold_accuracies, wf, label='Accuracy (%)', color='#3470a3')
plt.bar(xf + wf/2, [score * 100 for score in fold_macro_f1s], wf, label='Macro F1 (x100)', color='#e07a5f')

plt.axhline(np.mean(fold_accuracies), color='#3470a3', linestyle='--', alpha=0.7, label=f'Mean Acc: {np.mean(fold_accuracies):.2f}%')
plt.axhline(np.mean(fold_macro_f1s) * 100, color='#e07a5f', linestyle=':', alpha=0.7, label=f'Mean F1: {np.mean(fold_macro_f1s):.4f}')

plt.xticks(xf, fold_labels)
plt.ylabel('Percentage / Scaled Score', fontsize=11)
plt.title(f'Cross-Validation Variance Across {K} Folds', fontsize=13, fontweight='bold')
plt.ylim(0, 100)
plt.legend(loc='lower right', ncol=2)
plt.tight_layout()
plt.savefig("plots/4_kfold_variance.png", dpi=300)
plt.close('all')

print("All 4 figures have been saved to the 'plots/' directory.")