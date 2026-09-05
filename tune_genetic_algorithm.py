import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import random
import copy

# ================= 1. DEFINE THE SEARCH SPACE =================
PARAM_SPACE = {
    'lr': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    'batch_size': [256, 512, 1024, 2048],
    'hidden1': [128, 256, 512],
    'hidden2': [32, 64, 128],
    'dropout1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'dropout2': [0.1, 0.2, 0.3, 0.4]
}

# ================= 2. DYNAMIC ARCHITECTURE =================
class DynamicMLP(nn.Module):
    def __init__(self, h1, h2, d1, d2, input_dim=1025, num_classes=6):
        super(DynamicMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(d1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(d2),
            nn.Linear(h2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ================= 3. FITNESS EVALUATION =================
def evaluate_fitness(params, train_loader, val_loader, device):
    """Trains the configuration for 3 epochs and returns the Validation F1 Score."""
    model = DynamicMLP(params['hidden1'], params['hidden2'], params['dropout1'], params['dropout2']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=(device=='cuda'))
    
    # Train for a very short burst (3 epochs) just to gauge learning capacity
    model.train()
    for epoch in range(3):
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device=='cuda')):
                outputs = model(bx)
                loss = criterion(outputs, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    # Evaluate Fitness
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device=='cuda')):
                outputs = model(bx)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(by.numpy())
            
    return f1_score(targets, preds, average='macro')

# ================= 4. GENETIC ALGORITHM LOGIC =================
def create_individual():
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}

def crossover(parent1, parent2):
    """Uniform crossover: 50% chance to inherit each gene from either parent."""
    child = {}
    for key in PARAM_SPACE.keys():
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

def mutate(individual, mutation_rate=0.2):
    """Randomly alters a gene to introduce new genetic material."""
    mutated = copy.deepcopy(individual)
    for key in PARAM_SPACE.keys():
        if random.random() < mutation_rate:
            mutated[key] = random.choice(PARAM_SPACE[key])
    return mutated

# ================= 5. EVOLUTION LOOP =================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Genetic Algorithm on {device}...")
    
    # 1. Load Data
    print("Loading full dataset...")
    data = torch.load("fakeddit_features_full.pt", weights_only=False)
    X = np.array([np.concatenate([i["text_vector"], i["image_vector"], [i["cosine_similarity"]]]) for i in data], dtype=np.float32)
    y = np.array([i["label_6way"] for i in data], dtype=np.int64)
    
    # 2. Isolate a tiny 5% proxy dataset for rapid fitness evaluation
    X_proxy, _, y_proxy, _ = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)
    X_t, X_v, y_t, y_v = train_test_split(X_proxy, y_proxy, test_size=0.2, stratify=y_proxy, random_state=42)
    
    X_val_t = torch.tensor(X_v, dtype=torch.float32)
    y_val_t = torch.tensor(y_v, dtype=torch.long)
    
    # 3. GA Parameters
    POP_SIZE = 10
    GENERATIONS = 5
    MUTATION_RATE = 0.25
    
    population = [create_individual() for _ in range(POP_SIZE)]
    best_overall = None
    best_fitness = 0.0
    
    print("\nStarting Evolution...")
    for gen in range(GENERATIONS):
        print(f"\n--- GENERATION {gen + 1}/{GENERATIONS} ---")
        
        # Evaluate Fitness for entire population
        fitness_scores = []
        for i, ind in enumerate(population):
            # Dynamic DataLoader based on individual's batch size gene
            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_t, dtype=torch.float32), torch.tensor(y_t, dtype=torch.long)), 
                batch_size=ind['batch_size'], shuffle=True, pin_memory=True
            )
            val_loader = DataLoader(
                TensorDataset(X_val_t, y_val_t), 
                batch_size=ind['batch_size'], shuffle=False, pin_memory=True
            )
            
            fit_val = evaluate_fitness(ind, train_loader, val_loader, device)
            fitness_scores.append((fit_val, ind))
            print(f"Ind {i+1}: F1 = {fit_val:.4f} | {ind}")
            
        # Sort by fitness descending
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_overall = fitness_scores[0][1]
            
        # Elitism: Keep the top 2 exactly as they are
        new_population = [fitness_scores[0][1], fitness_scores[1][1]]
        
        # Breed the rest
        while len(new_population) < POP_SIZE:
            # Tournament selection (pick 2 random, take the best)
            tournament = random.sample(fitness_scores[:5], 2)
            parent1 = tournament[0][1]
            tournament = random.sample(fitness_scores[:5], 2)
            parent2 = tournament[0][1]
            
            child = crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)
            
        population = new_population

    print("\n================ EVOLUTION COMPLETE ================")
    print(f"Optimal Configuration Found (Validation F1: {best_fitness:.4f}):")
    for k, v in best_overall.items():
        print(f"  {k}: {v}")
    print("\nNext Step: Plug these exact values into your train_kfold.py script!")