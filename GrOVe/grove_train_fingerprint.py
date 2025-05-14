
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime

# === Define Datasets ===
dataset_info = {
    'Cora': lambda: Planetoid(root='pyg_data/Cora', name='Cora'),
    'CiteSeer': lambda: Planetoid(root='pyg_data/CiteSeer', name='CiteSeer'),
    'PubMed': lambda: Planetoid(root='pyg_data/PubMed', name='PubMed'),
    'AmazonComputers': lambda: Amazon(root='pyg_data/AmazonComputers', name='Computers'),
    'AmazonPhotos': lambda: Amazon(root='pyg_data/AmazonPhotos', name='Photo'),
    'CoauthorCS': lambda: Coauthor(root='pyg_data/CoauthorCS', name='CS'),
    'CoauthorPhysics': lambda: Coauthor(root='pyg_data/CoauthorPhysics', name='Physics'),
}

# === Fingerprint Classifier (Csim) ===
class FingerprintClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# === Generate Embedding Distance Vectors ===
def generate_distance_vectors(embedding_a, embedding_b):
    return torch.abs(embedding_a - embedding_b)

# === Train and Evaluate Fingerprint Classifier ===
def train_csim(distance_vectors, labels, device):
    classifier = FingerprintClassifier(distance_vectors.size(1)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(distance_vectors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return classifier

@torch.no_grad()
def evaluate_csim(classifier, distance_vectors, labels):
    classifier.eval()
    outputs = classifier(distance_vectors)
    preds = outputs.argmax(dim=1)
    accuracy = accuracy_score(labels.cpu(), preds.cpu())
    f1 = f1_score(labels.cpu(), preds.cpu())
    precision = precision_score(labels.cpu(), preds.cpu())
    recall = recall_score(labels.cpu(), preds.cpu())
    auroc = roc_auc_score(labels.cpu(), outputs.softmax(dim=1)[:, 1].cpu())
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auroc
    }

# === Main Experiment ===
results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset_name in dataset_info:
    print(f"\\n📊 Training Fingerprint Classifier (Csim) on: {dataset_name}")

    for model_name in ['GraphSAGE']:
        print(f"🔧 Loading embeddings for {model_name} on {dataset_name}...")

        # Load target embeddings
        target_path = f'trained_models/GROVE/Target/{model_name}/{dataset_name}/run1.pt'
        target_model = torch.load(target_path, map_location=device)
        target_embeddings = target_model['conv1.weight']

        # Load surrogate and independent embeddings
        surrogates, independents = [], []

        for run in range(1, 4):
            surrogate_path = f'trained_models/GROVE/Surrogate/{model_name}/{dataset_name}/run{run}.pt'
            surrogate_model = torch.load(surrogate_path, map_location=device)
            surrogates.append(surrogate_model['conv1.weight'])

            independent_path = f'trained_models/GROVE/Independent/{model_name}/{dataset_name}/run{run}.pt'
            independent_model = torch.load(independent_path, map_location=device)
            independents.append(independent_model['conv1.weight'])

        # Generate Distance Vectors
        positive_vectors = [generate_distance_vectors(target_embeddings, s) for s in surrogates]
        negative_vectors = [generate_distance_vectors(target_embeddings, i) for i in independents]

        # Prepare Training Data
        distance_vectors = torch.cat(positive_vectors + negative_vectors)
        labels = torch.cat([
            torch.ones(len(positive_vectors)), 
            torch.zeros(len(negative_vectors))
        ]).long().to(device)

        # Train Fingerprint Classifier (Csim)
        csim = train_csim(distance_vectors, labels, device)

        # Evaluate Fingerprint Classifier
        metrics = evaluate_csim(csim, distance_vectors, labels)
        metrics.update({'dataset': dataset_name, 'model': model_name})
        results.append(metrics)
        print(f"✅ Fingerprint Classifier Metrics: {metrics}")

# Save results
df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'grove_fingerprint_classifier_metrics_{timestamp}.csv', index=False)
print("🎉 Fingerprint Classifier Training Complete.")

