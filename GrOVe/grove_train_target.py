
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import pandas as pd
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

# === Define Models ===
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

model_classes = {
    'GraphSAGE': GraphSAGE,
    'GCN': GCN,
    'GAT': GAT,
}

# === Training and Evaluation ===
def train_model(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_model(data, model):
    model.eval()
    out = model(data.x, data.edge_index)
    logits = out[data.test_mask]
    preds = logits.argmax(dim=1).cpu()
    labels = data.y[data.test_mask].cpu()
    return (preds == labels).sum().item() / len(labels)

# === Main Training Loop ===
results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset_name, dataset_loader in dataset_info.items():
    print(f"\\n📊 Training Target Models on: {dataset_name}")
    dataset = dataset_loader()
    base_data = dataset[0]

    # Enforce 60/20/20 train/val/test split
    num_nodes = base_data.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    base_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    base_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    base_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    base_data.train_mask[perm[:train_size]] = True
    base_data.val_mask[perm[train_size:train_size+val_size]] = True
    base_data.test_mask[perm[train_size+val_size:]] = True

    for model_name, model_class in model_classes.items():
        for run in range(1, 4):
            print(f"🚀 {model_name} | Run {run}/3 on {dataset_name} starting...")
            model = model_class(dataset.num_node_features, 128, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            for epoch in range(1, 201):
                train_model(base_data.to(device), model, optimizer)
                if epoch % 50 == 0:
                    print(f"  Epoch {epoch}...")

            print(f"✅ Target Model trained for {model_name} run {run} on {dataset_name}")

            # Save the trained model
            model_dir = f'trained_models/GROVE/Target/{model_name}/{dataset_name}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'run{run}.pt')
            torch.save(model.state_dict(), model_path)

            acc = evaluate_model(base_data.to(device), model)
            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'run': run,
                'accuracy': acc,
                'model_path': model_path
            })
            print(f"📈 Node Classification Accuracy: {acc:.4f}")

# Save results
df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'grove_target_model_metrics_{timestamp}.csv', index=False)
print("🎉 Target Model Training Complete.")
