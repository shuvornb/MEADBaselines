
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
from datetime import datetime
import networkx as nx
import random

# === Dataset loaders ===
dataset_info = {
    'Cora': lambda: Planetoid(root='pyg_data/Cora', name='Cora'),
    'CiteSeer': lambda: Planetoid(root='pyg_data/CiteSeer', name='CiteSeer'),
    'PubMed': lambda: Planetoid(root='pyg_data/PubMed', name='PubMed'),
    'AmazonComputers': lambda: Amazon(root='pyg_data/AmazonComputers', name='Computers'),
    'AmazonPhotos': lambda: Amazon(root='pyg_data/AmazonPhotos', name='Photo'),
    'CoauthorCS': lambda: Coauthor(root='pyg_data/CoauthorCS', name='CS'),
    'CoauthorPhysics': lambda: Coauthor(root='pyg_data/CoauthorPhysics', name='Physics'),
}

# === GNN Models ===
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

# === Soft Nearest Neighbor Loss ===
def snn_loss(x, y, T=0.5):
    x = F.normalize(x, p=2, dim=1)
    dist_matrix = torch.cdist(x, x, p=2) ** 2
    eye = torch.eye(len(x), device=x.device).bool()
    sim = torch.exp(-dist_matrix / T)
    mask_same = y.unsqueeze(1) == y.unsqueeze(0)
    sim = sim.masked_fill(eye, 0)
    denom = sim.sum(1)
    nom = (sim * mask_same.float()).sum(1)
    loss = -torch.log(nom / (denom + 1e-10) + 1e-10).mean()
    return loss

# === Trigger Graph Generator ===
def generate_key_graph(base_data, num_nodes=10, edge_prob=0.3):
    trigger = nx.erdos_renyi_graph(num_nodes, edge_prob)
    edge_index = torch.tensor(list(trigger.edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x = torch.randn((num_nodes, base_data.num_node_features)) * 0.1
    label = torch.randint(0, base_data.y.max().item() + 1, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=label)

# === Combine base and trigger ===
def combine_with_trigger(base_data, trigger_data, device):
    base_data = base_data.to(device)
    trigger_data = trigger_data.to(device)

    x = torch.cat([base_data.x, trigger_data.x], dim=0)
    edge_index = torch.cat([base_data.edge_index, trigger_data.edge_index + base_data.x.size(0)], dim=1)
    y = torch.cat([base_data.y, trigger_data.y], dim=0)

    train_mask = torch.cat([base_data.train_mask, torch.ones(trigger_data.num_nodes, dtype=torch.bool, device=device)])
    val_mask = torch.cat([base_data.val_mask, torch.zeros(trigger_data.num_nodes, dtype=torch.bool, device=device)])
    test_mask = torch.cat([base_data.test_mask, torch.zeros(trigger_data.num_nodes, dtype=torch.bool, device=device)])

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


# === Metric Evaluation ===
def compute_metrics(y_true, y_pred, y_score=None):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'auroc': roc_auc_score(y_true, y_score, multi_class='ovo') if y_score is not None else None
    }

def train_with_snnl(model, data, optimizer, T=0.5, alpha=0.1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss_nll = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    snnl = snn_loss(out[data.train_mask], data.y[data.train_mask], T=T)
    loss = loss_nll - alpha * snnl
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_model(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)[data.test_mask]
    pred = logits.argmax(dim=1)
    probs = logits.exp()
    return compute_metrics(data.y[data.test_mask].cpu(), pred.cpu(), probs.cpu())

@torch.no_grad()
def verify_watermark(model, trigger):
    model.eval()
    out = model(trigger.x, trigger.edge_index)
    return (out.argmax(dim=1) == trigger.y).float().mean().item()

# === Main Experiment Loop ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []

for dataset_name, dataset_loader in dataset_info.items():
    dataset = dataset_loader()
    base_data = dataset[0]
    print(f"\\n📊 Dataset: {dataset_name} | Nodes: {base_data.num_nodes} | Features: {base_data.num_node_features} | Classes: {dataset.num_classes}")

    perm = torch.randperm(base_data.num_nodes)
    train_size = int(0.6 * base_data.num_nodes)
    val_size = int(0.2 * base_data.num_nodes)
    base_data.train_mask = torch.zeros(base_data.num_nodes, dtype=torch.bool)
    base_data.val_mask = torch.zeros(base_data.num_nodes, dtype=torch.bool)
    base_data.test_mask = torch.zeros(base_data.num_nodes, dtype=torch.bool)
    base_data.train_mask[perm[:train_size]] = True
    base_data.val_mask[perm[train_size:train_size+val_size]] = True
    base_data.test_mask[perm[train_size+val_size:]] = True
    print(f"  ➕ Train: {train_size}, Val: {val_size}, Test: {base_data.num_nodes - train_size - val_size}")


    for model_name, model_class in model_classes.items():
        for run in range(1, 4):
            model = model_class(base_data.num_node_features, 128, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            trigger = generate_key_graph(base_data).to(device)
            combined = combine_with_trigger(base_data, trigger, device)

            for epoch in range(1, 201):
                train_with_snnl(model, combined, optimizer)

            metrics = evaluate_model(model, combined)
            wm_acc = verify_watermark(model, trigger)
            model_dir = f"trained_models/SurviveWM/{model_name}/{dataset_name}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"run{run}.pt")
            torch.save(model.state_dict(), model_path)
            metrics.update({
                'dataset': dataset_name,
                'model': model_name,
                'run': run,
                'watermark_acc': wm_acc,
                'model_path': model_path,
                'num_nodes': base_data.num_nodes,
                'num_features': base_data.num_node_features,
                'num_classes': dataset.num_classes,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': base_data.num_nodes - train_size - val_size
            })
            results.append(metrics)
            print(f"[{dataset_name}][{model_name}] Run {run} complete. Model saved.")

# Save results
df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'survive_multimodel_results_{timestamp}.csv', index=False)
print("✅ All experiments completed.")
