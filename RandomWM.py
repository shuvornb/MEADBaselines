
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import networkx as nx
import random
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

# === Trigger Graph Generation ===
def generate_trigger_graph(num_nodes=10, edge_prob=0.1, p_feat=0.1, num_classes=7, feat_dim=1433):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    x = torch.zeros((num_nodes, feat_dim))
    for i in range(num_nodes):
        ones_idx = torch.randperm(feat_dim)[:int(p_feat * feat_dim)]
        x[i, ones_idx] = 1
    y = torch.tensor([random.randint(0, num_classes - 1) for _ in range(num_nodes)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# === Combine Graphs ===
def combine_graphs(base_data, trigger_data):
    x = torch.cat([base_data.x, trigger_data.x], dim=0)
    edge_index = torch.cat([
        base_data.edge_index,
        trigger_data.edge_index + base_data.x.size(0)
    ], dim=1)
    y = torch.cat([base_data.y, trigger_data.y], dim=0)
    train_mask = torch.cat([
        base_data.train_mask,
        torch.ones(trigger_data.num_nodes, dtype=torch.bool)
    ])
    val_mask = torch.cat([
        base_data.val_mask,
        torch.zeros(trigger_data.num_nodes, dtype=torch.bool)
    ])
    test_mask = torch.cat([
        base_data.test_mask,
        torch.zeros(trigger_data.num_nodes, dtype=torch.bool)
    ])
    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# === Evaluation Metrics ===
def compute_metrics(y_true, y_pred, y_score=None):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'auroc': roc_auc_score(y_true, y_score, multi_class='ovo') if y_score is not None else None
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
    probs = logits.exp().cpu()
    return compute_metrics(labels, preds, probs)

@torch.no_grad()
def verify_trigger(trigger, model):
    model.eval()
    device = next(model.parameters()).device
    out = model(trigger.x.to(device), trigger.edge_index.to(device))
    pred = out.argmax(dim=1)
    acc = (pred == trigger.y.to(device)).sum().item() / trigger.num_nodes
    return acc

# === Main Loop ===
results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset_name, dataset_loader in dataset_info.items():
    print(f"\\n📚 Starting experiments on dataset: {dataset_name}")
    dataset = dataset_loader()
    base_data = dataset[0]

    print(f"🔍 Dataset Stats: Nodes={base_data.num_nodes}, Features={base_data.num_node_features}, Classes={dataset.num_classes}")
    '''
    if not hasattr(base_data, 'train_mask'):
        print(f"  ⚙️  No masks found — generating custom train/val/test splits (60/20/20)")
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
    else:
        print(f"  ⚙️  Using existing predefined masks.")
    '''

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



    print(f"  ➡️ Train nodes: {base_data.train_mask.sum().item()}, Val nodes: {base_data.val_mask.sum().item()}, Test nodes: {base_data.test_mask.sum().item()}")

    for model_name, model_class in model_classes.items():
        for run in range(1, 4):
            print(f"🚀 {model_name} | Run {run}/3 on {dataset_name} starting...")

            trigger = generate_trigger_graph(num_classes=dataset.num_classes, feat_dim=dataset.num_node_features)
            combined_data = combine_graphs(base_data, trigger).to(device)

            model = model_class(dataset.num_node_features, 128, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            for epoch in range(1, 201):
                train_model(combined_data, model, optimizer)
                if epoch % 50 == 0:
                    print(f"  Epoch {epoch}: training...")

            print(f"✅ Training completed for {model_name} run {run} on {dataset_name}")

            metrics = evaluate_model(combined_data, model)
            watermark_acc = verify_trigger(trigger, model)

            model_dir = f'trained_models/RandomWM/{model_name}/{dataset_name}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'run{run}.pt')
            torch.save(model.state_dict(), model_path)

            metrics.update({
                'dataset': dataset_name,
                'model': model_name,
                'run': run,
                'watermark_acc': watermark_acc,
                'model_path': model_path
            })
            results.append(metrics)

            print(f"📈 Testing completed for {model_name} run {run} on {dataset_name}")

# Save all results
df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'random_multimodel_results_{timestamp}.csv', index=False)
print("🎉 All watermark training and evaluation complete.")
