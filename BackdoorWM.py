import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import random
from datetime import datetime

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

# === Models ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

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

model_classes = {'GCN': GCN, 'GraphSAGE': GraphSAGE, 'GAT': GAT}

# === Feature-based Trigger Injection ===
def inject_backdoor_trigger(data, trigger_rate=0.01, trigger_feat_val=0.99, l=20, target_label=0):
    num_nodes = data.num_nodes
    num_feats = data.num_node_features
    num_trigger_nodes = int(trigger_rate * num_nodes)

    trigger_nodes = random.sample(range(num_nodes), num_trigger_nodes)
    for node in trigger_nodes:
        feature_indices = random.sample(range(num_feats), l)
        data.x[node][feature_indices] = trigger_feat_val
        data.y[node] = target_label
    return data, trigger_nodes

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
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro'),
        'auroc': roc_auc_score(labels, probs, multi_class='ovo')
    }

@torch.no_grad()
def verify_backdoor(data, model, trigger_nodes, target_label):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[trigger_nodes] == target_label).sum().item()
    return correct / len(trigger_nodes)

# === Main ===
results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset_name, dataset_loader in dataset_info.items():
    print(f"\n📚 Starting experiments on dataset: {dataset_name}")
    dataset = dataset_loader()
    data = dataset[0].to(device)

    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[perm[:train_size]] = True
    data.val_mask[perm[train_size:train_size+val_size]] = True
    data.test_mask[perm[train_size+val_size:]] = True

    print(f"🔍 Dataset Stats: Nodes={data.num_nodes}, Features={data.num_node_features}, Classes={dataset.num_classes}")
    print(f"➡️ Train nodes: {data.train_mask.sum().item()}, Val nodes: {data.val_mask.sum().item()}, Test nodes: {data.test_mask.sum().item()}")

    for model_name, model_class in model_classes.items():
        for run in range(1, 4):
            print(f"🚀 {model_name} | Run {run}/3 on {dataset_name} starting...")

            poisoned_data = data.clone()
            poisoned_data, trigger_nodes = inject_backdoor_trigger(poisoned_data, 
                                                                   trigger_rate=0.01, 
                                                                   l=20,
                                                                   target_label=0)

            model = model_class(dataset.num_node_features, 128, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            for epoch in range(1, 201):
                train_model(poisoned_data, model, optimizer)
                if epoch % 50 == 0:
                    print(f"  Epoch {epoch}: training...")

            print(f"✅ Training completed for {model_name} run {run} on {dataset_name}")

            metrics = evaluate_model(poisoned_data, model)
            wm_acc = verify_backdoor(poisoned_data, model, trigger_nodes, target_label=0)

            model_dir = f'trained_models/BackdoorWM/{model_name}/{dataset_name}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'run{run}.pt')
            torch.save(model.state_dict(), model_path)

            metrics.update({
                'dataset': dataset_name,
                'model': model_name,
                'run': run,
                'watermark_acc': wm_acc,
                'model_path': model_path
            })
            results.append(metrics)

            print(f"📈 Testing completed for {model_name} run {run} on {dataset_name}")

# Save all results
output_df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_df.to_csv(f'backdoor_multimodel_results_{timestamp}.csv', index=False)
print("\n🎉 All watermark training and evaluation complete.")
