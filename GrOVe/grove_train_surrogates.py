
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

model_classes = {
    'GraphSAGE': GraphSAGE,
}

# === Load Target Model and Generate Surrogates ===
def load_target_model(model_class, model_path, input_dim, output_dim):
    model = model_class(input_dim, 128, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_surrogate_data(target_model, data):
    target_model.eval()
    with torch.no_grad():
        embeddings = target_model.conv1(data.x, data.edge_index).relu()
    return embeddings

# === Training and Evaluation ===
def train_surrogate(data, embeddings, model_class, device):
    model = model_class(embeddings.size(1), 128, data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(embeddings.to(device), data.edge_index.to(device))
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model

# === Main Loop ===
results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset_name, dataset_loader in dataset_info.items():
    print(f"\\n📊 Generating Surrogate Models for: {dataset_name}")
    dataset = dataset_loader()
    base_data = dataset[0]

    # Enforce 60/20/20 train/val/test splits
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

    for model_name in model_classes:
        for run in range(1, 4):
            print(f"🚀 {model_name} | Generating Surrogate {run}/3 on {dataset_name}")
            
            # Load the corresponding target model
            target_model_path = f'trained_models/GROVE/Target/{model_name}/{dataset_name}/run{run}.pt'
            target_model = load_target_model(GraphSAGE, target_model_path, base_data.num_node_features, base_data.y.max().item() + 1).to(device)

            # Generate embeddings from target model
            embeddings = generate_surrogate_data(target_model, base_data.to(device))

            # Train surrogate using these embeddings
            surrogate_model = train_surrogate(base_data, embeddings, GraphSAGE, device)

            # Save the surrogate model
            model_dir = f'trained_models/GROVE/Surrogate/{model_name}/{dataset_name}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'run{run}.pt')
            torch.save(surrogate_model.state_dict(), model_path)

            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'run': run,
                'model_path': model_path
            })

            print(f"✅ Surrogate Model {run} for {model_name} on {dataset_name} saved.")

# Save all results
df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'grove_surrogate_models_{timestamp}.csv', index=False)
print("🎉 Surrogate Model Generation Complete.")
