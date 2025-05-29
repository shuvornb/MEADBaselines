import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, FAConv, GCN2Conv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import networkx as nx
import random
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dataset import CustomDataset

# === Dataset Names to Load ===
dataset_names = [
    'cora', 'citeseer', 'pubmed',
    'amazon-computers', 'amazon-photo',
    'coauthor-cs', 'coauthor-physics'
]

# === Model Definitions ===
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn1 = GCNConv(in_feats, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GCN model parameters reset')
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)

        embedding = x

        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}

class GCN2(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5, alpha=0.1, theta=0.5, layer_num=2):
        super(GCN2, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.theta = theta
        self.layer_num = layer_num

        self.initial_proj = nn.Linear(in_feats, hidden_dim)
        self.convs = nn.ModuleList([
            GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i + 1) for i in range(layer_num)
        ])
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GCN2 model parameters reset')
        self.initial_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = F.relu(self.initial_proj(x))

        x_tmp = x0
        for conv in self.convs:
            x_tmp = F.dropout(x_tmp, p=self.dropout, training=self.training)
            x_tmp = F.relu(conv(x_tmp, x0, edge_index))

        embedding = x_tmp
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.sage1 = SAGEConv(in_feats, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GraphSAGE model parameters reset')
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)

        embedding = x
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}

class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gat1 = GATConv(in_feats, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GAT parameters reset')
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        embedding = x

        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}

class FAGCN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5):
        super(FAGCN, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_proj = nn.Linear(in_feats, hidden_dim)
        self.faconv1 = FAConv(hidden_dim)
        self.faconv2 = FAConv(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('FAGCN model parameters reset')
        self.input_proj.reset_parameters()
        self.faconv1.reset_parameters()
        self.faconv2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x_0 = x.clone()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.faconv1(x, x_0, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.faconv2(x, x_0, edge_index)

        embedding = x
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}

model_classes = {
    'GCN': GCN,
    'GraphSAGE': GraphSAGE,
    'GAT': GAT,
    'FAGCN': FAGCN,
    'GCN2' : GCN2
}

# === Trigger Graph Generator ===
def generate_trigger_graph(num_nodes=10, edge_prob=0.1, p_feat=0.1, num_classes=7, feat_dim=1433):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.zeros((num_nodes, feat_dim))
    for i in range(num_nodes):
        ones_idx = torch.randperm(feat_dim)[:int(p_feat * feat_dim)]
        x[i, ones_idx] = 1
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# === Complete Workflow: Training & Evaluation with Progress Bar ===
def train_and_evaluate_all(runs=3):
    results = []
    for dataset_name in dataset_names:
        print(f"\n📚 Starting {dataset_name} experiments")
        for model_name, model_class in model_classes.items():
            for run in range(runs):
                data = CustomDataset(name=dataset_name).get().to("cuda")

                model = model_class(data.num_features, out_feats=data.num_classes, hidden_dim=128).to("cuda")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                # Trigger Graph
                trigger = generate_trigger_graph(num_nodes=10, feat_dim=data.num_features, num_classes=data.num_classes).to("cuda")

                # Training
                for epoch in tqdm(range(200), desc=f"Training {dataset_name} {model_name} Run {run + 1}"):
                    model.train()
                    optimizer.zero_grad()
                    logits, _ = model(data.x, data.edge_index)
                    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()

                # Evaluation
                model.eval()
                logits, outputs = model(data.x, data.edge_index)
                y_true = data.y[data.test_mask].cpu()
                y_pred = outputs['hard_label'][data.test_mask].cpu()

                metrics = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'run': run + 1,
                    'accuracy': accuracy_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred, average='macro'),
                    'precision': precision_score(y_true, y_pred, average='macro'),
                    'recall': recall_score(y_true, y_pred, average='macro'),
                    'auroc': roc_auc_score(y_true, outputs['soft_label'][data.test_mask].detach().cpu().numpy(), multi_class='ovo')
                }

                results.append(metrics)

    df = pd.DataFrame(results)
    summary = df.groupby(['dataset', 'model']).agg(['mean', 'std']).reset_index()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'randomwm_allmodels_results_{timestamp}.csv', index=False)
    summary.to_csv(f'randomwm_allmodels_summary_{timestamp}.csv', index=False)
    print("🎉 All watermark training and evaluation complete.")

train_and_evaluate_all(runs=3)