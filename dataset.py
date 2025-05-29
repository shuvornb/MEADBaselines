import os

import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor


class CustomDataset:
    def __init__(self, name: str, root: str = "./dataset"):
        self.name = name.lower()
        self.root = root
        self.dataset = self._load_dataset()
        self.data = self.dataset[0]  # For all these datasets, usually only one graph
        self._add_info()

    def _load_dataset(self):
        if self.name in ['cora', 'citeseer', 'pubmed']:
            return self._load_planetoid()
        elif self.name in ['amazon-photo', 'amazon-computers']:
            return self._load_amazon()
        elif self.name in ['coauthor-cs', 'coauthor-physics']:
            return self._load_coauthor()
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")

    def get(self) -> Dataset:
        return self.data

    def _load_planetoid(self):
        name = self.name.split('-')[-1].capitalize()
        dataset = Planetoid(root=os.path.join(self.root, self.name), name=name)
        data = dataset[0]
        self._split_planetoid_masks(data)
        dataset.data = data
        return dataset

    def _split_planetoid_masks(self, data):
        """
        Retain original val/test masks.
        For each class, select up to 100 training samples from the remaining nodes.
        """
        labels = data.y
        num_classes = int(labels.max().item()) + 1
        num_nodes = labels.size(0)

        val_mask = data.val_mask
        test_mask = data.test_mask

        excluded_mask = val_mask | test_mask
        available_mask = ~excluded_mask

        train_idx = []
        generator = torch.Generator().manual_seed(42)

        print("Training samples per class:")

        for c in range(num_classes):
            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            class_available = class_idx[available_mask[class_idx]]
            n_select = min(100, class_available.size(0))

            if n_select == 0:
                print(f"  Class {c}: 0 available samples (skipped)")
                continue

            perm = class_available[torch.randperm(class_available.size(0), generator=generator)]
            selected = perm[:n_select]
            train_idx.append(selected)

            print(f"  Class {c}: selected {n_select} samples")

        train_idx = torch.cat(train_idx, dim=0) if train_idx else torch.tensor([], dtype=torch.long)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        data.train_mask = train_mask

    def _load_amazon(self):
        name = self.name.split('-')[-1].capitalize()
        dataset = Amazon(root=os.path.join(self.root, self.name), name=name)
        data = dataset[0]
        self._split_masks(data)
        dataset.data = data
        return dataset

    def _load_coauthor(self):
        name = self.name.split('-')[-1].capitalize()
        dataset = Coauthor(root=os.path.join(self.root, self.name), name=name)
        data = dataset[0]
        self._split_masks(data)
        dataset.data = data
        return dataset

    def _split_masks(self, data):
        """
        For Amazon and Coauthor datasets:
        - train: 100 per class
        - val: 500 from remaining nodes
        - test: 1000 from remaining nodes after val
        """
        labels = data.y
        num_classes = int(labels.max().item()) + 1
        num_nodes = data.num_nodes

        train_idx = []
        used_mask = torch.zeros(num_nodes, dtype=torch.bool)
        generator = torch.Generator().manual_seed(42)

        # 1. Select 100 nodes per class for training
        print("Training samples per class:")
        for c in range(num_classes):
            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            perm = class_idx[torch.randperm(class_idx.size(0), generator=generator)]
            n_select = min(100, perm.size(0))
            selected = perm[:n_select]
            train_idx.append(selected)
            used_mask[selected] = True
            print(f"  Class {c}: selected {n_select} samples")

        train_idx = torch.cat(train_idx, dim=0)

        # 2. Select 500 nodes for validation from unused nodes
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        remaining_perm = remaining_idx[torch.randperm(remaining_idx.size(0), generator=generator)]

        val_idx = remaining_perm[:500]
        used_mask[val_idx] = True

        # 3. Select 1000 nodes for test from remaining
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        test_idx = remaining_idx[:1000]

        # Convert to masks
        data.train_mask = self._index_to_mask(train_idx, num_nodes)
        data.val_mask = self._index_to_mask(val_idx, num_nodes)
        data.test_mask = self._index_to_mask(test_idx, num_nodes)

    def _index_to_mask(self, index, size):
        mask = torch.zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

    def _add_info(self):
        self.data.num_features = self.dataset.num_features
        self.data.num_classes = self.dataset.num_classes
        self.data.name = self.dataset.name

    def stats(self):
        print('Dataset Stats')
        print('data: ', self.data)
        print(
            f'train num: {self.data.train_mask.sum()}, val num: {self.data.val_mask.sum()}, test num: {self.data.test_mask.sum()}')


class IndependentDataset:
    def __init__(self, name: str, root: str = "./dataset"):
        self.name = name.lower()
        self.root = root

    def _load_dataset(self, num_class_samples, seed):
        if self.name in ['cora', 'citeseer', 'pubmed']:
            return self._load_planetoid(num_class_samples, seed)
        elif self.name in ['amazon-photo', 'amazon-computers']:
            return self._load_amazon(num_class_samples, seed)
        elif self.name in ['coauthor-cs', 'coauthor-physics']:
            return self._load_coauthor(num_class_samples, seed)
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")

    def generate(self, num_class_samples=100, seed=42):
        """
        Re-generate train/val/test split with new parameters.
        """
        self.dataset = self._load_dataset(num_class_samples, seed)
        self.data = self.dataset[0]  # For all these datasets, usually only one graph
        self._add_info()
        return self.data

    # def get(self) -> Dataset:
    #     return self.data

    def _load_planetoid(self, num_class_samples, seed):
        name = self.name.split('-')[-1].capitalize()
        dataset = Planetoid(root=os.path.join(self.root, self.name), name=name)
        data = dataset[0]
        self._split_planetoid_masks(data, num_class_samples, seed)
        dataset.data = data
        return dataset

    def _split_planetoid_masks(self, data, num_class_samples=100, seed=42):
        """
        Retain original val/test masks.
        From the remaining nodes (not in val/test), randomly select a portion as new train_mask.
        """
        train_ratio = 0.5

        num_nodes = data.y.size(0)
        val_mask = data.val_mask
        test_mask = data.test_mask

        # Nodes that are not in val/test
        available_mask = ~(val_mask | test_mask)
        available_indices = available_mask.nonzero(as_tuple=True)[0]

        # Select a portion of available nodes
        num_classes = int(data.y.max().item()) + 1
        num_train = int(num_classes * num_class_samples)
        generator = torch.Generator().manual_seed(seed)
        perm = available_indices[torch.randperm(available_indices.size(0), generator=generator)]
        selected = perm[:num_train]

        # Create new train_mask
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[selected] = True

        data.train_mask = train_mask

    def _split_planetoid_masks_by_class_num(self, data, num_class_samples=100, seed=42):
        """
        Retain original val/test masks.
        For each class, select up to `num_class_samples` training samples from the remaining nodes.
        """
        labels = data.y
        num_classes = int(labels.max().item()) + 1
        num_nodes = labels.size(0)

        val_mask = data.val_mask
        test_mask = data.test_mask
        train_mask = data.train_mask

        excluded_mask = val_mask | test_mask | train_mask  # attention, exclude train
        available_mask = ~excluded_mask

        train_idx = []
        generator = torch.Generator().manual_seed(seed)

        print("Training samples per class:")

        for c in range(num_classes):
            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            class_available = class_idx[available_mask[class_idx]]
            n_select = min(num_class_samples, class_available.size(0))

            if n_select == 0:
                print(f"  Class {c}: 0 available samples (skipped)")
                continue

            perm = class_available[torch.randperm(class_available.size(0), generator=generator)]
            selected = perm[:n_select]
            train_idx.append(selected)

            print(f"  Class {c}: selected {n_select} samples")

        train_idx = torch.cat(train_idx, dim=0) if train_idx else torch.tensor([], dtype=torch.long)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        data.train_mask = train_mask

    def _load_amazon(self, num_class_samples, seed):
        name = self.name.split('-')[-1].capitalize()
        dataset = Amazon(root=os.path.join(self.root, self.name), name=name)
        data = dataset[0]
        self._split_masks(data, num_class_samples, seed)
        dataset.data = data
        return dataset

    def _load_coauthor(self, num_class_samples, seed):
        name = self.name.split('-')[-1].capitalize()
        dataset = Coauthor(root=os.path.join(self.root, self.name), name=name)
        data = dataset[0]
        self._split_masks(data, num_class_samples, seed)
        dataset.data = data
        return dataset

    def _split_masks(self, data, num_class_samples=100, seed=42):
        """
        For Amazon and Coauthor datasets:
        - train: `num_class_samples` per class
        - val: 500 from remaining nodes
        - test: 1000 from remaining nodes after val
        """
        labels = data.y
        num_classes = int(labels.max().item()) + 1
        num_nodes = data.num_nodes

        train_idx = []
        used_mask = torch.zeros(num_nodes, dtype=torch.bool)
        generator = torch.Generator().manual_seed(seed)

        print("Training samples per class:")
        for c in range(num_classes):
            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            perm = class_idx[torch.randperm(class_idx.size(0), generator=generator)]
            n_select = min(num_class_samples, perm.size(0))
            selected = perm[:n_select]
            train_idx.append(selected)
            used_mask[selected] = True
            print(f"  Class {c}: selected {n_select} samples")

        train_idx = torch.cat(train_idx, dim=0)

        # 2. Select 500 nodes for validation from unused nodes
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        remaining_perm = remaining_idx[torch.randperm(remaining_idx.size(0), generator=generator)]

        val_idx = remaining_perm[:500]
        used_mask[val_idx] = True

        # 3. Select 1000 nodes for test from remaining
        remaining_idx = (~used_mask).nonzero(as_tuple=True)[0]
        test_idx = remaining_idx[:1000]

        # Convert to masks
        data.train_mask = self._index_to_mask(train_idx, num_nodes)
        data.val_mask = self._index_to_mask(val_idx, num_nodes)
        data.test_mask = self._index_to_mask(test_idx, num_nodes)

    def _index_to_mask(self, index, size):
        mask = torch.zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

    def _add_info(self):
        self.data.num_features = self.dataset.num_features
        self.data.num_classes = self.dataset.num_classes
        self.data.name = self.dataset.name

    def stats(self):
        print('Dataset Stats')
        print('data: ', self.data)
        print(
            f'train num: {self.data.train_mask.sum()}, val num: {self.data.val_mask.sum()}, test num: {self.data.test_mask.sum()}')
