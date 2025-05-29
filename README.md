
# Graph Watermarking Baselines

This repository provides baseline implementations for watermarking Graph Neural Networks (GNNs) to evaluate robustness against model extraction attacks. Each baseline is implemented in a standalone script and uses a unified dataset loading utility.

## 📂 Code Structure

- `dataset.py`: Utility module for loading and preparing graph datasets.
- `random_wm.py`: Implements **Random Trigger Graph Watermarking**.
- `backdoor_wm.py`: Implements **Backdoor-based Watermarking**.
- `survive_wm.py`: Implements the **SurviveWM** watermarking defense.
- `grove.py`: Implements the **GROVE** fingerprinting-based ownership verification method (does not alter model training).

## ⚙️ How to Run

Each Python script is standalone and can be run directly:

```bash
python random_wm.py
python backdoor_wm.py
python survive_wm.py
python grove.py
````

Each script:

* Loads datasets via `dataset.py`
* Trains on multiple GNN architectures
* Evaluates performance over 3 runs for each (dataset, model) combination
* Logs accuracy and watermark verification metrics

## 🧠 GNN Architectures Used

The following GNN backbones are supported and tested:

* **GCN** (Graph Convolutional Network)
* **GAT** (Graph Attention Network)
* **GraphSAGE**
* **GCNII**
* **FAGCN**

## 📁 Datasets

The following datasets are used across all experiments:

* `Cora`
* `CiteSeer`
* `PubMed`
* `AmazonComputers`
* `AmazonPhotos`
* `CoauthorCS`
* `CoauthorPhysics`

These datasets include citation networks, co-authorship networks, and product co-purchase graphs.

## 📊 Evaluation

Each baseline script reports the following metrics:

* **Node Classification Accuracy**
* **Fidelity** between clean and watermarked model predictions
* **Watermark Verification Accuracy** (for applicable methods)

Metrics are reported as mean ± standard deviation over 3 independent runs.

## 💡 Notes

* All scripts are GPU-enabled. Ensure CUDA is available on your system.
* Logs and results are saved automatically during execution.
* You may adapt the scripts for custom models or datasets as needed.

## 📜 Citation

If you use this repository in your research, please cite the appropriate papers for each baseline or acknowledge this GitHub project for comparison purposes.

