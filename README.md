# Kolmogorov-Arnold Networks (KANs) for 3D Point Cloud Classification

This repository contains a KAN-based version of Dynamic Graph Convolutional Neural Network (DGCNN) tailored for 3D point cloud classification. The project demonstrates the performance of  KANDdgcnn on the ModelNet40 dataset.

For computational efficiency, this model uses a single `EdgeKANconv` layer, though additional layers can be used as needed.

## Getting Started

### Dataset

To download the ModelNet40 dataset, visit [this GitHub repository](https://github.com/antao97/PointCloudDatasets).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/said-ohamouddou/KANdgcnn/
   cd KANdgcnn
   ```

2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  
### Usage

To execute the model, run the following command:

```bash
python main.py --exp_name=kan_dgcnn_1024 --model=KANdgcnn --num_points=1024 --k=5 --use_sgd=True
```
We use Weights & Biases (wandb) for visualization.

### Results

In its current configuration (one `EdgKANconv` layer, one KAN layer in the output and k=5 for the local neighborhood graph construction), the model achieves:

- **Test Accuracy**: 0.914
- **Balanced Accuracy**: 0.884

These results were obtained after training the model on a GeForce RTX 4060 Ti (16 GB VRAM) for 250 epochs on Ubuntu 20.04 LTS. The model begins to overfit after approximately 150 epochs.

#### Visualizations

![Training Accuracy vs Epochs](metrics/accuracy.svg)  ![Validation Accuracy vs Epochs](metrics/loss.svg)

### Acknowledgments

Special thanks to the following projects and repositories that made this work possible:

- [pyKAN by KindXiaoming](https://github.com/KindXiaoming/pykan): The first implementation of KAN
- [DGCNN by WangYueFt](https://github.com/WangYueFt/dgcnn): PyTorch DGCNN implementation
- [torch-conv-KAN by IvanDrokin](https://github.com/IvanDrokin/torch-conv-kan): Efficient convolutional KAN implementation


