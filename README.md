# WaveGC: Wavelet Graph Convolutional Networks

<p align="center">
<a href="#about">About</a> •
<a href="#key-features">Key Features</a> •
<a href="#installation">Installation</a> •
<a href="#usage">Usage</a> •
<a href="#configuration">Configuration</a> •
<a href="#project-structure">Structure</a> •
<a href="#credits">Credits</a> •
<a href="#license">License</a>
</p>

<p align="center">
<img src="[https://img.shields.io/badge/python-3.9+-blue.svg](https://www.google.com/search?q=https://img.shields.io/badge/python-3.9%2B-blue.svg)" alt="Python Version">
<img src="[https://img.shields.io/badge/pytorch-2.0+-orange.svg](https://www.google.com/search?q=https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)" alt="PyTorch Version">
<img src="[https://img.shields.io/badge/license-MIT-green.svg](https://www.google.com/search?q=https://img.shields.io/badge/license-MIT-green.svg)" alt="License">
</p>

## About

**WaveGC** is a PyTorch-based Deep Learning library implementing **Wavelet Graph Convolutional Networks (WaveGCNet)**.

This project is an implementation of the paper **"A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition"**. It provides a flexible and modular framework for training and evaluating graph neural networks that utilize spectral wavelet transforms and tight frames for efficient graph representation learning.

The architecture is designed to handle complex graph data by leveraging spectral properties (eigenvectors and eigenvalues) and wavelet scattering transforms, enabling robust node classification and link prediction tasks.

**Authors:** Sergei Gerasimov and Soumodeep Hoodaty.

## Key Features

* **WaveGCNet Architecture:** Implementation of the Wavelet Graph Convolution network with customizable heads, scales, and polynomial approximations ().
* **Spectral Processing:** Built-in support for handling Laplacian eigenvectors (`eigvs`) and spectral matrices (`U`) for advanced positional encodings.
* **Modular Design:** powered by [Hydra](https://hydra.cc/) for hierarchical configuration management.
* **Experiment Tracking:** Integrated with **WandB** for logging metrics and visualizing training progress.
* **Reproducibility:** seeded random number generators and strict configuration versioning.
* **Developer Friendly:** Includes `pre-commit` hooks, type hinting, and a clean project structure based on modern PyTorch templates.

## Installation

### Prerequisites

* Python  3.9
* CUDA (optional, for GPU acceleration)

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/gerasimovsergey2001/wavegc.git
cd wavegc

```


2. **Create a virtual environment (Recommended):**
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# OR using Conda
conda create -n wavegc python=3.9
conda activate wavegc

```


3. **Install dependencies:**
This project uses a `pyproject.toml` for dependency management.
```bash
pip install .
# OR manually
pip install -r requirements.txt

```


4. **Install Pre-commit hooks (for developers):**
```bash
pre-commit install

```



## Usage

The project uses `train.py` for training models and `inference.py` for evaluation.

### Training

To train the WaveGC model, use the `train.py` script. You can specify the configuration file using the `-cn` (config name) flag.

```bash
# Train using the default configuration
python3 train.py

# Train using a specific experiment config (e.g., WaveGC on Amazon Photo)
python3 train.py -cn=wavegc_amazon

```

You can override any parameter from the command line using Hydra syntax:

```bash
# Override model parameters (e.g., embedding dimension and dropout)
python3 train.py model.emb_dim=64 model.dropout=0.5

# Change the dataset
python3 train.py dataset=amazon

```

### Inference

To evaluate a trained model or generate predictions:

```bash
python3 inference.py model_path="path/to/checkpoint.pth"

```

## Configuration

Configurations are managed via [Hydra](https://hydra.cc/) and stored in the `src/configs` directory. The structure is as follows:

* **`src/configs/train_config.yaml`**: Main entry point for training configuration.
* **`src/configs/model/`**: Model architectures (e.g., `wavegc.yaml`, `baseline.yaml`).
* **`src/configs/datasets/`**: Dataset definitions (e.g., `amazon.yaml`).
* **`src/configs/trainer/`**: Trainer settings (optimizer, scheduler, epochs).
* **`src/configs/logger/`**: WandB/CometML settings.

### Example: WaveGC Model Config (`src/configs/model/wavegc.yaml`)

```yaml
_target_: src.model.wavegc.WaveGCNet
emb_dim: 32
heads_num: 4
num_layers: 4
mpnn: "gcn"
K: 6       # Polynomial order
J: 3       # Wavelet scale
tight_frames: True

```

## Project Structure

```
wavegc/
├── data/                   # Dataset storage
├── src/
│   ├── configs/            # Hydra configuration files
│   ├── datasets/           # PyG dataset implementations (Amazon, etc.)
│   ├── layer/              # Custom layers (WaveGC, Encoders)
│   ├── model/              # Full model architectures (WaveGCNet)
│   ├── trainer/            # Training and inference loops
│   ├── utils/              # Utility scripts (logging, seeding)
│   └── ...
├── train.py                # Main training script
├── inference.py            # Main inference script
├── pyproject.toml          # Project metadata and dependencies
└── README.md

```

## Credits

This repository was developed by **Sergei Gerasimov** and **Soumodeep Hoodaty**.

* **Original Paper:** *A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition*.
* **Original Repository:** The official implementation by the paper authors can be found at [https://github.com/liun-online/WaveGC](https://github.com/liun-online/WaveGC).

This project utilizes the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) for structure and best practices.

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
