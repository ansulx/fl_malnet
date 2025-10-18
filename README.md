# FL MalNet: Research-Grade Federated Learning for Malware Detection

A professional, research-grade federated learning framework for malware detection using graph neural networks on the MalNet dataset. Optimized for IIT Roorkee GPU servers.

## 🎯 Key Features

- **Research-Grade Architecture**: Professional modular design with clear separation of concerns
- **Graph-Based Detection**: Function call graphs for malware analysis using GNNs
- **Privacy-Preserving**: Differential privacy with ε=1.0 guarantees and secure aggregation
- **Multiple GNN Architectures**: GCN, GAT, SAGE, and lightweight models
- **Advanced Aggregation**: FedAvg, FedMedian, Krum for robust federated learning
- **GPU Optimized**: Configured for IIT Roorkee server with CUDA support
- **Research Metrics**: Comprehensive evaluation and monitoring

## 🏗️ Architecture

### Core Components

```
core/
├── data_loader.py          # Research-grade graph data loading
├── models.py              # GNN architectures (GCN, GAT, SAGE)
├── federated_learning.py   # Server and client implementations
├── privacy.py             # Differential privacy mechanisms
└── data_splitter.py       # Non-IID data distribution

experiments/
├── research_experiment.py  # Main experiment framework
├── graph_baseline.py      # Centralized baseline
├── graph_federated.py      # Federated learning
└── graph_quick_test.py     # Quick testing

config/
└── research_config.yaml   # Research configuration
```

### Model Architectures

- **ResearchGNN**: Advanced GNN with multiple layer types and pooling strategies
- **LightweightGNN**: Optimized for resource-constrained environments
- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks  
- **SAGE**: GraphSAGE networks

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
python -m venv fl_env
source fl_env/bin/activate  # On Windows: fl_env\Scripts\activate
```

### Configuration

Edit `config/research_config.yaml` for your setup:

```yaml
# Server Configuration (IIT Roorkee)
server:
  device: "cuda"
  gpu_ids: [0]  # Primary GPU
  mixed_precision: true
  max_memory_usage: 0.8  # 80% of GPU memory

# Model Configuration
model:
  gnn_type: "gcn"  # Options: gcn, gat, sage, lightweight
  hidden_dim: 128
  num_layers: 4
  dropout: 0.3

# Federated Learning
federated:
  num_clients: 10
  num_rounds: 50
  local_epochs: 10
  aggregation: "fedavg"
```

### Running Experiments

```bash
# Quick test
python experiments/graph_quick_test.py

# Research experiment
python experiments/research_experiment.py

# Baseline comparison
python experiments/graph_baseline.py
```

## 📊 Dataset

**MalNet Graphs**: Function call graphs from Android APKs
- **Classes**: 5 malware types (benign, adware, downloader, trojan, addisplay)
- **Format**: `.edgelist` files with node features
- **Size**: 5,000 samples per split (train/val/test)
- **Features**: Degree, in-degree, out-degree, clustering coefficient

## 🔬 Research Features

### Privacy Mechanisms
- **Differential Privacy**: ε-differential privacy with noise calibration
- **Secure Aggregation**: Cryptographic protocols for weight aggregation
- **Privacy Budget Tracking**: Real-time privacy guarantee monitoring

### Aggregation Strategies
- **FedAvg**: Standard federated averaging
- **FedMedian**: Robust to Byzantine attacks
- **Krum**: Byzantine-robust aggregation
- **WeightedFedAvg**: Sample-weighted averaging

### Evaluation Metrics
- **Accuracy**: Classification performance
- **Privacy Budget**: Privacy guarantee tracking
- **Communication Cost**: Bandwidth analysis
- **Convergence**: Training stability metrics

## 🖥️ Server Requirements

### IIT Roorkee Server Setup
- **GPU**: CUDA-compatible GPU (recommended: RTX 3080/4090)
- **Memory**: 16GB+ RAM, 8GB+ GPU memory
- **Storage**: 50GB+ free space
- **Python**: 3.8+ with PyTorch 1.12+

### Performance Optimization
- **Mixed Precision**: Automatic for GPU training
- **Memory Management**: Configurable GPU memory usage
- **Data Loading**: Optimized with persistent workers
- **Batch Processing**: Efficient graph batching

## 📈 Expected Results

### Baseline Performance
- **Centralized GCN**: 85-90% accuracy
- **Centralized GAT**: 87-92% accuracy
- **Centralized SAGE**: 86-91% accuracy

### Federated Learning
- **FedAvg**: 80-85% accuracy (10 clients)
- **FedMedian**: 82-87% accuracy (robust)
- **Privacy-Preserving**: 75-80% accuracy (ε=1.0)

### Communication Efficiency
- **Model Size**: 0.02-0.23 MB per model
- **Rounds**: 50 rounds for convergence
- **Bandwidth**: ~2-5 MB per round

## 🔧 Configuration

### Model Configuration
```yaml
model:
  gnn_type: "gcn"           # GNN architecture
  hidden_dim: 128           # Hidden dimension
  num_layers: 4             # Number of layers
  dropout: 0.3              # Dropout rate
  activation: "relu"        # Activation function
  normalization: "batch"    # Normalization type
  pooling: "mean_max"      # Global pooling
```

### Federated Learning
```yaml
federated:
  num_clients: 10           # Number of clients
  num_rounds: 50           # Training rounds
  local_epochs: 10         # Local training epochs
  aggregation: "fedavg"     # Aggregation strategy
  participation_rate: 0.8   # Client participation
  split_strategy: "dirichlet" # Data distribution
  alpha: 0.5               # Non-IID parameter
```

### Privacy Settings
```yaml
privacy:
  enabled: true             # Enable privacy mechanisms
  epsilon: 1.0             # Privacy parameter
  delta: 1e-5             # Privacy parameter
  noise_multiplier: 1.1    # Noise scaling
  max_grad_norm: 1.0      # Gradient clipping
```

## 📊 Results & Performance

### Current Implementation Status
- ✅ **Graph Dataset Loading**: MalNet graphs with function call analysis
- ✅ **GNN Models**: 4 architectures (GCN, GAT, SAGE, Lightweight)
- ✅ **Federated Learning**: Server-client architecture with aggregation
- ✅ **Privacy Mechanisms**: Differential privacy and secure aggregation
- ✅ **Research Framework**: Professional experiment management

### Expected Performance
- **Training Time**: 2-4 hours (50 rounds, 10 clients)
- **Memory Usage**: 4-8 GB GPU memory
- **Accuracy**: 80-90% on test set
- **Privacy**: ε=1.0 differential privacy guarantee

### Technical Specifications
- **Model Parameters**: 4K-60K parameters per model
- **Graph Processing**: Up to 2000 nodes per graph
- **Batch Size**: 16 samples per batch (GPU optimized)
- **Communication**: 2-5 MB per federated round

## 🚀 Deployment

### Local Development
```bash
# Quick test
python experiments/graph_quick_test.py

# Full experiment
python experiments/research_experiment.py
```

### Production Deployment
- **Docker**: Containerized deployment
- **Kubernetes**: Scalable orchestration
- **Monitoring**: Comprehensive logging and metrics
- **Security**: Encrypted communication and storage

## 📚 Research Applications

### Academic Research
- **Federated Learning**: Privacy-preserving collaborative learning
- **Malware Detection**: Graph-based threat analysis
- **Privacy**: Differential privacy in practice
- **Security**: Byzantine-robust aggregation

### Industry Applications
- **Enterprise Security**: Collaborative threat detection
- **Mobile Security**: Privacy-preserving malware analysis
- **IoT Security**: Distributed threat intelligence
- **Compliance**: GDPR-compliant data processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- **Issues**: GitHub Issues
- **Documentation**: README and code comments
- **Research**: Academic collaboration welcome

---

**Note**: This is a research-grade implementation optimized for academic and industrial research. For production deployment, additional security and monitoring features may be required.