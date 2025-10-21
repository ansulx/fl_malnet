# 🏗️ Clean Federated Learning Architecture

## Simplified, Modular, Research-Grade Design

### Core Principle: KISS (Keep It Simple, Stupid)

```
┌─────────────────────────────────────────────────────────────┐
│                      🖥️  CENTRAL SERVER                     │
│  - Global Model (GNN)                                        │
│  - Aggregates updates from devices                           │
│  - Visible in terminal dashboard                             │
└─────────────────────────────────────────────────────────────┘
                            ↕️ ↕️ ↕️
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 📱 Device 1 │    │ 📱 Device 2 │    │ 📱 Device 3 │
│ Local Model │    │ Local Model │    │ Local Model │
│ Local Data  │    │ Local Data  │    │ Local Data  │
│ Trains      │    │ Trains      │    │ Trains      │
│ Sends Update│    │ Sends Update│    │ Sends Update│
└─────────────┘    └─────────────┘    └─────────────┘
```

## Directory Structure (Simplified)

```
fl_malnet/
├── server/
│   ├── model.py          # Global GNN model
│   ├── aggregator.py     # FedAvg aggregation
│   ├── server.py         # Main server logic
│   └── dashboard.py      # Terminal visualization
│
├── device/
│   ├── client.py         # Client training logic
│   └── local_trainer.py  # Local model training
│
├── data/
│   ├── loader.py         # MalNet data loading
│   └── splitter.py       # Split data across devices
│
├── shared/
│   ├── model.py          # Shared GNN architecture
│   ├── config.py         # Configuration
│   └── utils.py          # Helper functions
│
├── visualization/
│   └── terminal_ui.py    # Real-time terminal dashboard
│
└── run_fl.py             # Main entry point
```

## Key Features

### ✅ What's Included:
1. **Single Server** - Central coordinator with global model
2. **Multiple Devices** - 5-10 devices with local data
3. **Real-Time Dashboard** - Terminal UI showing everything
4. **FedAvg** - Simple, proven aggregation
5. **GNN Model** - Graph neural network for malware
6. **Clear Logging** - Every step visible
7. **Modular Design** - Easy to understand and extend

### ❌ What's Removed:
- ❌ Complex meta-learning (MAML)
- ❌ Multiple aggregation strategies
- ❌ Advanced privacy mechanisms (keep basic DP)
- ❌ Too many baselines
- ❌ Confusing abstractions

## Terminal Dashboard Layout

```
╔════════════════════════════════════════════════════════════════════╗
║           🖥️  FEDERATED LEARNING SERVER - LIVE STATUS             ║
╚════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────┐
│ SERVER STATUS                                                       │
├────────────────────────────────────────────────────────────────────┤
│ Status: ● RUNNING        Round: 15/50        GPU: cuda:0          │
│ Global Model Accuracy: 87.3% ↑                                     │
│ Total Samples Trained: 25,340                                      │
│ Connected Devices: 5/5                                             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ CONNECTED DEVICES (Live Training)                                  │
├────────────────────────────────────────────────────────────────────┤
│ Device 1 [Hospital]      ● Training  Acc: 85.2%  Samples: 1,024  │
│ Device 2 [University]    ● Training  Acc: 88.1%  Samples: 978    │
│ Device 3 [Company]       ● Waiting   Acc: 86.5%  Samples: 1,102  │
│ Device 4 [Lab]           ● Training  Acc: 89.3%  Samples: 1,045  │
│ Device 5 [Institute]     ● Training  Acc: 87.7%  Samples: 991    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ CURRENT ROUND PROGRESS                                             │
├────────────────────────────────────────────────────────────────────┤
│ Phase: Local Training → Aggregation → Evaluation                  │
│ Progress: [████████████████░░░░] 80% (4/5 devices completed)      │
│                                                                    │
│ Recent Updates:                                                    │
│ [15:32:45] Device 2 completed training (Loss: 0.234)              │
│ [15:32:43] Device 1 completed training (Loss: 0.267)              │
│ [15:32:41] Device 4 started training...                           │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ MODEL IMPROVEMENT OVER TIME                                        │
├────────────────────────────────────────────────────────────────────┤
│ Accuracy:                                                          │
│ 90% │                                        ╭─●                   │
│     │                                   ╭────╯                     │
│ 85% │                          ╭────●───╯                         │
│     │                    ╭─────╯                                   │
│ 80% │            ╭───●───╯                                         │
│     │      ╭─────╯                                                 │
│ 75% │  ●───╯                                                       │
│     └──────────────────────────────────────────────────────────   │
│        R1  R3  R5  R7  R9  R11 R13 R15                            │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ SYSTEM METRICS                                                     │
├────────────────────────────────────────────────────────────────────┤
│ Communication: 3.2 MB/round  │  GPU Memory: 4.2 GB / 24 GB        │
│ Round Time: 45.3s            │  Privacy Budget: ε=1.0, δ=1e-5     │
└────────────────────────────────────────────────────────────────────┘
```

## Workflow

### 1. Server Startup
```bash
python run_fl.py --mode server --num-devices 5 --rounds 50
```

### 2. Device Connection
Devices automatically connect and receive:
- Initial global model
- Local data assignment
- Training instructions

### 3. Training Rounds
```
FOR each round:
    1. Server broadcasts model → Devices
    2. Devices train locally (visible in dashboard)
    3. Devices send updates → Server
    4. Server aggregates (FedAvg)
    5. Server evaluates global model
    6. Display results in dashboard
```

### 4. Continuous Monitoring
- Supervisor sees everything in real-time
- Every device status visible
- Model accuracy updated live
- Clear progress indicators

## Research-Grade Features (Simplified)

1. **Proper FL Protocol**
   - Correct FedAvg implementation
   - Privacy-preserving (basic DP)
   - Non-IID data distribution

2. **Graph Neural Networks**
   - GCN for malware detection
   - MalNet dataset (function call graphs)
   - Proper graph batching

3. **Comprehensive Logging**
   - Every round logged
   - Device-level metrics
   - Communication costs
   - Model checkpoints

4. **Evaluation**
   - Test set evaluation
   - Per-device performance
   - Convergence tracking
   - Privacy-utility tradeoff

## Configuration (Simple)

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8080
  device: "cuda"
  
model:
  type: "GCN"
  hidden_dim: 64
  num_layers: 3
  
federated:
  num_devices: 5
  num_rounds: 50
  local_epochs: 3
  
data:
  dataset: "malnet-graphs-tiny"
  batch_size: 16
  
visualization:
  refresh_rate: 1.0  # seconds
  show_graphs: true
```

## Why This Is Better

### Before (Complicated):
- ❌ Meta-learning, attention mechanisms, Byzantine-robust, etc.
- ❌ Multiple files, complex abstractions
- ❌ Hard to understand what's happening
- ❌ Supervisor confused

### After (Clean):
- ✅ Simple FedAvg (proven, understood)
- ✅ Clear architecture (server + devices)
- ✅ Visual dashboard (see everything)
- ✅ Modular code (easy to modify)
- ✅ Still research-grade (proper FL, GNN, evaluation)

## Research Value

This clean implementation is STILL research-grade because:

1. **Proper FL Implementation**
   - Correct algorithms
   - Privacy preservation
   - Realistic scenarios

2. **Real Problem**
   - Malware detection
   - Graph data
   - Federated setting

3. **Good Evaluation**
   - Multiple metrics
   - Convergence analysis
   - Communication costs

4. **Reproducible**
   - Clear code
   - Well documented
   - Easy to extend

## Supervisor Can See:

✅ Server running with global model
✅ 5 devices connected and training
✅ Real-time accuracy improvements
✅ Each device's contribution
✅ Model convergence
✅ System metrics

**Everything visible, nothing hidden!**

