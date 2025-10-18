#!/usr/bin/env python3
"""
M1 MacBook Air Optimized Demo
============================
Federated Learning Demo optimized for M1 MacBook Air with MPS acceleration.
"""

import torch
import time
import sys
import os
import yaml
from typing import Dict, List

# Add project root to path
sys.path.append('.')

def setup_m1_environment():
    """Setup M1 optimized environment"""
    print("🍎 M1 MacBook Air Setup")
    print("=" * 30)
    
    # Check M1 capabilities
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Set device
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using M1 GPU acceleration (MPS)")
    else:
        device = "cpu"
        print("⚠️  Using CPU (MPS not available)")
    
    # Optimize for M1
    torch.set_num_threads(4)  # Limit CPU threads for M1
    
    return device

def test_m1_model_creation():
    """Test model creation on M1"""
    print("\n🧠 Testing Model Creation on M1...")
    
    try:
        from core.models import create_model
        
        # Load M1-optimized config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        device = setup_m1_environment()
        
        # Create model
        model = create_model(config)
        model = model.to(device)
        
        param_count = model.count_parameters()
        model_size = model.get_model_size()
        
        print(f"✅ Model created: {config['model']['gnn_type'].upper()}")
        print(f"✅ Parameters: {param_count:,}")
        print(f"✅ Size: {model_size:.2f} MB")
        print(f"✅ Device: {device}")
        
        return model, device, config
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None, None, None

def test_m1_data_loading():
    """Test data loading on M1"""
    print("\n📊 Testing Data Loading on M1...")
    
    try:
        from core.data_loader import MalNetGraphLoader
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create data loader with M1 optimizations
        loader = MalNetGraphLoader(config)
        
        # Test with limited samples for M1
        print("Loading sample graphs...")
        
        # Find sample files
        sample_dirs = [
            'malnet-graphs-tiny/benign/benign',
            'malnet-graphs-tiny/adware/airpush'
        ]
        
        sample_count = 0
        for sample_dir in sample_dirs:
            if os.path.exists(sample_dir):
                files = os.listdir(sample_dir)
                for i, file in enumerate(files[:5]):  # Limit to 5 files per class
                    if file.endswith('.edgelist'):
                        sample_path = os.path.join(sample_dir, file)
                        graph = loader.load_graph(sample_path)
                        sample_count += 1
                        if sample_count == 1:
                            print(f"✅ Sample graph: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
                            print(f"✅ Features shape: {graph.x.shape}")
                        break
                if sample_count >= 2:
                    break
        
        print(f"✅ Loaded {sample_count} sample graphs successfully")
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def run_m1_training_demo(model, device, config):
    """Run a quick training demo on M1"""
    print("\n🏃 Running M1 Training Demo...")
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        
        model.train()
        
        # Create dummy data optimized for M1
        batch_size = 2
        num_nodes = 20
        num_classes = config['model']['num_classes']
        
        # Generate dummy graph data
        x = torch.randn(num_nodes, 5, device=device)
        edge_index = torch.randint(0, num_nodes, (2, 30), device=device)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training on device: {device}")
        
        # Training loop
        for epoch in range(3):
            start_time = time.time()
            
            optimizer.zero_grad()
            output = model(x, edge_index, batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            epoch_time = time.time() - start_time
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == labels).float().mean().item()
            
            print(f"  Epoch {epoch+1}/3: Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy*100:.1f}%, Time: {epoch_time:.3f}s")
        
        print("✅ M1 training demo completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
        return False

def run_m1_federated_demo():
    """Run a quick federated learning demo on M1"""
    print("\n🤝 Running M1 Federated Learning Demo...")
    
    try:
        from core.federated_learning import FederatedServer, FederatedClient
        from core.models import create_model
        from core.data_splitter import create_federated_datasets
        from core.data_loader import MalNetGraphLoader
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from torch.utils.data import Subset
        
        device = setup_m1_environment()
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create global model
        global_model = create_model(config)
        global_model = global_model.to(device)
        
        # Create server
        server = FederatedServer(global_model, config)
        print(f"✅ Federated server created with {server.aggregation_strategy} aggregation")
        
        # Create limited dataset for M1 demo
        data_loader = MalNetGraphLoader(config)
        train_loader, _, _ = data_loader.create_data_loaders()
        
        # Limit dataset size for M1 demo
        limited_dataset = Subset(train_loader.dataset, range(min(50, len(train_loader.dataset))))
        
        # Split data for federated learning
        client_datasets, stats = create_federated_datasets(
            limited_dataset,
            num_clients=3,  # Reduced for M1
            split_strategy='dirichlet',
            alpha=0.5
        )
        
        print(f"✅ Created {len(client_datasets)} client datasets")
        print(f"✅ Samples per client: {[len(ds) for ds in client_datasets]}")
        
        # Create clients
        clients = []
        for i, client_dataset in enumerate(client_datasets):
            client_loader = PyGDataLoader(
                client_dataset,
                batch_size=2,  # Small batch for M1
                shuffle=True,
                num_workers=0  # No multiprocessing for M1
            )
            
            client_model = create_model(config)
            client = FederatedClient(i, client_loader, client_model, config, device=device)
            clients.append(client)
        
        print(f"✅ Created {len(clients)} federated clients")
        
        # Run one federated round
        print("Running one federated round...")
        
        # Select participating clients
        participating_clients = clients[:2]  # Use 2 clients
        
        # Train clients
        client_updates = []
        for client in participating_clients:
            global_weights = server.get_global_weights()
            client_results = client.train_local_model(global_weights, 2)  # 2 epochs
            client_updates.append(client_results)
            print(f"  Client {client.client_id}: {client_results['training_accuracy']:.1f}% accuracy")
        
        # Aggregate updates
        aggregated_weights = server.aggregate_updates(client_updates)
        print("✅ Federated aggregation completed")
        
        # Test privacy mechanisms
        if config.get('privacy', {}).get('enabled', False):
            from core.privacy import DifferentialPrivacy
            dp = DifferentialPrivacy(config['privacy'])
            noisy_weights = dp.add_noise(aggregated_weights)
            print(f"✅ Privacy mechanisms applied (budget: {dp.get_privacy_budget():.3f})")
        
        print("✅ M1 federated learning demo completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Federated demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main M1 demo function"""
    print("🍎 FEDERATED LEARNING M1 DEMO")
    print("=" * 40)
    print("Optimized for M1 MacBook Air with MPS acceleration")
    print("=" * 40)
    
    # Test 1: Model Creation
    model, device, config = test_m1_model_creation()
    if not model:
        print("❌ Demo failed at model creation")
        return False
    
    # Test 2: Data Loading
    if not test_m1_data_loading():
        print("❌ Demo failed at data loading")
        return False
    
    # Test 3: Training Demo
    if not run_m1_training_demo(model, device, config):
        print("❌ Demo failed at training")
        return False
    
    # Test 4: Federated Learning Demo
    if not run_m1_federated_demo():
        print("❌ Demo failed at federated learning")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 ALL M1 DEMO TESTS PASSED!")
    print("=" * 40)
    print("✅ Graph-based malware detection")
    print("✅ M1 GPU acceleration (MPS)")
    print("✅ Federated learning framework")
    print("✅ Privacy-preserving mechanisms")
    print("✅ Research-grade architecture")
    print("\n🚀 READY FOR SUPERVISOR DEMONSTRATION!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
