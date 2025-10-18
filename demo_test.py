#!/usr/bin/env python3
"""
Quick Demo Test for Federated Learning Project
=============================================
Simple test script to demonstrate the system works for supervisor.
"""

import torch
import sys
import os
import time

# Add project root to path
sys.path.append('.')

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    try:
        import torch
        import torch_geometric
        import numpy as np
        import yaml
        print("✅ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🧠 Testing model creation...")
    try:
        from core.models import create_model
        import yaml
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = create_model(config)
        param_count = model.count_parameters()
        model_size = model.get_model_size()
        
        print(f"✅ Model created: {config['model']['gnn_type'].upper()}")
        print(f"✅ Parameters: {param_count:,}")
        print(f"✅ Size: {model_size:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_data_loading():
    """Test data loading with limited samples"""
    print("\n📊 Testing data loading...")
    try:
        from core.data_loader import MalNetGraphLoader
        import yaml
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test with a single graph file
        loader = MalNetGraphLoader(config)
        
        # Find a sample file
        sample_paths = [
            'malnet-graphs-tiny/benign/benign',
            'malnet-graphs-tiny/adware/airpush',
            'malnet-graphs-tiny/trojan/artemis'
        ]
        
        sample_file = None
        for path in sample_paths:
            if os.path.exists(path):
                files = os.listdir(path)
                if files:
                    sample_file = os.path.join(path, files[0])
                    break
        
        if sample_file:
            graph = loader.load_graph(sample_file)
            print(f"✅ Graph loaded: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
            print(f"✅ Features shape: {graph.x.shape}")
            return True
        else:
            print("❌ No sample files found")
            return False
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_privacy_mechanisms():
    """Test privacy mechanisms"""
    print("\n🔒 Testing privacy mechanisms...")
    try:
        from core.privacy import DifferentialPrivacy
        
        config = {
            'epsilon': 1.0,
            'delta': 1e-5,
            'noise_multiplier': 1.1,
            'max_grad_norm': 1.0
        }
        
        dp = DifferentialPrivacy(config)
        
        # Test noise addition
        dummy_weights = {
            'test.weight': torch.randn(5, 3),
            'test.bias': torch.randn(3)
        }
        
        noisy_weights = dp.add_noise(dummy_weights)
        budget = dp.get_privacy_budget()
        
        print(f"✅ Differential privacy initialized")
        print(f"✅ Noise addition successful")
        print(f"✅ Privacy budget: {budget:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Privacy mechanism error: {e}")
        return False

def test_federated_components():
    """Test federated learning components"""
    print("\n🤝 Testing federated components...")
    try:
        from core.federated_learning import FederatedServer
        from core.models import create_model
        import yaml
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model and server
        model = create_model(config)
        server = FederatedServer(model, config)
        
        # Test aggregation with dummy updates
        dummy_updates = [
            {'weights': model.get_weights(), 'sample_count': 10},
            {'weights': model.get_weights(), 'sample_count': 8}
        ]
        
        aggregated = server.aggregate_updates(dummy_updates)
        
        print(f"✅ Federated server initialized")
        print(f"✅ Weight aggregation successful")
        print(f"✅ Aggregation strategy: {server.aggregation_strategy}")
        return True
        
    except Exception as e:
        print(f"❌ Federated components error: {e}")
        return False

def run_quick_training_test():
    """Run a quick training test"""
    print("\n🏃 Running quick training test...")
    try:
        from core.models import create_model
        import yaml
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = create_model(config)
        model.train()
        
        # Create dummy data
        batch_size = 2
        num_nodes = 10
        dummy_x = torch.randn(num_nodes, 5)
        dummy_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        dummy_batch = torch.zeros(num_nodes, dtype=torch.long)
        dummy_labels = torch.randint(0, 5, (batch_size,))
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training step
        optimizer.zero_grad()
        output = model(dummy_x, dummy_edge_index, dummy_batch)
        loss = criterion(output, dummy_labels)
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful")
        print(f"✅ Loss: {loss.item():.4f}")
        print(f"✅ Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Training test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FEDERATED LEARNING PROJECT DEMO TEST")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_model_creation,
        test_data_loading,
        test_privacy_mechanisms,
        test_federated_components,
        run_quick_training_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)  # Small delay for readability
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for supervisor demonstration.")
        print("\n📋 READY FOR DEMO:")
        print("✅ Graph-based malware detection")
        print("✅ Federated learning framework")
        print("✅ Privacy-preserving mechanisms")
        print("✅ Research-grade architecture")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
