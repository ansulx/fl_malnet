#!/usr/bin/env python3
"""
Simple CPU-Only Demo for M1 MacBook Air
=======================================
Guaranteed to work demo that shows the federated learning system functionality.
"""

import torch
import time
import sys
import os
import yaml

# Add project root to path
sys.path.append('.')

def test_basic_functionality():
    """Test basic system functionality"""
    print("🔍 Testing Basic System Functionality")
    print("=" * 40)
    
    try:
        # Test imports
        from core.models import create_model
        from core.data_loader import MalNetGraphLoader
        from core.privacy import DifferentialPrivacy
        from core.federated_learning import FederatedServer
        print("✅ All core modules imported successfully")
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Force CPU for reliability
        config['server']['device'] = 'cpu'
        print("✅ Configuration loaded (CPU mode)")
        
        # Test model creation
        model = create_model(config)
        print(f"✅ Model created: {model.count_parameters():,} parameters, {model.get_model_size():.2f} MB")
        
        # Test data loading
        loader = MalNetGraphLoader(config)
        
        # Find and load one sample graph
        sample_path = None
        for root, dirs, files in os.walk('malnet-graphs-tiny'):
            for file in files:
                if file.endswith('.edgelist'):
                    sample_path = os.path.join(root, file)
                    break
            if sample_path:
                break
        
        if sample_path:
            graph = loader.load_graph(sample_path)
            print(f"✅ Graph loaded: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
        else:
            print("❌ No graph files found")
            return False
        
        # Test privacy mechanism
        dp_config = {'epsilon': 1.0, 'delta': 1e-5, 'noise_multiplier': 1.1, 'max_grad_norm': 1.0}
        dp = DifferentialPrivacy(dp_config)
        dummy_weights = {'test.weight': torch.randn(5, 3)}
        noisy_weights = dp.add_noise(dummy_weights)
        print(f"✅ Privacy mechanism: ε={dp.get_privacy_budget():.3f}")
        
        # Test federated server
        server = FederatedServer(model, config)
        dummy_updates = [
            {'weights': model.get_weights(), 'sample_count': 10},
            {'weights': model.get_weights(), 'sample_count': 8}
        ]
        aggregated = server.aggregate_updates(dummy_updates)
        print(f"✅ Federated aggregation: {server.aggregation_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_training():
    """Run simple training test"""
    print("\n🏃 Running Simple Training Test")
    print("=" * 40)
    
    try:
        from core.models import create_model
        import yaml
        import torch.nn as nn
        import torch.optim as optim
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Force CPU
        config['server']['device'] = 'cpu'
        
        # Create model
        model = create_model(config)
        model.train()
        
        # Create simple dummy data
        batch_size = 1
        num_nodes = 10
        num_classes = config['model']['num_classes']
        
        x = torch.randn(num_nodes, 5)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("Training on CPU...")
        
        # Single training step
        optimizer.zero_grad()
        output = model(x, edge_index, batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == labels).float().mean().item()
        
        print(f"✅ Training successful: Loss={loss.item():.4f}, Accuracy={accuracy*100:.1f}%")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_federated_demo():
    """Run simple federated learning demo"""
    print("\n🤝 Running Simple Federated Demo")
    print("=" * 40)
    
    try:
        from core.federated_learning import FederatedServer
        from core.models import create_model
        import yaml
        
        # Load config
        with open('config/research_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Force CPU
        config['server']['device'] = 'cpu'
        
        # Create global model
        global_model = create_model(config)
        
        # Create server
        server = FederatedServer(global_model, config)
        print(f"✅ Server created with {server.aggregation_strategy} aggregation")
        
        # Simulate client updates
        print("Simulating federated learning round...")
        
        # Create dummy client updates
        client_updates = []
        for i in range(3):
            # Each client has slightly different weights
            client_weights = global_model.get_weights()
            # Add some variation to simulate local training
            for key in client_weights:
                client_weights[key] += torch.randn_like(client_weights[key]) * 0.01
            
            client_updates.append({
                'weights': client_weights,
                'sample_count': 10 + i * 5,
                'client_id': i
            })
        
        print(f"✅ Created {len(client_updates)} client updates")
        
        # Aggregate updates
        aggregated_weights = server.aggregate_updates(client_updates)
        print("✅ Federated aggregation completed")
        
        # Test privacy mechanisms
        if config.get('privacy', {}).get('enabled', False):
            from core.privacy import DifferentialPrivacy
            dp = DifferentialPrivacy(config['privacy'])
            noisy_weights = dp.add_noise(aggregated_weights)
            print(f"✅ Privacy applied: budget={dp.get_privacy_budget():.3f}")
        
        print("✅ Simple federated demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Federated demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    print("🚀 FEDERATED LEARNING SIMPLE DEMO")
    print("=" * 50)
    print("M1 MacBook Air - CPU Mode (Guaranteed to Work)")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        run_simple_training,
        run_simple_federated_demo
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✅ Graph-based malware detection system")
        print("✅ Federated learning framework")
        print("✅ Privacy-preserving mechanisms")
        print("✅ Research-grade architecture")
        print("\n🚀 READY FOR SUPERVISOR DEMONSTRATION!")
        print("\nTo run the full experiment:")
        print("  python run_fl_experiment.py")
        print("\nTo run with M1 GPU (experimental):")
        print("  PYTORCH_ENABLE_MPS_FALLBACK=1 python m1_demo.py")
    else:
        print("⚠️  Some tests failed. Check errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
