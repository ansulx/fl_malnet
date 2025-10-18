#!/usr/bin/env python3
"""
Run Federated Learning Experiment on M1
=======================================
Simple script to run the full federated learning experiment optimized for M1.
"""

import torch
import sys
import os
import time

# Add project root to path
sys.path.append('.')

def main():
    """Run the federated learning experiment"""
    print("🚀 STARTING FEDERATED LEARNING EXPERIMENT")
    print("=" * 50)
    print("Optimized for M1 MacBook Air")
    print("=" * 50)
    
    # Check M1 setup
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    try:
        # Import and run the research experiment
        from experiments.research_experiment import run_research_experiment
        
        print("📊 Starting research experiment...")
        print("This will run:")
        print("  - Graph data loading")
        print("  - Model training")
        print("  - Federated learning rounds")
        print("  - Privacy mechanisms")
        print("  - Performance evaluation")
        print()
        
        # Run the experiment
        start_time = time.time()
        results = run_research_experiment()
        end_time = time.time()
        
        print("\n" + "=" * 50)
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"📊 Results saved to: results/experiment_results.json")
        print(f"📋 Report saved to: results/experiment_report.txt")
        
        # Show key results
        if 'final_accuracy' in results['metrics']:
            final_acc = results['metrics']['final_accuracy'][-1]
            print(f"🎯 Final accuracy: {final_acc:.2f}%")
        
        print("\n✅ READY FOR SUPERVISOR PRESENTATION!")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
