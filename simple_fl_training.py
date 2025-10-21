#!/usr/bin/env python3
"""
Simplified Federated Learning Training Script
==============================================
Direct implementation without the ResearchExperiment class to avoid segfaults.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
from torch_geometric.loader import DataLoader as PyGDataLoader

# Load configuration
with open('config/research_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Starting Federated Learning Training")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# Step 1: Load Data
print("\n📊 Loading data...")
from core.data_loader import MalNetGraphLoader
data_loader = MalNetGraphLoader(config)
train_loader, val_loader, test_loader = data_loader.create_data_loaders()
print(f"✓ Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

# Step 2: Create Global Model
print("\n🔨 Creating global model...")
from core.models import create_model
global_model = create_model(config).to(device)
print(f"✓ Model created: {global_model.count_parameters():,} parameters")

# Step 3: Split Data for Clients
print(f"\n👥 Creating {config['federated']['num_clients']} clients...")
from core.data_splitter import create_federated_datasets
client_datasets = create_federated_datasets(
    train_loader.dataset,
    config['federated']['num_clients'],
    config['federated']['split_strategy'],
    config['federated']['alpha']
)
print(f"✓ Data split complete")

# Step 4: Create Clients
from core.federated_learning import FederatedClient
clients = []
for i in range(config['federated']['num_clients']):
    client_loader = PyGDataLoader(
        client_datasets[i],
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    client_model = create_model(config)
    client = FederatedClient(
        client_id=i,
        local_data=client_loader,
        model=client_model,
        config=config,
        device=device
    )
    clients.append(client)
print(f"✓ {len(clients)} clients created")

# Step 5: Training Loop
print(f"\n🎯 Starting training: {config['federated']['num_rounds']} rounds")
print("=" * 60)

num_rounds = config['federated']['num_rounds']
local_epochs = config['federated']['local_epochs']
participation_rate = config['federated']['participation_rate']

start_time = time.time()

for round_num in range(1, num_rounds + 1):
    round_start = time.time()
    print(f"\n📍 Round {round_num}/{num_rounds}")
    
    # Select participating clients
    import numpy as np
    num_participants = max(1, int(participation_rate * len(clients)))
    participating_clients = np.random.choice(len(clients), size=num_participants, replace=False)
    
    # Get global weights
    global_weights = global_model.get_weights()
    
    # Train clients
    client_updates = []
    for client_id in participating_clients:
        client = clients[client_id]
        client_results = client.train_local_model(global_weights, local_epochs)
        client_updates.append(client_results)
    
    # Aggregate updates (FedAvg)
    aggregated_weights = {}
    for key in global_weights.keys():
        aggregated_weights[key] = sum(update['weights'][key] * update['num_samples'] 
                                     for update in client_updates) / sum(update['num_samples'] for update in client_updates)
    
    # Update global model
    global_model.set_weights(aggregated_weights)
    
    # Evaluate
    global_model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = global_model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, batch.y)
            val_loss += loss.item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    round_time = time.time() - round_start
    print(f"   Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f} | Time: {round_time:.1f}s")

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("🎉 Training completed successfully!")
print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print("=" * 60)

# Save model
torch.save(global_model.state_dict(), 'results/final_model.pth')
print(f"💾 Model saved to results/final_model.pth")

