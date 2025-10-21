#!/usr/bin/env python3
"""
Clean Federated Learning System - Real-Time Dashboard
======================================================
Simplified, visual, research-grade FL for supervisor demonstration

Run: python run_clean_fl.py

ROBUST VERSION: Handles timeouts, errors, and automatic recovery
"""

import os
import sys
import time
import threading
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Set timeouts and error handling
torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Prevent CUDA fragmentation

# ANSI colors for terminal
class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_BLACK = '\033[40m'


# ============================================================================
# 1. SIMPLE GNN MODEL
# ============================================================================

class MalwareGNN(nn.Module):
    """Simple GCN for malware detection"""
    
    def __init__(self, input_dim=3, num_classes=5, hidden_dim=64):
        super(MalwareGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # Adaptive input features
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
    
    def get_weights(self):
        return {name: param.clone().detach() for name, param in self.named_parameters()}
    
    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.copy_(weights[name])


# ============================================================================
# 2. FEDERATED SERVER
# ============================================================================

class FederatedServer:
    """Central FL server with live status tracking"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        self.model = self.model.to(self.device)
        
        # Status tracking
        self.status = "INITIALIZING"
        self.current_round = 0
        self.total_rounds = config['num_rounds']
        self.global_accuracy = 0.0
        self.global_loss = 0.0
        self.accuracy_history = deque(maxlen=50)
        self.round_times = deque(maxlen=10)
        self.total_samples = 0
        
        # Device tracking
        self.connected_devices = []
        self.device_status = {}
        
    def register_device(self, device_id, device_name, num_samples):
        """Register a new device"""
        self.connected_devices.append(device_id)
        self.device_status[device_id] = {
            'name': device_name,
            'status': 'CONNECTED',
            'accuracy': 0.0,
            'loss': 0.0,
            'samples': num_samples,
            'last_update': time.time()
        }
        self.total_samples += num_samples
    
    def aggregate_updates(self, device_updates):
        """FedAvg aggregation"""
        self.status = "AGGREGATING"
        
        total_samples = sum(u['num_samples'] for u in device_updates)
        aggregated_weights = {}
        
        # Weighted average
        for key in self.model.state_dict().keys():
            weighted_sum = torch.zeros_like(self.model.state_dict()[key])
            for update in device_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += weight * update['weights'][key]
            aggregated_weights[key] = weighted_sum
        
        # Update global model
        self.model.load_state_dict(aggregated_weights)
        
    def evaluate(self, test_loader, max_retries=3):
        """Evaluate global model with error handling"""
        self.status = "EVALUATING"
        
        for attempt in range(max_retries):
            try:
                self.model.eval()
                
                correct = 0
                total = 0
                total_loss = 0.0
                batch_count = 0
                
                with torch.no_grad():
                    for batch_data in test_loader:
                        try:
                            # Handle both list and Batch formats
                            if isinstance(batch_data, list):
                                batch, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
                            else:
                                batch = batch_data.to(self.device)
                                labels = batch.y
                            
                            output = self.model(batch.x, batch.edge_index, batch.batch)
                            loss = F.cross_entropy(output, labels)
                            
                            _, predicted = torch.max(output.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            total_loss += loss.item()
                            batch_count += 1
                            
                            # Periodically clear cache
                            if batch_count % 10 == 0:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            else:
                                raise
                
                if total > 0 and batch_count > 0:
                    self.global_accuracy = 100.0 * correct / total
                    self.global_loss = total_loss / batch_count
                    self.accuracy_history.append(self.global_accuracy)
                    
                    # Clear cache after evaluation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    return {'accuracy': self.global_accuracy, 'loss': self.global_loss}
                else:
                    # No valid batches processed
                    return {'accuracy': self.global_accuracy, 'loss': self.global_loss}
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    # Return last known values
                    return {'accuracy': self.global_accuracy, 'loss': self.global_loss}


# ============================================================================
# 3. FEDERATED DEVICE (CLIENT)
# ============================================================================

class FederatedDevice:
    """FL device/client that trains locally"""
    
    def __init__(self, device_id, device_name, local_data, model, config):
        self.device_id = device_id
        self.device_name = device_name
        self.local_data = local_data
        self.model = model
        self.config = config
        self.device = config['device']
        
        self.status = "IDLE"
        self.local_accuracy = 0.0
        self.local_loss = 0.0
        self.num_samples = len(local_data.dataset)
    
    def train_local(self, global_weights, num_epochs, max_retries=3):
        """Train on local data with error handling and retry logic"""
        self.status = "TRAINING"
        
        for attempt in range(max_retries):
            try:
                # Load global weights
                self.model.load_state_dict(global_weights)
                self.model.train()
                
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    correct = 0
                    total = 0
                    batch_count = 0
                    
                    for batch_data in self.local_data:
                        try:
                            # Handle both list and Batch formats
                            if isinstance(batch_data, list):
                                batch, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
                            else:
                                batch = batch_data.to(self.device)
                                labels = batch.y
                            
                            optimizer.zero_grad()
                            output = self.model(batch.x, batch.edge_index, batch.batch)
                            loss = F.cross_entropy(output, labels)
                            loss.backward()
                            
                            # Gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            
                            epoch_loss += loss.item()
                            _, predicted = torch.max(output.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            batch_count += 1
                            
                            # Periodically clear cache to prevent memory issues
                            if batch_count % 10 == 0:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            else:
                                raise
                    
                    if batch_count > 0:
                        self.local_loss = epoch_loss / batch_count
                        self.local_accuracy = 100.0 * correct / total if total > 0 else 0.0
                
                self.status = "COMPLETED"
                
                # Clear cache before returning
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Return updated weights
                return {
                    'device_id': self.device_id,
                    'weights': self.model.state_dict(),
                    'num_samples': self.num_samples,
                    'accuracy': self.local_accuracy,
                    'loss': self.local_loss
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    # On final failure, return last known good state
                    self.status = "ERROR"
                    return {
                        'device_id': self.device_id,
                        'weights': global_weights,  # Return unchanged weights
                        'num_samples': self.num_samples,
                        'accuracy': 0.0,
                        'loss': 999.0
                    }


# ============================================================================
# 4. TERMINAL DASHBOARD
# ============================================================================

class TerminalDashboard:
    """Real-time terminal dashboard"""
    
    def __init__(self, server, devices):
        self.server = server
        self.devices = devices
        self.start_time = time.time()
        self.log_messages = deque(maxlen=10)
        
    def clear_screen(self):
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_messages.append(f"[{timestamp}] {message}")
    
    def draw_progress_bar(self, progress, width=40):
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {int(progress * 100)}%"
    
    def draw_accuracy_graph(self, history, width=60, height=7):
        """Draw ASCII accuracy graph"""
        if len(history) < 2:
            return ["No data yet..."]
        
        values = list(history)
        min_val = min(values)
        max_val = max(values)
        
        # Normalize to 0-height range
        if max_val - min_val > 0:
            normalized = [(v - min_val) / (max_val - min_val) * (height - 1) for v in values]
        else:
            normalized = [height // 2] * len(values)
        
        lines = []
        for h in range(height - 1, -1, -1):
            line = f"{int(min_val + (max_val - min_val) * h / (height - 1)):3d}% │ "
            for i, val in enumerate(normalized):
                if abs(val - h) < 0.5:
                    line += "●" if i == len(normalized) - 1 else "─"
                elif val > h:
                    line += "│"
                else:
                    line += " "
            lines.append(line)
        
        lines.append("     └" + "─" * len(normalized))
        return lines
    
    def render(self):
        """Render full dashboard"""
        self.clear_screen()
        
        # Header
        print(f"\n{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}")
        print(f"{Color.CYAN}{Color.BOLD}{'🖥️  FEDERATED LEARNING SERVER - LIVE DASHBOARD':^80}{Color.RESET}")
        print(f"{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}\n")
        
        # Server Status Box
        runtime = time.time() - self.start_time
        status_color = Color.GREEN if self.server.status == "RUNNING" else Color.YELLOW
        
        print(f"{Color.BLUE}┌{'─' * 78}┐{Color.RESET}")
        print(f"{Color.BLUE}│ {Color.BOLD}SERVER STATUS{' ' * 65}│{Color.RESET}")
        print(f"{Color.BLUE}├{'─' * 78}┤{Color.RESET}")
        print(f"{Color.BLUE}│{Color.RESET} Status: {status_color}● {self.server.status:<15}{Color.RESET} "
              f"Round: {Color.BOLD}{self.server.current_round}/{self.server.total_rounds}{Color.RESET}     "
              f"Device: {Color.CYAN}{self.server.device}{Color.RESET}     "
              f"Runtime: {int(runtime)}s {Color.BLUE}│{Color.RESET}")
        print(f"{Color.BLUE}│{Color.RESET} Global Accuracy: {Color.GREEN}{Color.BOLD}{self.server.global_accuracy:.2f}%{Color.RESET}  "
              f"Loss: {self.server.global_loss:.4f}  "
              f"Samples: {self.server.total_samples:,} "
              f"{Color.BLUE}│{Color.RESET}")
        print(f"{Color.BLUE}│{Color.RESET} Connected Devices: {Color.GREEN}{len(self.server.connected_devices)}/{len(self.devices)}{Color.RESET}"
              f"{' ' * 48}{Color.BLUE}│{Color.RESET}")
        print(f"{Color.BLUE}└{'─' * 78}┘{Color.RESET}\n")
        
        # Connected Devices Box
        print(f"{Color.MAGENTA}┌{'─' * 78}┐{Color.RESET}")
        print(f"{Color.MAGENTA}│ {Color.BOLD}CONNECTED DEVICES (Real-Time Status){' ' * 42}│{Color.RESET}")
        print(f"{Color.MAGENTA}├{'─' * 78}┤{Color.RESET}")
        
        for device in self.devices:
            status_symbol = {
                'TRAINING': f'{Color.YELLOW}●',
                'COMPLETED': f'{Color.GREEN}✓',
                'IDLE': f'{Color.BLUE}○',
                'CONNECTED': f'{Color.CYAN}○'
            }.get(device.status, f'{Color.WHITE}○')
            
            print(f"{Color.MAGENTA}│{Color.RESET} Device {device.device_id} [{device.device_name:<12}] "
                  f"{status_symbol} {device.status:<10}{Color.RESET} "
                  f"Acc: {device.local_accuracy:5.1f}%  "
                  f"Samples: {device.num_samples:>5} "
                  f"{Color.MAGENTA}│{Color.RESET}")
        
        print(f"{Color.MAGENTA}└{'─' * 78}┘{Color.RESET}\n")
        
        # Accuracy Graph
        if len(self.server.accuracy_history) > 1:
            print(f"{Color.CYAN}┌{'─' * 78}┐{Color.RESET}")
            print(f"{Color.CYAN}│ {Color.BOLD}GLOBAL MODEL ACCURACY OVER TIME{' ' * 46}│{Color.RESET}")
            print(f"{Color.CYAN}├{'─' * 78}┤{Color.RESET}")
            
            graph_lines = self.draw_accuracy_graph(self.server.accuracy_history)
            for line in graph_lines:
                print(f"{Color.CYAN}│{Color.RESET} {line}{' ' * (76 - len(line))}{Color.CYAN}│{Color.RESET}")
            
            print(f"{Color.CYAN}└{'─' * 78}┘{Color.RESET}\n")
        
        # Recent Activity Log
        print(f"{Color.GREEN}┌{'─' * 78}┐{Color.RESET}")
        print(f"{Color.GREEN}│ {Color.BOLD}RECENT ACTIVITY{' ' * 63}│{Color.RESET}")
        print(f"{Color.GREEN}├{'─' * 78}┤{Color.RESET}")
        
        for msg in list(self.log_messages)[-5:]:
            print(f"{Color.GREEN}│{Color.RESET} {msg:<76} {Color.GREEN}│{Color.RESET}")
        
        print(f"{Color.GREEN}└{'─' * 78}┘{Color.RESET}\n")


# ============================================================================
# 5. MAIN FEDERATED LEARNING ORCHESTRATOR
# ============================================================================

def load_malnet_data():
    """Load and prepare MalNet data with optimized settings"""
    from core.data_loader import MalNetGraphLoader
    
    config = {
        'dataset': {
            'path': 'malnet-graphs-tiny',
            'max_nodes': 2000,
            'batch_size': 8,  # Reduced from 16 to prevent timeouts
            'num_workers': 0,
            'pin_memory': False
        },
        'model': {'num_classes': 5}
    }
    
    data_loader = MalNetGraphLoader(config)
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()
    
    return train_loader, test_loader


def split_data_for_devices(train_loader, num_devices=5):
    """Split training data across devices (Non-IID) with optimized batch size"""
    dataset = train_loader.dataset
    total_samples = len(dataset)
    samples_per_device = total_samples // num_devices
    
    device_datasets = []
    start_idx = 0
    
    for i in range(num_devices):
        end_idx = start_idx + samples_per_device
        if i == num_devices - 1:
            end_idx = total_samples
        
        device_data = [dataset[j] for j in range(start_idx, end_idx)]
        # Reduced batch size from 16 to 8 for stability
        device_loader = PyGDataLoader(device_data, batch_size=8, shuffle=True, num_workers=0)
        device_datasets.append(device_loader)
        
        start_idx = end_idx
    
    return device_datasets


def run_federated_learning():
    """Main FL orchestration with live dashboard"""
    
    print(f"\n{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'🚀 INITIALIZING FEDERATED LEARNING SYSTEM':^80}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'═' * 80}{Color.RESET}\n")
    
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_devices': 5,
        'num_rounds': 20,
        'local_epochs': 3
    }
    
    print(f"⚙️  Device: {Color.GREEN}{config['device'].upper()}{Color.RESET}")
    print(f"📱 Devices: {Color.GREEN}{config['num_devices']}{Color.RESET}")
    print(f"🔄 Rounds: {Color.GREEN}{config['num_rounds']}{Color.RESET}")
    print(f"📊 Loading MalNet dataset...")
    
    # Load data
    train_loader, test_loader = load_malnet_data()
    print(f"   {Color.GREEN}✓{Color.RESET} Loaded {len(train_loader.dataset)} training samples")
    
    # Detect input dimension from first sample
    first_sample = train_loader.dataset[0]
    if isinstance(first_sample, tuple):
        input_dim = first_sample[0].x.shape[1]
    else:
        input_dim = first_sample.x.shape[1]
    print(f"   {Color.GREEN}✓{Color.RESET} Detected input dimension: {input_dim} features")
    
    # Split data for devices
    device_datasets = split_data_for_devices(train_loader, config['num_devices'])
    print(f"   {Color.GREEN}✓{Color.RESET} Split data across {config['num_devices']} devices\n")
    
    # Create global model
    print(f"🤖 Creating global model...")
    global_model = MalwareGNN(input_dim=input_dim, num_classes=5, hidden_dim=64)
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"   {Color.GREEN}✓{Color.RESET} Model created: {num_params:,} parameters\n")
    
    # Create server
    print(f"🖥️  Starting FL server...")
    server = FederatedServer(global_model, config)
    server.status = "RUNNING"
    print(f"   {Color.GREEN}✓{Color.RESET} Server ready\n")
    
    # Create devices
    print(f"📱 Connecting devices...")
    devices = []
    device_names = ['Hospital', 'University', 'Company', 'Lab', 'Institute']
    
    for i, (name, local_data) in enumerate(zip(device_names, device_datasets)):
        device = FederatedDevice(
            device_id=i+1,
            device_name=name,
            local_data=local_data,
            model=MalwareGNN(input_dim=input_dim, num_classes=5, hidden_dim=64).to(config['device']),
            config=config
        )
        devices.append(device)
        server.register_device(i+1, name, len(local_data.dataset))
        print(f"   {Color.GREEN}✓{Color.RESET} Device {i+1} [{name}] connected ({len(local_data.dataset)} samples)")
    
    print(f"\n{Color.GREEN}{Color.BOLD}✅ System ready! Starting federated training...{Color.RESET}\n")
    time.sleep(2)
    
    # Create dashboard
    dashboard = TerminalDashboard(server, devices)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check for existing checkpoint to resume from
    start_round = 1
    checkpoint_file = 'checkpoints/latest_checkpoint.pth'
    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file)
            server.model.load_state_dict(checkpoint['model_state'])
            start_round = checkpoint['round'] + 1
            server.accuracy_history = deque(checkpoint['accuracy_history'], maxlen=50)
            dashboard.log(f"Resumed from checkpoint at round {checkpoint['round']}")
            print(f"\n{Color.YELLOW}📁 Resuming from round {checkpoint['round']}{Color.RESET}\n")
            time.sleep(2)
        except Exception as e:
            print(f"\n{Color.YELLOW}⚠️  Could not load checkpoint: {e}{Color.RESET}\n")
            start_round = 1
    
    # Training loop with robust error handling
    for round_num in range(start_round, config['num_rounds'] + 1):
        try:
            server.current_round = round_num
            round_start = time.time()
            
            dashboard.render()
            dashboard.log(f"Starting Round {round_num}/{config['num_rounds']}")
            time.sleep(0.5)
            
            # Get global weights
            global_weights = server.model.state_dict()
            
            # Train all devices with error recovery
            device_updates = []
            for device in devices:
                try:
                    dashboard.render()
                    dashboard.log(f"Device {device.device_id} [{device.device_name}] training...")
                    
                    update = device.train_local(global_weights, config['local_epochs'])
                    
                    # Only include successful updates
                    if update['loss'] < 900:  # Filter out error returns
                        device_updates.append(update)
                        dashboard.log(f"Device {device.device_id} completed (Acc: {device.local_accuracy:.1f}%, Loss: {device.local_loss:.3f})")
                    else:
                        dashboard.log(f"Device {device.device_id} failed, skipping...")
                        device.status = "ERROR"
                    
                    dashboard.render()
                    time.sleep(0.3)
                    
                except Exception as e:
                    dashboard.log(f"Device {device.device_id} error: {str(e)[:50]}")
                    device.status = "ERROR"
                    continue
            
            # Only aggregate if we have at least one valid update
            if len(device_updates) > 0:
                # Aggregate
                dashboard.log(f"Aggregating updates from {len(device_updates)} devices...")
                dashboard.render()
                server.aggregate_updates(device_updates)
                time.sleep(0.3)
                
                # Evaluate
                dashboard.log("Evaluating global model...")
                dashboard.render()
                results = server.evaluate(test_loader)
                
                round_time = time.time() - round_start
                server.round_times.append(round_time)
                
                dashboard.log(f"Round {round_num} complete: Accuracy={results['accuracy']:.2f}%, Loss={results['loss']:.4f}")
                
                # Save checkpoint every 5 rounds or on last round
                if round_num % 5 == 0 or round_num == config['num_rounds']:
                    try:
                        torch.save({
                            'round': round_num,
                            'model_state': server.model.state_dict(),
                            'accuracy_history': list(server.accuracy_history),
                            'global_accuracy': server.global_accuracy
                        }, checkpoint_file)
                        dashboard.log(f"Checkpoint saved at round {round_num}")
                    except Exception as e:
                        dashboard.log(f"Checkpoint save failed: {str(e)[:50]}")
            else:
                dashboard.log(f"No valid updates in round {round_num}, skipping aggregation")
            
            # Final render
            dashboard.render()
            time.sleep(1)
            
            # Clear GPU cache between rounds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except KeyboardInterrupt:
            dashboard.log("Training interrupted by user")
            raise
        except Exception as e:
            dashboard.log(f"Round {round_num} error: {str(e)[:50]}")
            # Try to continue to next round
            time.sleep(2)
            continue
    
    # Final summary
    dashboard.render()
    print(f"\n{Color.GREEN}{Color.BOLD}{'═' * 80}{Color.RESET}")
    print(f"{Color.GREEN}{Color.BOLD}{'✅ FEDERATED LEARNING COMPLETE':^80}{Color.RESET}")
    print(f"{Color.GREEN}{Color.BOLD}{'═' * 80}{Color.RESET}\n")
    
    print(f"🎯 Final Global Accuracy: {Color.GREEN}{Color.BOLD}{server.global_accuracy:.2f}%{Color.RESET}")
    print(f"📈 Improvement: +{server.global_accuracy - list(server.accuracy_history)[0]:.2f}%")
    print(f"⏱️  Average Round Time: {np.mean(server.round_times):.1f}s")
    print(f"💾 Total Samples Trained: {server.total_samples:,}")
    print(f"\n{Color.CYAN}Dashboard closed. Training complete!{Color.RESET}\n")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_federated_learning()
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}⚠️  Training interrupted by user{Color.RESET}\n")
    except Exception as e:
        print(f"\n\n{Color.RED}❌ Error: {e}{Color.RESET}\n")
        import traceback
        traceback.print_exc()

