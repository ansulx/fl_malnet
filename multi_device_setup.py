#!/usr/bin/env python3
"""
Multi-Device Federated Learning Setup
====================================
Setup script for running federated learning across multiple real devices.
"""

import torch
import socket
import threading
import json
import time
import sys
import os
import yaml
from typing import Dict, List, Any
import argparse

# Add project root to path
sys.path.append('.')

class FederatedServerNode:
    """Federated learning server node"""
    
    def __init__(self, config_path: str = "config/research_config.yaml", port: int = 8888):
        """Initialize server node"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.port = port
        self.clients = {}
        self.global_model = None
        self.server_socket = None
        
        print(f"🖥️  FEDERATED SERVER NODE")
        print(f"📡 Port: {port}")
        print(f"🌐 IP: {socket.gethostbyname(socket.gethostname())}")
    
    def start_server(self):
        """Start the federated learning server"""
        from core.federated_learning import FederatedServer
        from core.models import create_model
        
        # Create global model
        self.global_model = create_model(self.config)
        self.server = FederatedServer(self.global_model, self.config)
        
        # Start server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(10)
        
        print(f"✅ Server started on port {self.port}")
        print(f"✅ Global model: {self.global_model.count_parameters():,} parameters")
        print(f"✅ Waiting for client connections...")
        
        # Accept client connections
        while True:
            try:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.start()
            except KeyboardInterrupt:
                print("\n🛑 Server shutting down...")
                break
    
    def handle_client(self, client_socket, address):
        """Handle client connection"""
        client_id = len(self.clients)
        self.clients[client_id] = {
            'socket': client_socket,
            'address': address,
            'connected': True
        }
        
        print(f"📱 Client {client_id} connected from {address}")
        
        try:
            while True:
                # Receive message from client
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                message = json.loads(data)
                response = self.process_client_message(client_id, message)
                
                # Send response to client
                client_socket.send(json.dumps(response).encode('utf-8'))
                
        except Exception as e:
            print(f"❌ Error handling client {client_id}: {e}")
        finally:
            client_socket.close()
            self.clients[client_id]['connected'] = False
            print(f"📱 Client {client_id} disconnected")
    
    def process_client_message(self, client_id: int, message: Dict) -> Dict:
        """Process message from client"""
        msg_type = message.get('type')
        
        if msg_type == 'register':
            return {'status': 'registered', 'client_id': client_id}
        
        elif msg_type == 'get_global_model':
            global_weights = self.server.get_global_weights()
            # Convert tensors to lists for JSON serialization
            serializable_weights = {}
            for key, value in global_weights.items():
                serializable_weights[key] = value.detach().cpu().numpy().tolist()
            return {'type': 'global_model', 'weights': serializable_weights}
        
        elif msg_type == 'send_update':
            # Convert lists back to tensors
            weights = {}
            for key, value in message['weights'].items():
                weights[key] = torch.tensor(value)
            
            client_update = {
                'weights': weights,
                'sample_count': message['sample_count']
            }
            
            # Aggregate update (simplified for demo)
            aggregated = self.server.aggregate_updates([client_update])
            
            return {'status': 'update_received', 'round_complete': True}
        
        else:
            return {'status': 'unknown_message'}

class FederatedClientNode:
    """Federated learning client node"""
    
    def __init__(self, server_ip: str, server_port: int, client_id: int = None):
        """Initialize client node"""
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_id = client_id
        self.socket = None
        self.model = None
        
        print(f"📱 FEDERATED CLIENT NODE")
        print(f"📡 Server: {server_ip}:{server_port}")
        print(f"🆔 Client ID: {client_id if client_id else 'Auto'}")
    
    def connect_to_server(self):
        """Connect to federated learning server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))
        
        # Register with server
        register_msg = {'type': 'register'}
        self.socket.send(json.dumps(register_msg).encode('utf-8'))
        
        response = json.loads(self.socket.recv(4096).decode('utf-8'))
        if response['status'] == 'registered':
            self.client_id = response['client_id']
            print(f"✅ Connected to server as Client {self.client_id}")
            return True
        return False
    
    def run_federated_learning(self, config_path: str = "config/research_config.yaml"):
        """Run federated learning as client"""
        from core.models import create_model
        from core.data_loader import MalNetGraphLoader
        from torch_geometric.loader import DataLoader as PyGDataLoader
        import torch.nn as nn
        import torch.optim as optim
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create local model
        self.model = create_model(config)
        
        # Load local data
        data_loader = MalNetGraphLoader(config)
        train_loader, _, _ = data_loader.create_data_loaders()
        
        # Use subset of data for this client
        from torch.utils.data import Subset
        client_data_size = 20  # Small dataset for demo
        client_dataset = Subset(train_loader.dataset, range(min(client_data_size, len(train_loader.dataset))))
        client_loader = PyGDataLoader(client_dataset, batch_size=2, shuffle=True, num_workers=0)
        
        print(f"✅ Local model: {self.model.count_parameters():,} parameters")
        print(f"✅ Local data: {len(client_dataset)} samples")
        
        # Federated learning loop
        round_num = 0
        while True:
            round_num += 1
            print(f"\n🔄 FEDERATED ROUND {round_num}")
            
            # Get global model from server
            get_model_msg = {'type': 'get_global_model'}
            self.socket.send(json.dumps(get_model_msg).encode('utf-8'))
            
            response = json.loads(self.socket.recv(4096).decode('utf-8'))
            if response['type'] == 'global_model':
                # Update local model with global weights
                global_weights = {}
                for key, value in response['weights'].items():
                    global_weights[key] = torch.tensor(value)
                
                self.model.set_weights(global_weights)
                print(f"✅ Received global model from server")
            
            # Train locally
            print(f"🏋️  Training locally on {len(client_dataset)} samples...")
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(2):  # 2 local epochs
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_data in client_loader:
                    if isinstance(batch_data, list):
                        batch, labels = batch_data
                    else:
                        batch = batch_data
                        labels = batch.y
                    
                    optimizer.zero_grad()
                    output = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                accuracy = 100.0 * correct / total
                print(f"   Epoch {epoch+1}/2: Loss: {total_loss/len(client_loader):.4f}, Accuracy: {accuracy:.1f}%")
            
            # Send update to server
            local_weights = self.model.get_weights()
            serializable_weights = {}
            for key, value in local_weights.items():
                serializable_weights[key] = value.detach().cpu().numpy().tolist()
            
            update_msg = {
                'type': 'send_update',
                'weights': serializable_weights,
                'sample_count': len(client_dataset)
            }
            
            self.socket.send(json.dumps(update_msg).encode('utf-8'))
            response = json.loads(self.socket.recv(4096).decode('utf-8'))
            
            if response['status'] == 'update_received':
                print(f"✅ Sent update to server")
            
            time.sleep(2)  # Wait before next round

def main():
    """Main function for multi-device setup"""
    parser = argparse.ArgumentParser(description='Multi-Device Federated Learning')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                       help='Run as server or client')
    parser.add_argument('--port', type=int, default=8888,
                       help='Server port (default: 8888)')
    parser.add_argument('--server-ip', type=str, default='localhost',
                       help='Server IP address (for client mode)')
    parser.add_argument('--client-id', type=int,
                       help='Client ID (optional, auto-assigned if not provided)')
    parser.add_argument('--config', type=str, default='config/research_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        print("🚀 STARTING FEDERATED LEARNING SERVER")
        server = FederatedServerNode(args.config, args.port)
        server.start_server()
    
    elif args.mode == 'client':
        print("🚀 STARTING FEDERATED LEARNING CLIENT")
        client = FederatedClientNode(args.server_ip, args.port, args.client_id)
        if client.connect_to_server():
            client.run_federated_learning(args.config)

if __name__ == "__main__":
    main()
