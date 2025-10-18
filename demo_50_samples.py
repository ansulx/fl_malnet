#!/usr/bin/env python3
"""
Simple Demo: Analyzing 50 Malware Samples
========================================
Shows how nodes, edges, and frequency work with real examples.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

print("🔍 UNDERSTANDING MALWARE GRAPHS - 50 SAMPLE ANALYSIS")
print("=" * 70)

# Dataset path
dataset_path = 'malnet-graphs-tiny'

def analyze_edgelist(file_path):
    """Count nodes and edges in an edgelist file"""
    nodes = set()
    edges = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            source = int(parts[0])
                            target = int(parts[1])
                            nodes.add(source)
                            nodes.add(target)
                            edges.append((source, target))
                        except:
                            continue
    except:
        return 0, 0
    
    return len(nodes), len(edges)

# Analyze samples
results = []
categories = ['benign', 'adware', 'downloader', 'trojan', 'addisplay']

print("\n📊 Analyzing 10 samples from each category (50 total)...\n")

for category in categories:
    print(f"\n🔍 {category.upper()}:")
    print("-" * 70)
    print(f"{'#':<5} {'Nodes':<10} {'Edges':<10} {'File Name':<45}")
    print("-" * 70)
    
    category_path = os.path.join(dataset_path, category)
    subfolder_name = os.listdir(category_path)[0] if os.listdir(category_path) else None
    
    if subfolder_name:
        subfolder = os.path.join(category_path, subfolder_name)
        if os.path.isdir(subfolder):
            files = [f for f in os.listdir(subfolder) if f.endswith('.edgelist')][:10]
            
            for i, filename in enumerate(files, 1):
                file_path = os.path.join(subfolder, filename)
                num_nodes, num_edges = analyze_edgelist(file_path)
                results.append({
                    'category': category,
                    'file': filename,
                    'nodes': num_nodes,
                    'edges': num_edges
                })
                short_name = filename[:42] + '...' if len(filename) > 45 else filename
                print(f"{i:<5} {num_nodes:<10} {num_edges:<10} {short_name:<45}")

# Create DataFrame
df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("📈 SUMMARY STATISTICS")
print("=" * 70)

for category in categories:
    cat_data = df[df['category'] == category]
    print(f"\n{category.upper()}:")
    print(f"  Samples: {len(cat_data)}")
    print(f"  Average Nodes: {cat_data['nodes'].mean():.1f}")
    print(f"  Average Edges: {cat_data['edges'].mean():.1f}")
    print(f"  Range: {cat_data['nodes'].min()}-{cat_data['nodes'].max()} nodes")

print("\n" + "=" * 70)
print("OVERALL (50 samples):")
print(f"  Total: {len(df)} samples")
print(f"  Average Nodes: {df['nodes'].mean():.1f}")
print(f"  Average Edges: {df['edges'].mean():.1f}")
print(f"  Node Range: {df['nodes'].min()} to {df['nodes'].max()}")

# Create visualization
print("\n📊 Creating visualization...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Node distribution
ax1.hist(df['nodes'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(df['nodes'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["nodes"].mean():.0f}')
ax1.set_xlabel('Number of Nodes (Functions)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency (Number of Files)', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Graph Sizes\n(50 samples)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add frequency annotations
heights, bins, patches = ax1.hist(df['nodes'], bins=20, alpha=0)
for i, (height, patch) in enumerate(zip(heights, patches)):
    if height > 0:
        ax1.text(patch.get_x() + patch.get_width()/2, height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

# 2. Edge distribution
ax2.hist(df['edges'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.axvline(df['edges'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["edges"].mean():.0f}')
ax2.set_xlabel('Number of Edges (Function Calls)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency (Number of Files)', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Connectivity\n(50 samples)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Category comparison
category_means = df.groupby('category')['nodes'].mean().sort_values()
colors = ['green', 'orange', 'red', 'purple', 'blue']
bars = ax3.bar(range(len(category_means)), category_means.values, color=colors)
ax3.set_xticks(range(len(category_means)))
ax3.set_xticklabels(category_means.index, rotation=45, ha='right')
ax3.set_ylabel('Average Number of Nodes', fontsize=12, fontweight='bold')
ax3.set_title('Average Graph Size by Category', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

# 4. Nodes vs Edges
colors_map = {'benign': 'green', 'adware': 'orange', 'downloader': 'red', 
              'trojan': 'purple', 'addisplay': 'blue'}
for category in categories:
    cat_data = df[df['category'] == category]
    ax4.scatter(cat_data['nodes'], cat_data['edges'], 
               alpha=0.7, s=100, c=colors_map[category], label=category, edgecolors='black')

ax4.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')
ax4.set_title('Nodes vs Edges by Category', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('50_samples_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as '50_samples_analysis.png'")
plt.show()

# Show example .edgelist content
print("\n" + "=" * 70)
print("📄 EXAMPLE: What's inside an .edgelist file")
print("=" * 70)

example_file = results[0]
example_path = os.path.join(dataset_path, 
                            example_file['category'],
                            os.listdir(os.path.join(dataset_path, example_file['category']))[0],
                            example_file['file'])

print(f"\nFile: {example_file['file']}")
print(f"Category: {example_file['category']}")
print(f"Nodes: {example_file['nodes']}, Edges: {example_file['edges']}")
print("\nFirst 10 lines:")
print("-" * 70)
print("Format: source_node → target_node (means 'Function A calls Function B')\n")

with open(example_path, 'r') as f:
    lines = [line for line in f if line.strip() and not line.startswith('#')][:10]
    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) >= 2:
            print(f"{i:2d}. Node {parts[0]:4s} → Node {parts[1]:4s}  (Function {parts[0]} calls Function {parts[1]})")

print("\n" + "=" * 70)
print("🎯 KEY INSIGHTS FOR SUPERVISOR:")
print("=" * 70)
print("""
1. WHAT IS FREQUENCY?
   - Frequency = How many files have that property
   - In the histogram: Y-axis shows how many files
   - Example: If 5 files have 800 nodes, frequency at 800 = 5

2. WHAT ARE NODES?
   - Nodes = Functions in the malware code
   - Each function = 1 node
   - Example: main(), login(), sendData() = 3 nodes

3. WHAT ARE EDGES?
   - Edges = Function calls (when one function calls another)
   - Example: main() calls login() = 1 edge

4. HOW TO READ THE GRAPH:
   - Tallest bar = Most common size
   - In our 50 samples: Most files have ~800 nodes
   - Different malware types have different patterns

5. REAL EXAMPLE FROM OUR DATA:
""")

sample = df.iloc[0]
print(f"   File: {sample['file'][:50]}")
print(f"   Category: {sample['category']}")
print(f"   Has {sample['nodes']} functions (nodes)")
print(f"   And {sample['edges']} function calls (edges)")
print(f"   This file contributed to the frequency count at ~{sample['nodes']} nodes!")

print("\n✅ Now you can explain: 'Out of 50 files, X files have Y nodes - that's the frequency!'")
print("=" * 70)
