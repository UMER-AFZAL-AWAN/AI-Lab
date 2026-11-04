# =============================================================================
# 03 — GRAPH NEURAL NETWORKS (GNN)
# =============================================================================
#
# LIKE YOU ARE 5:
# Normal NN: processes a table of numbers (each row is one item, independent).
# CNN: processes grids (pixels are neighbours by physical proximity).
# Transformer: processes sequences (tokens are ordered).
# GNN: processes GRAPHS — data with arbitrary connections.
#
# A GRAPH has NODES (things) and EDGES (connections between things).
# Examples:
# - Social network: people (nodes) + friendships (edges)
# - Molecule: atoms (nodes) + chemical bonds (edges)
# - Road network: intersections (nodes) + roads (edges)
#
# HOW GNN WORKS (Message Passing):
# Each node has a feature vector.
# In each layer:
#   1. AGGREGATE: collect features from all neighbouring nodes.
#   2. UPDATE: combine own features with the aggregated neighbourhood features.
# After L layers, each node's representation captures its L-hop neighbourhood.
#
# RELEVANCE TO AI VIDEO:
# Not directly used in AnimateDiff or RIFE.
# But scene graphs (describing relationships between objects in a video)
# are represented as graphs, and GNNs are used in advanced video understanding.
# Also: 3D neural rendering (NeRF follow-ons) sometimes uses GNN on point clouds.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("GRAPH REPRESENTATION")
print("=" * 60)
#
# A graph is represented by:
# - node_features: (N, F) — N nodes, each with F features
# - edge_index:    (2, E) — E edges. edge_index[0] = source, edge_index[1] = target

num_nodes  = 5    # 5 nodes in the graph
num_edges  = 8    # 8 directed edges
node_feat_dim = 4  # each node has 4 features

# Node features (e.g. atom type, charge, etc.)
node_features = torch.randn(num_nodes, node_feat_dim)

# Edge index: 8 edges connecting various nodes
# Example: a small social network
#   0 -> 1, 1 -> 0, 1 -> 2, 2 -> 1, 2 -> 3, 3 -> 2, 3 -> 4, 4 -> 3
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4],   # source nodes
    [1, 0, 2, 1, 3, 2, 4, 3],   # target nodes
], dtype=torch.long)

print(f"Node features: {node_features.shape}  ({num_nodes} nodes × {node_feat_dim} features)")
print(f"Edge index:    {edge_index.shape}  (2 × {num_edges} edges)")
print(f"Source nodes:  {edge_index[0].tolist()}")
print(f"Target nodes:  {edge_index[1].tolist()}")

print()
print("=" * 60)
print("MANUAL MESSAGE PASSING LAYER")
print("=" * 60)
#
# Step 1: For each edge, compute a message from the source node.
# Step 2: Aggregate all messages at each target node (sum/mean).
# Step 3: Update each node's features using its aggregated messages.

class SimpleGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Transform node features before aggregation
        self.W_self  = nn.Linear(in_features,  out_features)
        self.W_neigh = nn.Linear(in_features,  out_features)

    def forward(self, x, edge_index):
        """
        x:          (N, in_features)  — node features
        edge_index: (2, E)            — edges
        """
        src, dst = edge_index[0], edge_index[1]

        # Step 1: Messages from source nodes
        messages = x[src]   # (E, in_features) — one message per edge

        # Step 2: Aggregate at target nodes (sum aggregation)
        agg = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        agg.index_add_(0, dst, messages)   # sum messages for each target node
        # Degree normalisation (divide by number of neighbours)
        degree = torch.bincount(dst, minlength=x.shape[0]).float().clamp(min=1)
        agg    = agg / degree.unsqueeze(1)  # mean aggregation

        # Step 3: Update — combine own features + aggregated neighbourhood
        out = F.relu(self.W_self(x) + self.W_neigh(agg))
        return out

gnn_layer = SimpleGNNLayer(in_features=4, out_features=8)
out_features = gnn_layer(node_features, edge_index)
print(f"Input node features:  {node_features.shape}")
print(f"Output node features: {out_features.shape}  (each node now encodes its neighbourhood)")

print()
print("=" * 60)
print("GRAPH CLASSIFICATION (predict a property of the whole graph)")
print("=" * 60)
#
# After L message-passing layers, all node features are updated.
# To predict a graph-level property (e.g. is this molecule toxic?),
# aggregate all node features into one graph-level vector (global pooling).

class SmallGNN(nn.Module):
    def __init__(self, in_feat=4, hidden=16, num_classes=3):
        super().__init__()
        self.layer1 = SimpleGNNLayer(in_feat, hidden)
        self.layer2 = SimpleGNNLayer(hidden, hidden)
        self.classify= nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        # Global mean pool: average all node features -> one graph vector
        graph_repr = x.mean(dim=0, keepdim=True)  # (1, hidden)
        return self.classify(graph_repr)

gnn = SmallGNN()
logits = gnn(node_features, edge_index)
print(f"Graph-level prediction: {logits.shape}  (3 class scores for the whole graph)")

print()
print("=" * 60)
print("PYG vs DGL (libraries for GNN)")
print("=" * 60)
print("PyTorch Geometric (PyG): most popular GNN library for PyTorch.")
print("DGL (Deep Graph Library): also popular, slightly different API.")
print()
print("Real usage:")
print("  from torch_geometric.nn import GCNConv, GATConv, SAGEConv")
print("  from torch_geometric.data import Data")
print()
print("  graph = Data(x=node_features, edge_index=edge_index)")
print("  out = GATConv(in_channels=4, out_channels=8)(graph.x, graph.edge_index)")
print()
print("GAT (Graph Attention Network): uses attention to weight neighbour contributions.")
print("GraphSAGE: samples a fixed number of neighbours (scalable to huge graphs).")
print("GCN (Graph Convolutional Network): the original, simplest GNN (Kipf & Welling 2017).")

# =============================================================================
# EXPECTED OUTPUT:
# Node features: torch.Size([5, 4])
# Output node features: torch.Size([5, 8])
# Graph-level prediction: torch.Size([1, 3])
# =============================================================================
