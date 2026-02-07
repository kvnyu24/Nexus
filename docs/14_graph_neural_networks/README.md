# Graph Neural Networks

This directory contains comprehensive documentation for state-of-the-art graph neural network architectures for learning on graph-structured data.

## Overview

Graph Neural Networks (GNNs) are deep learning models designed to learn representations of graph-structured data. They have become fundamental tools for tasks involving relational data, from social network analysis to molecular property prediction and recommendation systems.

## Models Covered

### Core GNN Architectures
1. **[Base GNN](base_gnn.md)** - Foundational message passing GNN with multi-head attention
2. **[Message Passing](message_passing.md)** - Adaptive message passing with attention mechanisms
3. **[GraphSAGE](graph_sage.md)** - Inductive learning via sampling and aggregation
4. **[GATv2](gatv2.md)** - Graph attention networks with dynamic attention

### Graph Transformers
5. **[GPS](gps.md)** - General, Powerful, Scalable graph transformer
6. **[Exphormer](exphormer.md)** - Sparse graph transformer with expander graphs

## Key Concepts

### Message Passing Framework

The core paradigm of GNNs follows a message passing scheme:

1. **Message Computation**: Compute messages between connected nodes
   ```
   m_{ij} = φ(h_i, h_j, e_{ij})
   ```

2. **Aggregation**: Combine messages from neighbors
   ```
   m_i = ⊕_{j∈N(i)} m_{ij}
   ```

3. **Update**: Update node representations
   ```
   h_i^{(l+1)} = ψ(h_i^{(l)}, m_i)
   ```

### Architecture Patterns

**Local Message Passing**:
- GCN, GraphSAGE, GIN
- Aggregates information from immediate neighbors
- Limited to k-hop neighborhoods after k layers
- Efficient but may miss long-range dependencies

**Global Attention**:
- Graph Transformers, GPS, Exphormer
- Captures long-range dependencies
- More expressive but computationally expensive
- Addresses over-smoothing and bottleneck issues

**Hybrid Approaches**:
- GPS combines local MPNN + global attention
- Exphormer uses sparse attention via expander graphs
- Best of both worlds: expressiveness + efficiency

## Theoretical Foundations

### Expressiveness

**Weisfeiler-Lehman (WL) Test**: Measures GNN expressive power
- GCN/GraphSAGE: Limited to 1-WL test
- GIN: Matches 1-WL test (maximally expressive among MPNNs)
- Graph Transformers: Can exceed 1-WL test
- Higher-order GNNs: Approach k-WL tests

### Key Challenges

1. **Over-smoothing**: Node representations become indistinguishable
   - Caused by repeated averaging in message passing
   - Mitigated by: Skip connections, normalization, attention

2. **Over-squashing**: Information bottleneck in deep GNNs
   - Long-range information compressed through narrow paths
   - Addressed by: Virtual nodes, graph rewiring, attention

3. **Heterophily**: When connected nodes have different labels
   - Traditional GNNs assume homophily (similar nodes connect)
   - Solutions: Higher-order methods, attention mechanisms

4. **Scalability**: Expensive for large graphs
   - Full-batch training infeasible for million-node graphs
   - Solutions: Sampling (GraphSAGE), clustering, sparse attention

## Applications

### Node-Level Tasks
- **Node Classification**: Predict node labels
- **Node Regression**: Predict continuous node properties
- **Link Prediction**: Predict missing/future edges

### Graph-Level Tasks
- **Graph Classification**: Classify entire graphs
- **Graph Regression**: Predict graph properties
- **Graph Generation**: Generate new graphs

### Domain-Specific
- **Molecular Property Prediction**: Drug discovery (GPS, Exphormer)
- **Social Networks**: Community detection, influence analysis
- **Recommendation Systems**: User-item graphs
- **Knowledge Graphs**: Reasoning and completion
- **Traffic Prediction**: Road networks
- **Protein Structure**: Function prediction

## Model Comparison

| Model | Type | Complexity | Best For | Limitations |
|-------|------|------------|----------|-------------|
| Base GNN | MPNN | O(E·d²) | Small graphs | Limited expressiveness |
| Message Passing | MPNN | O(E·d²) | General purpose | k-hop limitation |
| GraphSAGE | Sampling | O(|S|·d²) | Large graphs | Approximate |
| GATv2 | Attention | O(E·d²) | Dynamic graphs | Quadratic in degree |
| GPS | Hybrid | O(N²·d + E·d²) | Diverse tasks | Memory intensive |
| Exphormer | Sparse Transformer | O(N·d²) | Large graphs | Approximation |

Legend:
- N: Number of nodes
- E: Number of edges
- d: Hidden dimension
- |S|: Sample size

## Implementation Reference

All models implemented in `/Users/kevinyu/Projects/Nexus/nexus/models/gnn/`:
- Modular design with shared components
- Support for batched graph processing
- Integration with PyTorch Geometric conventions
- Memory-efficient implementations

### Shared Components

**Attention Mechanisms**:
```python
from nexus.models.gnn.attention import GraphAttention
```

**Message Passing**:
```python
from nexus.models.gnn.message_passing import AdaptiveMessagePassingLayer
```

**Aggregation**:
- Mean, Sum, Max aggregation
- Attention-weighted aggregation
- Set-based aggregation

## Getting Started

Each model documentation includes:
1. **Overview & Motivation** - Problem setting and innovations
2. **Theoretical Background** - Mathematical foundations
3. **Mathematical Formulation** - Rigorous definitions
4. **Architecture Diagrams** - Visual representations
5. **Implementation Details** - Code walkthrough
6. **Optimization Tricks** - Training and inference tips
7. **Experiments & Results** - Benchmark performance
8. **Common Pitfalls** - What to avoid
9. **References** - Papers and resources

## Recommended Reading Order

### For Beginners
1. **Base GNN** - Understand core message passing
2. **GraphSAGE** - Learn about sampling and inductive learning
3. **GATv2** - Explore attention mechanisms

### For Practitioners
1. **Message Passing** - Advanced MPNN techniques
2. **GPS** - Hybrid local-global architectures
3. **Exphormer** - Efficient scaling to large graphs

### For Researchers
1. **GPS** - State-of-the-art modular framework
2. **Exphormer** - Novel sparse attention mechanisms
3. **GATv2** - Theoretical improvements over GAT

## Recent Trends (2023-2025)

1. **Graph Transformers**: Global attention mechanisms (GPS, Exphormer)
2. **Sparse Attention**: Efficient alternatives to full attention
3. **Positional Encodings**: LapPE, RWSE for structure-aware learning
4. **Subgraph Methods**: Higher expressiveness via subgraph sampling
5. **Equivariant GNNs**: Geometry-aware networks (E(3)-equivariance)
6. **Graph Foundation Models**: Pre-trained models for graphs

## Benchmark Datasets

### Small-Scale
- **Cora, CiteSeer, PubMed**: Citation networks (node classification)
- **MUTAG, PROTEINS**: Molecular graphs (graph classification)
- **Zachary's Karate Club**: Classic toy dataset

### Medium-Scale
- **ogbn-arxiv**: Citation network (169K nodes)
- **ogbg-molhiv**: Molecular property prediction
- **Reddit**: Social network (232K nodes)

### Large-Scale
- **ogbn-papers100M**: Citation network (111M nodes)
- **ogbn-products**: Amazon product network (2.4M nodes)
- **MAG240M**: Microsoft Academic Graph (244M nodes)

## Training Infrastructure

GNN models leverage specialized components:
- **ConfigValidatorMixin**: Configuration validation
- **FeatureBankMixin**: Feature caching and replay
- **HierarchicalVisualizer**: Graph structure visualization
- **Batch Processing**: Efficient batched graph operations

## Performance Optimization

### Memory Optimization
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision training
with torch.cuda.amp.autocast():
    output = model(x, edge_index)
```

### Scalability
```python
# Mini-batch training with sampling
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[15, 10, 5],  # 3-layer sampling
    batch_size=1024
)
```

### Inference Acceleration
```python
# Layer-wise inference for large graphs
@torch.no_grad()
def inference_layer_wise(model, x, edge_index):
    for layer in model.layers:
        x = layer(x, edge_index)
        # Save intermediate results
    return x
```

## Future Directions

1. **Scalability**: Billion-node graphs with distributed training
2. **Expressiveness**: Beyond 1-WL test (higher-order methods)
3. **Generalization**: Transfer learning and meta-learning for graphs
4. **Geometric Deep Learning**: Incorporating symmetries and invariances
5. **Dynamic Graphs**: Temporal graph neural networks
6. **Heterogeneous Graphs**: Different node/edge types
7. **Self-Supervised Learning**: Pre-training on graph structures

## Theoretical Guarantees

### Universal Approximation
- MPNNs are universal approximators for graph functions (within WL hierarchy)
- Graph Transformers can represent any permutation-equivariant function

### Stability
- Lipschitz bounds on message passing operations
- Spectral analysis of graph convolutions

### Generalization
- PAC bounds for graph learning
- Sample complexity analysis

## Architecture Deep Dive

### Message Passing Framework

All GNN architectures implement variants of the message passing scheme:

**Phase 1: Message Computation**
```python
# Compute message from neighbor j to node i
m_{ij} = MSG(h_i, h_j, e_{ij})

# Examples:
# GCN: m_{ij} = W h_j / sqrt(d_i * d_j)
# GAT: m_{ij} = α_{ij} W h_j
# GIN: m_{ij} = h_j
```

**Phase 2: Aggregation**
```python
# Combine messages from all neighbors
m_i = AGG_{j∈N(i)} m_{ij}

# Aggregation types:
# Sum: m_i = Σ_{j∈N(i)} m_{ij}
# Mean: m_i = 1/|N(i)| Σ_{j∈N(i)} m_{ij}
# Max: m_i = max_{j∈N(i)} m_{ij}
# Attention: m_i = Σ_{j∈N(i)} α_{ij} m_{ij}
```

**Phase 3: Update**
```python
# Update node representation
h_i^{(l+1)} = UPDATE(h_i^{(l)}, m_i)

# Common updates:
# GCN: h_i' = σ(m_i)
# GIN: h_i' = MLP((1+ε) h_i + m_i)
# GAT: h_i' = σ(m_i)
```

### Attention Mechanisms in GNNs

**Static Attention (GAT)**:
- Attention coefficients independent of query node
- Computed as: e_{ij} = a^T [W h_i || W h_j]
- Normalized: α_{ij} = softmax_j(e_{ij})

**Dynamic Attention (GATv2)**:
- Attention depends on both nodes jointly
- Computed as: e_{ij} = a^T LeakyReLU(W[h_i || h_j])
- More expressive ranking

**Multi-Scale Attention (GPS)**:
- Combines local (edge-based) and global (full graph) attention
- Hierarchical information aggregation
- Best of local + global

### Positional and Structural Encodings

**Why Needed?**
Graphs lack natural ordering, need to encode structure.

**Laplacian Positional Encoding (LapPE)**:
```python
L = D - A  # Graph Laplacian
λ, V = eig(L)  # Eigendecomposition
PE = Linear(V[:, 1:k])  # Use k smallest eigenvectors
```

Properties:
- Captures global graph geometry
- Related to graph diffusion
- Permutation invariant (up to sign)

**Random Walk Structural Encoding (RWSE)**:
```python
P = D^{-1} A  # Transition matrix
p_i(t) = [P^t]_{ii}  # t-step return probability
SE = MLP([p_i(1), p_i(2), ..., p_i(T)])
```

Properties:
- Captures local neighborhood structure
- Efficient to compute
- Node centrality information

**Centrality Encodings**:
- Degree centrality: |N(i)|
- Betweenness centrality: shortest paths through node
- PageRank: stationary distribution of random walks
- Eigenvector centrality: principal eigenvector of A

### Graph Pooling Strategies

**Node-Level Pooling** (for graph classification):

1. **Global Mean Pooling**:
```python
h_G = 1/N Σ_i h_i
```

2. **Global Sum Pooling**:
```python
h_G = Σ_i h_i
```

3. **Global Max Pooling**:
```python
h_G = max_i h_i
```

4. **Attention Pooling**:
```python
α_i = softmax(MLP(h_i))
h_G = Σ_i α_i h_i
```

**Hierarchical Pooling** (coarsening):
- DiffPool: Learnable soft clustering
- TopKPool: Select top-k nodes by score
- SAGPool: Self-attention based pooling

### Training Strategies

**Mini-Batch Training**:
```python
from torch_geometric.loader import NeighborLoader

# Sample k-hop neighborhoods
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 2-layer sampling
    batch_size=1024,
    shuffle=True
)

for batch in loader:
    output = model(batch.x, batch.edge_index)
    loss = criterion(output[batch.train_mask], batch.y[batch.train_mask])
```

**Full-Batch Training**:
```python
# For small graphs that fit in memory
output = model(data.x, data.edge_index)
loss = criterion(output[data.train_mask], data.y[data.train_mask])
```

**Graph-Level Batching**:
```python
from torch_geometric.loader import DataLoader

# Batch multiple graphs
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch.x: Concatenated node features
    # batch.edge_index: Concatenated edges with offset
    # batch.batch: Graph assignment per node
    output = model(batch.x, batch.edge_index, batch.batch)
```

### Addressing Key Challenges

**1. Over-Smoothing**

Problem: Node features become indistinguishable with depth.

Solutions:
- **Skip Connections**: h^{(l)} = h^{(l-1)} + GNN^{(l)}(h^{(l-1)})
- **Initial Residual**: h^{(l)} = α h^{(0)} + (1-α) GNN^{(l)}(h^{(l-1)})
- **DropEdge**: Randomly drop edges during training
- **PairNorm**: Normalize to maintain feature variance

**2. Over-Squashing**

Problem: Information bottleneck in message passing.

Solutions:
- **Graph Rewiring**: Add edges to improve connectivity
- **Virtual Nodes**: Global node connected to all
- **Global Attention**: Direct long-range connections (GPS)
- **Expander Graphs**: Sparse graphs with good expansion

**3. Heterophily**

Problem: Connected nodes have different labels.

Solutions:
- **Higher-Order GNNs**: Consider k-hop neighbors
- **Adaptive Aggregation**: Learn to weight neighbors
- **Attention Mechanisms**: Select relevant neighbors
- **Graph Transformers**: Non-local aggregation

**4. Scalability**

Problem: Large graphs don't fit in memory.

Solutions:
- **Sampling**: NeighborLoader, ClusterGCN
- **Quantization**: Reduce precision (FP16, INT8)
- **Distributed Training**: Split across GPUs
- **Sparse Operations**: Leverage graph sparsity

## Advanced Topics

### Equivariant GNNs

For geometric graphs (molecules, point clouds):

**E(3)-Equivariance**:
- Invariant to rotations and translations
- Examples: SchNet, DimeNet, EGNN
- Important for 3D molecular modeling

**SE(3)-Equivariance**:
- Also handles reflections
- Examples: TFN, Cormorant
- Used in protein structure prediction

### Temporal Graph Networks

For dynamic graphs:

**Discrete-Time Models**:
```python
# Graph evolves in discrete steps
G_t = {V_t, E_t}
h_t = GNN(h_{t-1}, G_t)
```

**Continuous-Time Models**:
```python
# Events occur at arbitrary times
h(t) = GNN(h(t-), events_before_t)
```

Applications:
- Social network dynamics
- Traffic prediction
- Recommender systems

### Heterogeneous Graph Networks

For graphs with multiple node/edge types:

```python
# Different message functions per edge type
for edge_type in edge_types:
    m_{edge_type} = MSG_{edge_type}(h_src, h_dst)

# Aggregate across edge types
m_i = Σ_{edge_type} AGG(m_{edge_type})
```

Examples: HAN, RGCN, HGT

### Self-Supervised Learning

**Contrastive Learning**:
- Graph-level: Augment and contrast
- Node-level: Context prediction
- Edge-level: Link prediction

**Generative Pre-training**:
- Masked node/edge prediction
- Graph reconstruction
- Attribute prediction

**Multi-Task Pre-training**:
- Combine multiple objectives
- Transfer to downstream tasks

## Practical Implementation Guide

### Data Preprocessing

**Graph Construction**:
```python
import torch
from torch_geometric.data import Data

# Create graph
x = torch.randn(100, 64)  # Node features
edge_index = torch.randint(0, 100, (2, 500))  # Edges
edge_attr = torch.randn(500, 8)  # Edge features
y = torch.randint(0, 10, (100,))  # Labels

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

**Feature Engineering**:
```python
# Node features
- One-hot encoding of categorical features
- Normalization of continuous features
- Centrality measures
- Structural features

# Edge features
- Edge weights
- Edge types
- Distance information
- Temporal information
```

**Data Splitting**:
```python
# Node classification
num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)
train_mask = perm[:int(0.7 * num_nodes)]
val_mask = perm[int(0.7 * num_nodes):int(0.85 * num_nodes)]
test_mask = perm[int(0.85 * num_nodes):]

# Graph classification
from torch_geometric.data import random_split
train_data, val_data, test_data = random_split(
    dataset, [0.7, 0.15, 0.15]
)
```

### Model Selection Guide

**Choose GCN/GraphSAGE when**:
- Large graphs (millions of nodes)
- Limited compute resources
- Homophilic graphs
- Baseline comparison needed

**Choose GAT/GATv2 when**:
- Node importance varies
- Dynamic graphs
- Need interpretability
- Sufficient compute available

**Choose GIN when**:
- Maximum discriminative power needed
- Graph classification
- WL-test equivalence important

**Choose GPS when**:
- Diverse tasks
- Long-range dependencies needed
- Moderate graph sizes (<100K nodes)
- State-of-the-art performance critical

**Choose Exphormer when**:
- Very large graphs (>100K nodes)
- Need global attention
- Memory constraints
- Scalability is priority

### Hyperparameter Tuning

**Critical Hyperparameters**:
```python
# Architecture
hidden_dim: 64-512  # Start with 128-256
num_layers: 2-10  # Start with 3-4
dropout: 0.0-0.5  # Start with 0.1

# Attention (GAT, GPS)
num_heads: 4-16  # Start with 8
attention_dropout: 0.0-0.3

# Training
learning_rate: 1e-5 to 1e-2  # Start with 1e-4
weight_decay: 0.0-1e-3  # Start with 5e-4
batch_size: 32-512  # Depends on graph size

# Positional encodings (GPS)
num_eigenvectors: 4-16  # Start with 8
walk_length: 8-32  # Start with 16
```

**Tuning Strategy**:
1. Start with reasonable defaults
2. Grid search on learning rate and weight decay
3. Tune architecture (layers, hidden dim)
4. Fine-tune attention and dropout
5. Optimize batch size and sampling

## Contributing

When adding new GNN models:
1. Follow message passing framework conventions
2. Include complexity analysis
3. Provide batched graph processing support
4. Document aggregation and update functions
5. Add benchmark results on standard datasets
6. Include ablation studies
7. Test on at least 3 benchmark datasets
8. Compare to baseline models
9. Provide hyperparameter recommendations
10. Document memory and time complexity

## References

### Foundational Papers
- **GCN**: Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, ICLR 2017)
- **GraphSAGE**: Inductive Representation Learning on Large Graphs (Hamilton et al., NeurIPS 2017)
- **GAT**: Graph Attention Networks (Veličković et al., ICLR 2018)
- **GIN**: How Powerful are Graph Neural Networks? (Xu et al., ICLR 2019)

### Recent Advances
- **GPS**: Recipe for a General, Powerful, Scalable Graph Transformer (Rampášek et al., NeurIPS 2022)
- **GATv2**: How Attentive are Graph Attention Networks? (Brody et al., ICLR 2022)
- **Exphormer**: Sparse Transformers for Graphs (Shirzad et al., ICML 2023)
- **GraphGPS**: General, Powerful, Scalable Graph Transformers (Rampášek et al., 2022)

### Theory & Expressiveness
- **Message Passing**: Neural Message Passing for Quantum Chemistry (Gilmer et al., ICML 2017)
- **WL Test**: The power of graph neural networks (Xu et al., ICLR 2019)
- **Over-smoothing**: Deeper Insights into Graph Convolutional Networks (Li et al., AAAI 2018)
- **Over-squashing**: Understanding over-squashing in GNNs (Alon & Yahav, ICML 2021)

### Graph Transformers
- **Graphormer**: Do Transformers Really Perform Bad for Graph Representation? (Ying et al., NeurIPS 2021)
- **SAN**: Rethinking Graph Transformers with Spectral Attention (Kreuzer et al., NeurIPS 2021)
- **Graph ViT**: Vision GNN: An Image is Worth Graph of Nodes (Han et al., NeurIPS 2022)

### Applications
- **Molecular**: Molecular property prediction with graph neural networks (Gilmer et al., 2017)
- **Social**: Graph Convolutional Neural Networks for Web-Scale Recommender Systems (Ying et al., 2018)
- **Biological**: Learning protein structure with GNNs (Ingraham et al., NeurIPS 2019)
- **Vision**: Learning to Simulate Complex Physics with Graph Networks (Sanchez-Gonzalez et al., ICML 2020)

### Surveys
- Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges (Bronstein et al., 2021)
- Graph Neural Networks: A Review of Methods and Applications (Zhou et al., 2020)
- A Comprehensive Survey on Graph Neural Networks (Wu et al., 2021)
- Graph Neural Networks: Foundation, Frontiers and Applications (Zhou et al., KDD 2022 Tutorial)

### Benchmarks
- **OGB**: Open Graph Benchmark (Hu et al., NeurIPS 2020)
- **GraphGym**: Design Space for Graph Neural Networks (You et al., NeurIPS 2020)
- **LRGB**: Long Range Graph Benchmark (Dwivedi et al., NeurIPS 2022)
- **TUDataset**: Benchmark Data Sets for Graph Kernels (Kersting et al., 2016)

### Libraries & Tools
- **PyTorch Geometric**: https://github.com/pyg-team/pytorch_geometric
- **DGL**: https://www.dgl.ai/
- **GraphGym**: https://github.com/snap-stanford/GraphGym
- **OGB**: https://ogb.stanford.edu/
- **NetworkX**: https://networkx.org/

### Code Resources
- **GraphGPS**: https://github.com/rampasek/GraphGPS
- **Benchmarking GNNs**: https://github.com/graphdeeplearning/benchmarking-gnns
- **PyG Examples**: https://github.com/pyg-team/pytorch_geometric/tree/master/examples
