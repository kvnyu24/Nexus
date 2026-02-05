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

## Contributing

When adding new GNN models:
1. Follow message passing framework conventions
2. Include complexity analysis
3. Provide batched graph processing support
4. Document aggregation and update functions
5. Add benchmark results on standard datasets
6. Include ablation studies

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

### Surveys
- Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges (Bronstein et al., 2021)
- Graph Neural Networks: A Review of Methods and Applications (Zhou et al., 2020)
- A Comprehensive Survey on Graph Neural Networks (Wu et al., 2021)

### Libraries
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric
- DGL: https://www.dgl.ai/
- GraphGym: https://github.com/snap-stanford/GraphGym
