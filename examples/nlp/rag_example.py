from nexus.models.nlp import RAGModule, EdgeLLM
from nexus.training import Trainer
from nexus.data import TextProcessor, TextDataset
import torch

# Configuration
config = {
    "hidden_size": 768,
    "num_heads": 8,
    "num_retrieval_docs": 5,
    "vocab_size": 50000,
    "max_seq_length": 512
}

# Initialize models
rag = RAGModule(config)
llm = EdgeLLM(config)

# Create embeddings for documents and query
batch_size = 4
seq_length = 128
doc_length = 1000

# Simulate document and query embeddings
query_embeddings = torch.randn(batch_size, seq_length, config["hidden_size"])
document_embeddings = torch.randn(batch_size, doc_length, config["hidden_size"])

# Process through RAG
outputs = rag(
    query_embeddings=query_embeddings,
    document_embeddings=document_embeddings
)

# Get retrieved documents and their weights
retrieved_docs = outputs["retrieved_docs"]
retrieval_weights = outputs["retrieval_weights"]

# Generate response using LLM with retrieved context
llm_outputs = llm(
    input_ids=torch.randint(0, config["vocab_size"], (batch_size, seq_length)),
    attention_mask=torch.ones(batch_size, seq_length),
    context_embeddings=retrieved_docs
)

print(f"Retrieved document shape: {retrieved_docs.shape}")
print(f"Retrieval weights shape: {retrieval_weights.shape}")
print(f"Final output shape: {llm_outputs['logits'].shape}") 