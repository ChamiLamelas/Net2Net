#!/usr/bin/env python3.8

import torch
import torch.nn as nn

class LayerEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(LayerEmbedding, self).__init__()
        self.embedding = nn.EmbeddingBag(100, embedding_dim, sparse=True)

    def forward(self, layer_names):
        # Convert layer names to indices
        indices = [self.layer_to_index(layer_name) for layer_name in layer_names]
        indices = torch.LongTensor(indices)

        # Create embeddings
        embeddings = self.embedding(indices)

        return embeddings

    def layer_to_index(self, layer_name):
        # Implement your logic to convert layer names to indices
        # For simplicity, let's assume each layer type gets a unique index
        layer_types = ["Linear", "Conv", "Pooling", ...]
        layer_type = layer_name.split('-')[0]
        return layer_types.index(layer_type)

# Example usage
layer_embedding_dim = 64
bi_lstm_hidden_size = 32

# Create an instance of LayerEmbedding
layer_embedding = LayerEmbedding(layer_embedding_dim)

# Example layer names
layer_names = ["Linear-256-512", "Conv-32-64-3", "Pooling-2"]

# Obtain embeddings
layer_embeddings = layer_embedding(layer_names)

# Create a bi-LSTM
bi_lstm = nn.LSTM(layer_embedding_dim, bi_lstm_hidden_size, bidirectional=True, batch_first=True)

# Pass layer embeddings through the bi-LSTM
output, _ = bi_lstm(layer_embeddings.unsqueeze(0))

# Now 'output' contains the low-dimensional representation of the input layer names
print(output)