import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x