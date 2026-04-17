import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x)
        avg = embeds.mean(dim=1)
        out = self.linear(avg)
        return out