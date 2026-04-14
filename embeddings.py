import re
from collections import Counter
import torch
import torch.nn as nn

text = """
the movie was great and the acting was amazing
the film was terrible and boring
i love this movie
i hate this movie
"""

text = text.lower()
text = re.sub(r"[^\w\s]", "", text)

tokens = text.split()

vocab = Counter(tokens)
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

vocab_size = len(word2idx)
print(vocab_size)

window_size = 2  
data = []
for i in range(window_size, len(tokens) - window_size):
    context = []
    
    for j in range(-window_size, window_size + 1):
        if j != 0:
            context.append(word2idx[tokens[i + j]])
    
    target = word2idx[tokens[i]]
    
    data.append((context, target))
  

contexts = torch.tensor([x[0] for x in data])
targets = torch.tensor([x[1] for x in data])

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
    
model = CBOW(vocab_size, embed_dim=50)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    
    output = model(contexts)
    loss = loss_fn(output, targets)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")