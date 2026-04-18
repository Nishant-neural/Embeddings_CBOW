import torch
import torch.nn as nn
from preprocess import preprocess_text, tokenize
from classifier import SentimentModel


checkpoint = torch.load('cbow_embeddings.pth')
embeddings = checkpoint['embeddings']
word2idx = checkpoint['word2idx']
idx2word = checkpoint['idx2word']

model = SentimentModel(embeddings)

sentences = [
    "the movie was great",
    "i love this movie",
    "the film was terrible",
    "i hate this movie"
]
labels = [1, 1, 0, 0]  

def sentence_to_indices(sentence, word2idx):
    tokens = tokenize(preprocess_text(sentence))
    return [word2idx.get(token, 0) for token in tokens]  

data = []
for sent, label in zip(sentences, labels):
    indices = sentence_to_indices(sent, word2idx)
    data.append((indices, label))

max_len = max(len(d[0]) for d in data)
padded_data = []
for indices, label in data:
    padded = indices + [0] * (max_len - len(indices))
    padded_data.append((padded, label))

inputs = torch.tensor([x[0] for x in padded_data])
targets = torch.tensor([x[1] for x in padded_data], dtype=torch.float32)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(inputs)
    loss = loss_fn(output.squeeze(), targets)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


torch.save(model.state_dict(), 'sentiment_classifier.pth')