import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from data import prepare_data
from cbow_model import CBOW


train_iter, test_iter = IMDB(split=('train', 'test'))
all_text = ""
for label, text in train_iter:
    all_text += text + " "
for label, text in test_iter:
    all_text += text + " "

contexts, targets, vocab, word2idx, idx2word, vocab_size = prepare_data(all_text)

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

embeddings = model.embedding.weight.data


torch.save({
    'embeddings': embeddings,
    'word2idx': word2idx,
    'idx2word': idx2word,
    'vocab_size': vocab_size
}, 'cbow_embeddings.pth')