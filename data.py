import torch
from preprocess import preprocess_text, tokenize, build_vocab

def create_cbow_data(tokens, word2idx, window_size=2):
    data = []
    for i in range(window_size, len(tokens) - window_size):
        context = []
        for j in range(-window_size, window_size + 1):
            if j != 0:
                context.append(word2idx[tokens[i + j]])
        target = word2idx[tokens[i]]
        data.append((context, target))
    return data

def prepare_data(text, window_size=2):
    text = preprocess_text(text)
    tokens = tokenize(text)
    vocab, word2idx, idx2word = build_vocab(tokens)
    data = create_cbow_data(tokens, word2idx, window_size)
    contexts = torch.tensor([x[0] for x in data])
    targets = torch.tensor([x[1] for x in data])
    return contexts, targets, vocab, word2idx, idx2word, len(word2idx)