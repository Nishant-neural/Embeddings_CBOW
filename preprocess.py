import re
from collections import Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def build_vocab(tokens):
    vocab = Counter(tokens)
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    return vocab, word2idx, idx2word