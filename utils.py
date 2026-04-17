import torch.nn.functional as F

def find_similar(word, word2idx, idx2word, embeddings, top_k=3):
    idx = word2idx[word]
    word_vec = embeddings[idx]
    
    similarities = []
    
    for i in range(len(embeddings)):
        sim = F.cosine_similarity(word_vec, embeddings[i], dim=0)
        similarities.append((idx2word[i], sim.item()))
    
    similarities = sorted(similarities, key=lambda x: -x[1])
    
    return similarities[1:top_k+1]