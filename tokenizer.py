import re

def tokenize_code(code):
    tokens = re.findall(r"[A-Za-z_]+|\d+|==|!=|<=|>=|[^\s]", code)
    return tokens

def build_vocab(tokens):
    vocab = sorted(list(set(tokens)))
    word2idx = {w:i for i,w in enumerate(vocab)}
    idx2word = {i:w for w,i in word2idx.items()}
    return word2idx, idx2word
