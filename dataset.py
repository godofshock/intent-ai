def create_dataset(samples, word2idx, seq_len=5):
    X, y = [], []

    for code, label in samples:
        tokens = code.split()

        for i in range(len(tokens) - seq_len):
            seq = tokens[i:i+seq_len]
            target = label

            try:
                X.append([word2idx[w] for w in seq])
                y.append(target)
            except:
                continue

    return X, y
