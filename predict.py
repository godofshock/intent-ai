import torch

def predict(model, seq):
    seq = torch.tensor([seq])
    out = model(seq)
    pred = torch.argmax(out, dim=1).item()
    return pred
