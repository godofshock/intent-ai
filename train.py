import torch
import torch.nn as nn
import torch.optim as optim

def train(model, X, y, epochs=5):
    X = torch.tensor(X)
    y = torch.tensor(y)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item()}")
