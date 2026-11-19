import copy
import torch
from torch import nn


def train_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'train': [], 'val': []}
    best_loss = float('inf')
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(x)
            train_n += len(x)
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                val_sum += loss.item() * len(x)
                val_n += len(x)
        train_loss, val_loss = train_sum / train_n, val_sum / val_n
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f'Época {epoch:02d}/{epochs} | treino={train_loss:.5f} | validação={val_loss:.5f}')
    model.load_state_dict(best_state)
    return history
