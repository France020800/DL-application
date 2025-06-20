import random

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score
from torch import nn
from tqdm import tqdm

def set_seed(SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    predictions = []
    ground_truths = []
    for (x, Y) in tqdm(test_loader, desc='Evaluating', leave=False):
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1)
        ground_truths.append(Y)
        predictions.append(preds.detach().cpu().numpy())

    return (accuracy_score(np.hstack(ground_truths), np.hstack(predictions)),
            classification_report(np.hstack(ground_truths), np.hstack(predictions), zero_division=0, digits=3))


def train(model, train_loader, optimizer, criterion, num_epochs=10, device='cpu'):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.3f}")


def get_msp_scores(model, loader, device='cpu'):
    scores = []
    with torch.no_grad():
        for data in loader:
            x, y = data
            output = model(x.to(device))
            s = output.max(dim=1)[0]
            scores.append(s)
        scores_t = torch.cat(scores)
        return scores_t


def get_score(model, dataloader, device='cpu'):
    loss = nn.MSELoss(reduction='none')
    model.eval()
    scores = []
    with torch.no_grad():
        for data in dataloader:
            x, y = data
            x = x.to(device)
            xr = model(x)
            l = loss(x, xr)
            score = l.mean([1, 2, 3])
            scores.append(-score)
    return scores