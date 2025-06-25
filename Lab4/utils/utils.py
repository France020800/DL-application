import random

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from torch import nn
from tqdm import tqdm

from NormalizeInverse import NormalizeInverse


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


def generate_adversarial_image(model, x, y, dataset, target_label, eps=1/255, verbose=False):
    loss = nn.CrossEntropyLoss()
    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    targeted_attack = True

    x.requires_grad = True

    if verbose: print(x.shape)

    model.eval()
    output = model(x)
    img = inv(x[0])
    if verbose:
        plt.imshow(img.permute(1, 2, 0).detach().cpu())
        plt.title(dataset.classes[output.argmax()])
        plt.show()

    if output.argmax().item() != y.item() or y.item() == target_label:
        if verbose: print('classifier is already wrong or target label same as GT!')
        return img, output
    else:
        done = False
        if verbose: print(f'Attack class: {dataset.classes[output.argmax()]}\nTarget class: {dataset.classes[target_label]}')
        n = 0

        if targeted_attack:
            target = torch.tensor(target_label).unsqueeze(0).cuda()
            if verbose: print(f'target: {dataset.classes[target.squeeze()]}')

        while not done:  # untargeted attack until success!

            x.retain_grad()

            output = model(x)

            model.zero_grad()
            if targeted_attack:
                yt = target  # targeted
            else:
                yt = y  # untargeted

            l = loss(output, yt)
            l.backward()

            if targeted_attack:
                x = x - eps * torch.sign(x.grad)  # fgsm targeted
            else:
                x = x + eps * torch.sign(x.grad)  # fgsm untargeted

            n += 1

            if verbose: print(output.argmax().item(), y.item())
            if not targeted_attack and output.argmax().item() != y.item():
                if verbose: print(f'Untargeted attack success! budget:{int(255 * n * eps)}/255')
                done = True

            if targeted_attack and output.argmax().item() == target:
                if verbose: print(f'Targeted attack({dataset.classes[output.argmax()]}) success! budget:{int(255 * n * eps)}/255')
                done = True

        return img, output


def load_pretrained_model(device='cpu'):
    model_load_path = 'pretrained_models/trained_model.pth'
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load(model_load_path))
    model = model.to(device)
    return model