import random
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm
from comet_ml import start
from comet_ml.integration.pytorch import log_model
from sklearn.metrics import accuracy_score, classification_report

def start_train(model, hyper_params, comet_project, dataset='mnist', device='cpu', early=True, patience=10):
    print(model)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {parameters}')
    mobile_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of free parameters: {mobile_parameters}')

    train_loader, validation_loader, test_loader = load_dataset(dataset, batch_size=hyper_params['batch_size'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])
    scheduler = StepLR(optimizer, step_size=hyper_params['step_size'], gamma=hyper_params['gamma'])
    # scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=hyper_params['gamma'])

    best_val_loss = float('inf')
    epochs_no_improve = 0

    my_var = os.getenv("comet_api_key")
    experiment = start(
        api_key=my_var,
        project_name=comet_project,
        workspace="france020800",
    )
    experiment.add_tag("pytorch")
    experiment.log_parameters(hyper_params)
    experiment.log_text(f'{parameters}')
    experiment.set_model_graph(str(model))

    train_losses = []
    train_accs = []
    val_accs = []
    model = model.to(device)
    for epoch in range(1, hyper_params['epochs'] + 1):
        experiment.log_current_epoch(epoch)
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, current_epoch=epoch)
        val_loss, val_acc = validate(model, device, validation_loader, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | ' +
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        scheduler.step()
        if epoch%hyper_params['step_size'] == 0:
            print(f'Current lr: ${scheduler.get_lr()}')

        experiment.log_metrics({
            'loss': train_loss,
            'val_acc': val_acc,
            'train_acc': train_acc
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and early:
                print(f'Early stopping triggered after {epoch} epochs.')
                break

    log_model(experiment, model, "Residual-CNN")
    accuracy_report = evaluate_model(model, test_loader, device=device)[1]
    experiment.log_text("Accuracy report on TEST:\n" + accuracy_report)
    plot_validation_curves(train_loss, val_accs)
    print(f'Accuracy report on TEST:\n {accuracy_report}')


def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(target).sum().item()
            total += data.size(0)
    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def train(model, device, train_loader, optimizer, criterion, current_epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in tqdm(train_loader, desc=f'Start training epoch {current_epoch}'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)
    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def train_epoch(model, train_loader, device, optimizer, current_epoch, loss_function = F.cross_entropy):
    model.train()
    current_losses = []
    for (x, Y) in tqdm(train_loader, desc=f'Start training epoch {current_epoch}'):
        optimizer.zero_grad()
        x = x.to(device)
        Y = Y.to(device)
        out = model(x)
        current_loss = loss_function(out, Y)
        current_loss.backward()
        optimizer.step()
        current_losses.append(current_loss.item())
    return np.mean(current_losses)


def distillation_loss(student_logits, teacher_logits, targets, T, alpha):
    criterion_CE = nn.CrossEntropyLoss()
    criterion_KL = nn.KLDivLoss(reduction="batchmean")

    log_p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)

    loss_KL = criterion_KL(log_p_student, p_teacher) * (T * T)
    loss_CE = criterion_CE(student_logits, targets)
    return alpha * loss_KL + (1.0 - alpha) * loss_CE


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Sottraggo la media e divido per la deviazione standard
        ])

        ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
        ds_test = MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar-10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Sottraggo la media e divido per la deviazione standard
        ])

        ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
        ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported')

    val_size = 5000
    I = np.random.permutation(len(ds_train))
    ds_val = Subset(ds_train, I[:val_size])
    ds_train = Subset(ds_train, I[val_size:])

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True, num_workers=4)

    return train_loader, validation_loader, test_loader

def plot_validation_curves(losses, accs):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss per Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(accs, color='deepskyblue')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Best Accuracy = {np.max(accs)} @ epoch {np.argmax(accs)}')