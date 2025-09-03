import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import PrecisionRecallDisplay, auc, precision_recall_curve, RocCurveDisplay, roc_curve
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FakeData
import matplotlib.pyplot as plt
from Lab4.utils import utils

hyper_params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'batch_size': 128,
    'num_epochs': 20,
    'input_size': 32*32*3,
    'epochs': 20
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_seed(42)

    ## Exercise 1.1: Implement an OOD detection pipeline

    train_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_id_loader = DataLoader(train_id_dataset, batch_size=hyper_params['batch_size'], shuffle=True, num_workers=2)
    test_id_loader = DataLoader(test_id_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    ood_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    ood_indices = [i for i, target in enumerate(ood_dataset.targets) if target < 20]
    test_ood_dataset = Subset(ood_dataset, ood_indices)
    # fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    test_ood_loader = DataLoader(test_ood_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    print(f"ID training samples: {len(train_id_dataset)}")
    print(f"ID test samples: {len(test_id_dataset)}")
    print(f"OOD test samples: {len(test_ood_dataset)}")
    model_load_path = 'pretrained_models/trained_model.pth'
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    # model = CNN()
    model = model.to(device)

    if os.path.exists(model_load_path):
        print('Load existing model')
        model.load_state_dict(torch.load(model_load_path))
    else:
        print("\nStarting training on CIFAR-10...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=hyper_params['learning_rate'], momentum=hyper_params['momentum'])
        utils.train(model, train_id_loader, optimizer, criterion, device=device, num_epochs=hyper_params['epochs'])
        print("Finished Training.")

    accuracy_report = utils.evaluate_model(model, test_id_loader, device=device)
    print(f'Model accuracy on CIFAR10: {accuracy_report[0]}')
    model_save_path = 'pretrained_models/trained_model.pth'
    torch.save(model.state_dict(), model_save_path)

    ## Get scores and plot
    scores_test = utils.compute_scores(model, device, test_id_loader, utils.max_logit)
    scores_fake = utils.compute_scores(model, device, test_ood_loader, utils.max_logit)

    plt.plot(sorted(scores_test.cpu()), label='test')
    plt.plot(sorted(scores_fake.cpu()), label='fake')
    plt.legend()
    plt.show()

    plt.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label='test')
    plt.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label='fake')
    plt.legend()
    plt.show()

    ## Exercise 1.2: Evaluate the OOD detection performance
    ypred = torch.cat((scores_test, scores_fake))
    y_test = torch.ones_like(scores_test)
    y_fake = torch.zeros_like(scores_fake)

    y = torch.cat((y_test, y_fake))

    RocCurveDisplay.from_predictions(y.cpu(), ypred.cpu())

    precision, recall, _ = precision_recall_curve(y.cpu().numpy(), ypred.cpu().numpy())
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_display.plot()
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


if __name__ == '__main__':
    main()
