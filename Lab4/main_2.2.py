import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from utils.AdversarialDataset import AdversarialDataset
from utils import utils

hyper_params = {
    'epochs': 10,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'batch_size': 64,
    'train_ratio': 0.8
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(42)

    # Load previous trained model
    model = utils.load_pretrained_model('pretrained_models/trained_model.pth', device)
    model.to(device)

    # Load ADV dataset
    data_dir = 'adv_dataset'
    labels_path = os.path.join(data_dir, 'labels.csv')
    adv_dataset = AdversarialDataset(data_dir, labels_path, transform=transform)
    dataset_size = len(adv_dataset)
    train_size = int(hyper_params['train_ratio'] * dataset_size)
    test_size = dataset_size - train_size

    adv_train_dataset, adv_test_dataset = torch.utils.data.random_split(adv_dataset, [train_size, test_size])

    adv_train_loader = DataLoader(adv_train_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
    adv_test_dataset = DataLoader(adv_test_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

    # Load OOD dataset
    ood_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    ood_indices = [i for i, target in enumerate(ood_dataset.targets) if target < 20]
    test_ood_dataset = Subset(ood_dataset, ood_indices)
    test_ood_loader = DataLoader(test_ood_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    # Load CIFAR-10 test dataset
    test_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_id_loader = DataLoader(test_id_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    # Adversarial sample exposure
    print("\nStarting adversarial salmple exposure...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hyper_params['learning_rate'], momentum=hyper_params['momentum'])
    utils.train(model, adv_train_loader, optimizer, criterion, device=device, num_epochs=hyper_params['epochs'])
    print("Finish adversarial sample exposure.")

    accuracy_report = utils.evaluate_model(model, test_id_loader, device=device)
    print(f'Model accuracy on CIFAR10: {accuracy_report[0]}')
    model_save_path = 'pretrained_models/ADV_exposure_model.pth'
    torch.save(model.state_dict(), model_save_path)

    # Evaluate model on OOD dataset
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

    ## Evaluate new OOD detection performance
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