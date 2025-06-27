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
    'batch_size': 128,
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    utils.set_seed(42)

    data_dir = '/data01/dl24framar/deep_learning_application/Lab4/adv_cifar10'
    labels_path = os.path.join(data_dir, 'labels.csv')
    adv_dataset = AdversarialDataset(data_dir, labels_path, transform=transform)
    adv_loader = DataLoader(adv_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    test_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_id_loader = DataLoader(test_id_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    ood_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    ood_indices = [i for i, target in enumerate(ood_dataset.targets) if target < 20]
    test_ood_dataset = Subset(ood_dataset, ood_indices)
    test_ood_loader = DataLoader(test_ood_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    # model = utils.load_pretrained_model('/data01/dl24framar/deep_learning_application/Lab4/pretrained_models/trained_model.pth', device)
    model = utils.load_pretrained_model('pretrained_models/adv_exposure_model.pth', device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hyper_params['learning_rate'], momentum=hyper_params['momentum'])

    '''print('Start adversarial sample exposure...')
    utils.train(model, adv_loader, optimizer, criterion, device=device, num_epochs=hyper_params['epochs'])'''
    accuracy_report = utils.evaluate_model(model, test_id_loader, device=device)
    print(f'Model accuracy on CIFAR10: {accuracy_report[0]}')

    #model_save_path = '/data01/dl24framar/deep_learning_application/Lab4/pretrained_models/adv_exposure_model.pth'
    #torch.save(model.state_dict(), model_save_path)

    id_val_scores = utils.get_ood_scores(test_id_loader, model, device)
    ood_scores = utils.get_ood_scores(test_ood_loader, model, device)

    plt.figure(figsize=(10, 6))
    plt.hist(id_val_scores, bins=50, alpha=0.7, label='ID (Validation)', color='blue', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.7, label='OOD', color='red', density=True)
    plt.title('Distribution of OOD Scores (1 - Max Softmax Probability)')
    plt.xlabel('OOD Score')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    all_scores = np.concatenate((id_val_scores, ood_scores))
    all_labels = np.concatenate((np.zeros(len(id_val_scores)), np.ones(len(ood_scores))))

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='OOD Detector')
    roc_display.plot()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    print(f"ROC AUC: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(all_labels, all_scores, pos_label=1)
    pr_auc_ood = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    pr_display_ood = PrecisionRecallDisplay(precision=precision, recall=recall,
                                            estimator_name='OOD Detector (OOD as Positive)')
    pr_display_ood.plot()
    plt.title('Precision-Recall Curve (OOD as Positive)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    print(f"Precision-Recall AUC (OOD as Positive): {pr_auc_ood:.4f}")

    id_ness_scores = 1 - all_scores
    precision_id, recall_id, _ = precision_recall_curve(all_labels, id_ness_scores, pos_label=0)  # pos_label=0 for ID
    pr_auc_id = auc(recall_id, precision_id)

    plt.figure(figsize=(8, 6))
    pr_display_id = PrecisionRecallDisplay(precision=precision_id, recall=recall_id,
                                           estimator_name='OOD Detector (ID as Positive)')
    pr_display_id.plot()
    plt.title('Precision-Recall Curve (ID as Positive)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    print(f"Precision-Recall AUC (ID as Positive): {pr_auc_id:.4f}")

