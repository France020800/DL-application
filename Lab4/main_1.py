import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import PrecisionRecallDisplay, auc, precision_recall_curve, RocCurveDisplay, roc_curve
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FakeData
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
from Lab4.utils import utils

hyper_params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'batch_size': 128,
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_seed(42)

    train_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_id_loader = DataLoader(train_id_dataset, batch_size=hyper_params['batch_size'], shuffle=True, num_workers=2)
    test_id_loader = DataLoader(test_id_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    ood_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    ood_indices = [i for i, target in enumerate(ood_dataset.targets) if target < 20]
    test_ood_dataset = Subset(ood_dataset, ood_indices)
    fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hyper_params['learning_rate'], momentum=hyper_params['momentum'])

    if model_load_path:
        print('Load existing model')
        model.load_state_dict(torch.load(model_load_path))
    else:
        print("\nStarting training on CIFAR-10...")
        utils.train(model, train_id_loader, optimizer, criterion, device=device, num_epochs=10)
        print("Finished Training.")
    accuracy_report = utils.evaluate_model(model, test_id_loader, device=device)
    print(f'Model accuracy on CIFAR10: {accuracy_report[0]}')
    model_save_path = 'pretrained_models/trained_model.pth'
    torch.save(model.state_dict(), model_save_path)

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

    '''print("\nCalculating OOD scores...")
    id_scores = utils.get_msp_scores(model, test_id_loader, device=device)
    ood_scores = utils.get_msp_scores(model, test_ood_loader, device=device)

    print(f"Calculated {len(id_scores)} ID scores and {len(ood_scores)} OOD scores.")

    print("Plotting histograms of OOD scores...")
    plt.hist(id_scores.cpu(), density=True, alpha=0.5, bins=25, label='test')
    plt.hist(ood_scores.cpu(), density=True, alpha=0.5, bins=25, label='fake')
    plt.legend()
    plt.show()
    plt.savefig('plots/ood_scores_plot_CIFAR100_dataset.png', dpi=300, bbox_inches='tight')

    print("Plotting in-distribution ROC curve...")
    y_in = torch.ones(len(id_scores))
    y_out = torch.zeros(len(ood_scores))
    y = torch.cat((y_in, y_out)).numpy()
    y_pred = torch.cat((id_scores, ood_scores)).cpu().numpy()

    metrics.RocCurveDisplay.from_predictions(y, y_pred)
    plt.show()
    plt.savefig('plots/ROC_curve.png', dpi=300, bbox_inches='tight')'''

