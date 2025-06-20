import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FakeData
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import utils

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

    print("\nCalculating OOD scores...")
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
    plt.savefig('plots/ROC_curve.png', dpi=300, bbox_inches='tight')

