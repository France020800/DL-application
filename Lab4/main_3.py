import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import FakeData

from utils import  utils
from torchvision.transforms import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = utils.load_pretrained_model('pretrained_models/trained_model.pth', device)

    batch_size = 128
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    # fake_loader = DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    ood_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    ood_indices = [i for i, target in enumerate(ood_dataset.targets) if target < 20]
    fakeset = Subset(ood_dataset, ood_indices)
    fake_loader = DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=2)

    combined_loader = DataLoader(ConcatDataset([testset, fakeset]), batch_size=batch_size, shuffle=False, num_workers=8,
                                 persistent_workers=True)

    # Perform ODIN detection
    best_T, best_epsilon = None, None
    best_auroc = 0

    T_values = [1.0, 10.0, 100.0]
    epsilon_values = [0.001, 0.01, 0.1]

    for T in T_values:
        for epsilon in epsilon_values:
            ood_scores = utils.odin_detection(model, combined_loader, device, T=T, epsilon=epsilon)
            labels = [0] * len(test_loader.dataset) + [1] * len(fake_loader.dataset)
            auroc, aupr = utils.evaluate_ood_scores(ood_scores, labels)
            print(f"T={T}, epsilon={epsilon}, AUROC={auroc:.4f}, AUPR={aupr:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_T, best_epsilon = T, epsilon

    print(f"Best T={best_T}, Best epsilon={best_epsilon}, Best AUROC={best_auroc:.4f}")