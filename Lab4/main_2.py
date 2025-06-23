import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from Lab4 import utils

hyper_params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'batch_size': 128,
    'eps': 1/255,
    'sample_id': 0
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    utils.set_seed(42)

    test_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    shuffled_indices = torch.randperm(len(test_id_dataset))
    shuffled_dataset = Subset(test_id_dataset, shuffled_indices)
    test_id_loader = DataLoader(shuffled_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    class_dict = {class_name: id_class for id_class, class_name in enumerate(test_id_dataset.classes)}

    model = utils.load_pretrained_model(device)
    accuracy_report = utils.evaluate_model(model, test_id_loader, device=device)
    print(f'Model accuracy on CIFAR10: {accuracy_report[0]}')

    model.train()
    for data in test_id_loader:
        images, labels = data
        break

    targeted_attack = True
    sample_id = hyper_params['sample_id']

    image, label = images[sample_id].to(device), labels[sample_id].to(device)

    x = image[None, :]
    y = label[None]

    utils.generate_adversarial_image(model, x, y, test_id_dataset, class_dict['airplane'], eps=hyper_params['eps'])


