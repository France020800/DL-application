import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from Lab4.NormalizeInverse import NormalizeInverse
from Lab4.utils import utils

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
    utils.set_seed(1234)
    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    test_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    shuffled_indices = torch.randperm(len(test_id_dataset))
    shuffled_dataset = Subset(test_id_dataset, shuffled_indices)
    test_id_loader = DataLoader(shuffled_dataset, batch_size=hyper_params['batch_size'], shuffle=False, num_workers=2)

    class_dict = {class_name: id_class for id_class, class_name in enumerate(test_id_dataset.classes)}

    model = utils.load_pretrained_model(device)
    accuracy_report = utils.evaluate_model(model, test_id_loader, device=device)
    print(f'Model accuracy on CIFAR10: {accuracy_report[0]}')

    model.train()

    idx = 0  # sample index
    image, label = test_id_dataset[idx]
    x = image.unsqueeze(0).to(device)  # add batch dimension and move to device
    y = torch.tensor([label]).to(device)  # make label a tensor and move to device

    adv_img, output = utils.generate_adversarial_image(
        model, x, y, test_id_dataset, class_dict['airplane'], eps=hyper_params['eps']
    )

    plt.imshow(adv_img.permute(1, 2, 0).detach().cpu())
    plt.title(test_id_dataset.classes[output.argmax()])
    plt.show()

    diff = (adv_img - x)
    diffi = inv(diff[0])
    plt.imshow(diffi.permute(1, 2, 0).detach().cpu())
    plt.title('diff')
    plt.show()

    diff_flat = diff.flatten()

    plt.hist(diff_flat.detach().cpu())

