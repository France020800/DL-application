import random

import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from Lab4.utils.NormalizeInverse import NormalizeInverse
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    utils.set_seed(42)
    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    test_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    class_dict = {class_name: id_class for id_class, class_name in enumerate(test_id_dataset.classes)}

    model = utils.load_pretrained_model('pretrained_models/trained_model.pth', device)
    model.train()

    idx = random.randint(0, len(test_id_dataset) - 1)
    image, label = test_id_dataset[idx]
    x = image.unsqueeze(0).to(device)
    y = torch.tensor([label]).to(device)

    # Select the adv_class
    all_classes = list(class_dict.values())
    all_classes.remove(label)
    adv_class = random.choice(all_classes)

    adv_img, output = utils.generate_adversarial_image(
        model, x, y, test_id_dataset, adv_class, eps=hyper_params['eps'], verbose=True
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

if __name__ == '__main__':
    main()