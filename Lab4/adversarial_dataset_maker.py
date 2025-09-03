import os
import csv
import random
import torch
import torchvision
from torchvision.utils import save_image
from torchvision.transforms import transforms
from tqdm import tqdm

from Lab4.utils import utils

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def make_adversarial_dataset(dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = './adv_dataset'
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, 'labels.csv')

    model = utils.load_pretrained_model('pretrained_models/trained_model.pth', device)
    class_dict = {class_name: id_class for id_class, class_name in enumerate(dataset.classes)}

    with open(labels_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for i in tqdm(range(0, int(len(dataset)/2)), desc='Generating adversarial images'):
            idx = random.randint(0, len(dataset) - 1)
            image, label = dataset[idx]
            x = image.unsqueeze(0).to(device)
            y = torch.tensor([label]).to(device)

            # Select the adv_class
            all_classes = list(class_dict.values())
            all_classes.remove(label)
            adv_class = random.choice(all_classes)

            adv_img, output = utils.generate_adversarial_image(
                model, x, y, dataset, adv_class, eps=1/255, verbose=False
            )

            filename = f'{i}.png'
            save_image(adv_img, os.path.join(output_dir, filename))
            writer.writerow([filename, label])
            torch.cuda.empty_cache()


if __name__ == '__main__':
    datasets = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    make_adversarial_dataset(datasets)