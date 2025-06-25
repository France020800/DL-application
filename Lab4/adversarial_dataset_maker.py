import os
import csv
import random
import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from Lab4.utils import utils

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = './adv_cifar10'
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, 'labels.csv')

    model = utils.load_pretrained_model(device)

    batch_size = 128
    train_id_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_id_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    with open(labels_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        idx = 0
        for batch_idx, (images, labels) in enumerate(loader):
            print(f'Processing batch {batch_idx + 1}/{len(loader)}')
            images = images.to(device)
            labels = labels.to(device)

            generated_adv_images_and_labels = []
            for img_idx in tqdm(range(images.size(0)), desc='Generating adversarial images'):
                x = images[img_idx].unsqueeze(0)
                y = labels[img_idx].unsqueeze(0)
                adv_img, _ = utils.generate_adversarial_image(model, x, y, train_id_dataset, random.randint(0, 9),
                                                              eps=1/255)

                generated_adv_images_and_labels.append(
                    (adv_img.detach().cpu(), y.item()))

            for adv_img, original_label in generated_adv_images_and_labels:
                filename = f'{idx}.png'
                save_image(adv_img, os.path.join(output_dir, filename))
                writer.writerow([filename, original_label])
                idx += 1
            torch.cuda.empty_cache()