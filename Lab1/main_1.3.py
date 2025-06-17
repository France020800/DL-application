from models.CNN import CNN
from models.ResidualCNN import ResidualCNN

import utils
import torch

if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available else 'cpu'
    hyper_params = {
        'epochs': 20,
        'lr': 0.1,
        'batch_size': 128,
        'in_channels': 3,
        'num_classes': 10,
        'conv_channels': [32, 32, 64, 64],
        'weight_decay': 5e-9,
        'step_size': 5,
        'gamma': 0.1,
        'patience': 3,
        'early': False
    }

    utils.set_seed(42)
    #model = CNN(in_channels=hyper_params['in_channels'], num_classes=hyper_params['num_classes'], conv_channels=hyper_params['conv_channels'])
    model = ResidualCNN(in_channels=hyper_params['in_channels'], num_classes=hyper_params['num_classes'], planes=hyper_params['conv_channels'], num_blocks=2)
    utils.start_train(model, hyper_params, dataset='cifar-10', comet_project='CNN vs ResidualCNN', device=device, early=False)

    #torch.save(model.state_dict(), 'pretrained_models/david_big_resnet_cifar10.pth')