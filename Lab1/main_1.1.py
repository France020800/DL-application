import os

from models.MLP import MLP
from comet_ml import start
from comet_ml.integration.pytorch import log_model

import utils
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    hyper_params = {
        'epochs': 25,
        'lr': 0.0001,
        'batch_size': 128,
        'input_size': 28 * 28,
        'width': 512,
        'depth': 10
    }

    model = MLP([hyper_params['input_size']] + [hyper_params['width']] * hyper_params['depth'] + [10]).to(device)
    utils.start_train(model, hyper_params, comet_project='MLP MNIST', device=device)