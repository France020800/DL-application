from models.MLP import MLP
import utils
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    hyper_params = {
        'epochs': 10,
        'lr': 0.0001,
        'batch_size': 128,
        'input_size': 28 * 28,
        'output_size': 10,
        'width': 512,
        'depth': 10,
        'weight_decay': 5e-9,
        'step_size': 4,
        'gamma': 0.1,
        'patience': 3,
        'early': False
    }

    utils.set_seed(42)
    model = MLP([hyper_params['input_size']] + [hyper_params['width']] * hyper_params['depth'] + hyper_params['output_size']).to(device)
    utils.start_train(model, hyper_params, comet_project='MLP vs ResidualMLP', device=device)