import utils
import os
from models.ResidualCNN import ResidualCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from comet_ml import start
from tqdm import tqdm

utils.set_seed(42)

hyper_params = {
    'batch_size': 128,
    'num_epochs': 25,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'temperature': 4.0,
    'alpha': 0.7,
    'in_channels': 3,
    'num_classes': 10,
    'out_channels': [32, 32, 64, 64],
}

my_var = os.getenv("comet_api_key")
experiment = start(
        api_key=my_var,
        project_name='Distilled cifar10',
        workspace="france020800",
)
experiment.add_tag("pytorch")
experiment.log_parameters(hyper_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = utils.load_dataset('cifar-10', hyper_params['batch_size'])

teacher_model = models.resnet18(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)
teacher_model.load_state_dict(torch.load('pretrained_models/resnet18_early.pth'))
teacher_model.to(device)
teacher_model.eval()

for param in teacher_model.parameters():
    param.requires_grad = False


student_model = ResidualCNN(in_channels=hyper_params['in_channels'], num_classes=hyper_params['num_classes'], planes=hyper_params['out_channels'], num_blocks=1)
experiment.set_model_graph(str(student_model))
student_model.to(device)


optimizer = optim.SGD(student_model.parameters(),
                      lr=hyper_params['learning_rate'],
                      momentum=hyper_params['momentum'],
                      weight_decay=hyper_params['weight_decay'])

scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[10, 15, 20],
                                           gamma=0.1)


def train_one_epoch(epoch_idx):
    student_model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch_idx+1}/{hyper_params['num_epochs']}]")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        student_outputs = student_model(images)

        loss = utils.distillation_loss(student_outputs,
                                 teacher_outputs,
                                 labels,
                                 hyper_params['temperature'],
                                 hyper_params['alpha'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = student_outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix({
            "Loss": f"{total_loss / total:.4f}",
            "Acc": f"{100.0 * correct/total:.2f}%"
        })


    scheduler.step()
    return total_loss


def evaluate():
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = student_model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc

if __name__ == '__main__':
    best_acc = 0.0
    val_accs = []
    for epoch in range(hyper_params['num_epochs']):
        experiment.log_current_epoch(epoch)
        train_loss = train_one_epoch(epoch)
        val_acc = evaluate()
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}. val_acc: {val_acc:.4f}%, train_loss: {train_loss:.4f}")

        experiment.log_metrics({
            'loss': train_loss,
            'val_acc': val_acc
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student_model.state_dict(), "best_student_cifar10.pth")


    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    accuracy_report = utils.evaluate_model(student_model, test_loader, device=device)[1]
    utils.plot_validation_curves(train_loss, val_accs)
    print(f'Accuracy report on TEST:\n {accuracy_report}')