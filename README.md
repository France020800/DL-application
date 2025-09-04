# 📚 Deep Learning Application Repository

Deep Learning Application repository! This repository contains the code and resources developed for a university examination in Deep Learning. 

---

## 🔬 Laboratory Sessions Overview

Below is a detailed report of what was done for each laboratory, focusing on the teacher requests.

## **Lab 1: Introduction to Neural Networks and Basic Architectures**
**Objective**: Work with simple DNN architectures like MLPs and CNNs. Training them and see what happen adding more hidden layers or adding residual connections.
### 1.1 MLP
Implement a simple Multilayer Perceptron to classify the 10 digits of MNIST.
<details>
<summary>MLP architecture</summary>

```python
class MLP(nn.Module):
    def __init__(self, layers):
    super().__init__()
    self.layers = nn.ModuleList()
    for in_features, out_features in zip(layers[:-1], layers[1:]):
        self.layers.append(nn.Linear(in_features, out_features))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)  # Last layer
        return x
```
</details>

Results show that adding more hidden layers does not improve model performance significantly. 
The best accuracy is obtained with 2 hidden layers and a width of 512.
- MLP width 512, depth 2: 97.0% accuracy
- MLP width 512, depth 4: 97.4% accuracy
- MLP width 512, depth 10: 97.0% accuracy
- MLP width 512, depth 20: 96.0% accuracy

<p align="middle">
    <img src="Lab1/plots/val_acc_mlp.svg" alt="Validation accuracy MNIST">
</p>

These results suggest that increasing depth beyond a certain point leads to a potential overfitting causing by the vanishing gradient problem.
 
### 1.2 Residual MLP
Implement an MLP with residual connections to mitigate the vanishing gradient problem.

<details>
<summary>Residual MLP architecture</summary>

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, in_features)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x += identity
        return x


class ResidualMLP(nn.Module):
    def __init__(self, in_features, out_features, width, depth):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, width))
        for _ in range(depth-1):
            self.layers.append(ResidualBlock(width))
        self.layers.append(nn.Linear(width, out_features))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)  # Last layer
        return x
```

</details>

Now is possible to train deeper networks without performance degradation.
- Residual MLP width 512, depth 2: 97.6% accuracy
- Residual MLP width 512, depth 4: 98.2% accuracy
- Residual MLP width 512, depth 10: 98.4% accuracy
- Residual MLP width 512, depth 20: 98.4% accuracy

<p align="middle">
    <img src="Lab1/plots/val_acc_residualmlp.svg" alt="Validation accuracy MNIST">
</p>

Detailed results are available on my [comet MLP](https://www.comet.com/france020800/mlp-vs-residualmlp/) project.

### 1.3 CNN and Residual CNN
Implement a simple Convolutional Neural Network (CNN) and a Residual CNN. \
The CIFAR-10 dataset is now used for image classification otherwise the architectural differences would not be visible on mnist.

<details>
<summary>CNN architecture</summary>

```python
class CNN(nn.Module):
    def __init__(self, in_channels, conv_channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch Normalization layers

        prev_channels = in_channels
        for out_channels in conv_channels:
            self.convs.append(nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(out_channels))  # Add BatchNorm2d for each conv layer
            prev_channels = out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(prev_channels, num_classes)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))  # Apply BatchNorm after convolution
        x = self.pool(x)  # shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
```
</details>

Accuracy results:
- CNN channels [32, 32, 64, 64]: 62.0% accuracy
- CNN channels [32, 32, 64, 64, 128, 128]: 68.9% accuracy
- Residual CNN channels [32, 32, 64, 64, 128, 128]: 71.1% accuracy
- Residual CNN channels 2x[32, 32, 64, 64, 128, 128]: 74.7% accuracy

Deeper networks enhance performance, and residual connections further improve them by facilitating gradient flow.

<p align="middle">
    <img src="Lab1/plots/cnn_vs_residualcnn.svg" alt="Validation accuracy CIFAR10">
</p>

Given the same weights, convolutional networks with residual connections are also faster to train:
- 677k parameters CNN: 56.5% accuracy after 20 minutes
- 1.6M parameters Residual CNN: 74.7% accuracy after 5 minutes

<p align="middle">
    <img src="Lab1/plots/Training%20time.svg" alt="20 epochs training time">
</p>

Detailed results are available on my [comet CNN](https://www.comet.com/france020800/cnn-vs-residualcnn) project.

## 2.2 *Distill* the knowledge from a large model into a smaller one

In this section is shown the results of knowledge distillation to transfer knowledge from a large, pre-trained model (teacher) to a smaller model (student). \
The goal is to achieve comparable performance with reduced computational resources.

**Student model**: The best Residual CNN from previous section
- 78.5% accuracy on CIFAR-10 

**Teacher model**: A pre-trained ResNet-18 from torchvision
- 84.0% accuracy on CIFAR-10

**Student model after distillation**:
- 82.4% accuracy on CIFAR-10

<p align="middle">
    <img src="Lab1/plots/val_acc_distillation.svg" alt="Validation accuracy Distillation">
</p>

The student model after distillation achieves performance close to the teacher model.

Detailed results are available on my [comet DISTILLATION](https://www.comet.com/france020800/distilled-cifar10) project.

---

## **Lab 3: Working with Transformers in the HuggingFace Ecosystem**

* **Objective**: This laboratory focuses on Recurrent Neural Networks (RNNs), suitable for processing sequential data such as text or time series. We explore their application in tasks like sentiment analysis or sequence prediction.

---

## **Lab 4: Adversarial Learning and OOD Detection**

* **Objective**: 

---

## 🤝 Contribution

This repository is primarily for a university examination. While direct contributions are not expected for this specific purpose, feedback or suggestions are welcome.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details. (If you have a `LICENSE` file; otherwise, you can remove this section or specify another license).

---

## 📧 Contact

For any questions or inquiries, please open an issue in this repository or contact me directly.

---
