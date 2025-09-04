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

### 2.2 *Distill* the knowledge from a large model into a smaller one

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

**Objective**: Learn to work with the Hugging Face ecosystem to adapt models to new tasks.

### 1.1 Load and explore the *Rotten Tomatoes* dataset

The dataset contains 10,662 movie reviews, labeled as positive or negative and is structured with two columns: 
- **text** - review content 
- **label** - 0 = negative, 1 = positive

Furthermore is splitted into:
- **train** - 8530 reviews
- **validation** - 1066 reviews
- **test** - 1066 reviews

<details>
<summary>Positive and negative review examples</summary>

```python
{'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 'label': 1}
{'text': 'simplistic , silly and tedious .', 'label': 0}
```

</details>

### 1.2  Load the Distilbert model and corresponding tokenizer.
Load the distilbert-base-uncased model from Hugging Face and its corresponding tokenizer. \
Next use the tokenizer to preprocess a few examples from the dataset.

<details>
<summary>Code</summary>

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('distilbert/distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

# Take the first 3 samples
samples = train_data['text'][:3]

tokens = tokenizer(samples, padding=True, truncation=True, return_tensors='pt')
outputs = model(**tokens)
print('Model outputs:', outputs)
```

</details>

<details>
<summary>Output</summary>

```python
Model outputs: BaseModelOutput(last_hidden_state=tensor([[[-0.0332, -0.0168,  0.0194,  ...,  0.0476,  0.5834,  0.3036],
         [-0.0235, -0.0555, -0.3638,  ...,  0.1877,  0.5781, -0.1577],
         [-0.0516, -0.1014, -0.1511,  ...,  0.1503,  0.2649, -0.1575],
         ...,
         [ 0.3688, -0.1147,  0.8428,  ..., -0.0708, -0.0178, -0.2516],
         [ 0.0654, -0.0206,  0.1889,  ...,  0.1159,  0.2323, -0.2404],
         [ 0.0373, -0.0104,  0.1203,  ...,  0.1049,  0.2852, -0.3035]],

        [[-0.2062, -0.0490, -0.4036,  ..., -0.1186,  0.6141,  0.3919],
         [-0.4361, -0.1647, -0.3533,  ...,  0.1086,  0.9478, -0.0272],
         [-0.1164,  0.1690,  0.2698,  ..., -0.1971,  0.4372,  0.2527],
         ...,
         [-0.2341,  0.4810, -0.2634,  ..., -0.3397,  0.2567,  0.1274],
         [ 0.7139,  0.0574, -0.3260,  ...,  0.2041, -0.3800, -0.3343],
         [ 0.5649,  0.2806, -0.0295,  ...,  0.1297, -0.3160, -0.1874]],

        [[-0.2705, -0.1265, -0.0500,  ..., -0.3721,  0.2477,  0.3306],
         [ 0.0502,  0.0702, -0.0243,  ..., -0.5188,  0.5020,  0.0597],
         [-0.2193, -0.2208,  0.3721,  ..., -0.3424, -0.3176,  0.8824],
         ...,
         [ 0.2169, -0.3040, -0.2062,  ..., -0.2185, -0.3271, -0.2299],
         [ 0.1381, -0.2591, -0.2001,  ..., -0.2420, -0.2505, -0.1271],
         [ 0.0629, -0.2533, -0.1480,  ..., -0.2985, -0.1985,  0.0025]]],
       grad_fn=<NativeLayerNormBackward0>), hidden_states=None, attentions=None)
```

</details>

### 1.3 Extract feature and train a classifier
Use the Distilbert model to extract features from the reviews and train a simple SVM classifier. \
This will be the baseline for future experiments.

Results on test set:
Validation Accuracy: 82.0%

Validation Classification Report:

               precision    recall  f1-score

           0       0.80      0.85      0.82
           1       0.84      0.79      0.81

### 2 Fine-tune Distilbert
### 2.1 Tokenize the dataset splits
Use the Distilbert tokenizer to preprocess the entire dataset splits (train, validation, test). \
New dataset structure:

```python
Dataset({
    features: ['text', 'label', 'input_ids', 'attention_mask'],
    num_rows: 8530
})
```

### 2.2 - 2.3 Load and fine-tune Distilbert
Try to fine-tune the entire Distilbert model on the Rotten Tomatoes dataset. \
Training setup:
- **Model**: full trainable *distilbert-base-uncased* with a classification head
- **Epochs**: 3
- **Dataset**: first 1000 samples of the training set
Reached accuracy: **83.0%**

Finetuning the entire Distilbert model is very expensive and prevents using the entire dataset due to limited computational resources. \
Training just 3 epochs requires about 20 minutes on *NVIDIA A100 GPU*.

### 3.1 LoRA - Low-Rank Adaptation
Implement Low-Rank Adaptation (LoRA) to fine-tune only a subset of the model's parameters. 
This solution open the way to use the entire dataset and more epochs. \
\
Results:

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
