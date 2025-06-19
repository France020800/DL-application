import random

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate
from transformers import TrainingArguments, Trainer
from comet_ml import Experiment
import os
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

if __name__ == '__main__':
    hyper_parameters = {
        'learning_rate': 2e-5,
        'epochs': 10,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'weight_decay': 0.01,
    }

    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    dataset = dataset.map(tokenize, batched=True)
    print(dataset['train'])
    print(dataset['validation'])
    print(dataset['test'])

    model = AutoModelForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased', num_labels=2)
    model = model.to(device)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    metric = evaluate.load("accuracy")
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=hyper_parameters['learning_rate'],
        per_device_train_batch_size=hyper_parameters['per_device_train_batch_size'],
        per_device_eval_batch_size=hyper_parameters['per_device_eval_batch_size'],
        num_train_epochs=hyper_parameters['epochs'],
        weight_decay=hyper_parameters['weight_decay'],
        logging_steps=100,
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    COMET_API_KEY = os.getenv("COMET_API_KEY")
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="BERT",
        workspace="france020800"
    )

    print("Fine-tuning...")
    trainer.train()

    print("\nEvaluating...")
    evaluation_results = trainer.evaluate()
    print(evaluation_results)