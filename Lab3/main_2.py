import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate
from transformers import TrainingArguments, Trainer
from comet_ml import Experiment
import os
import utils

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

if __name__ == '__main__':
    hyper_params = {
        'learning_rate': 2e-5,
        'epochs': 10,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'weight_decay': 0.01,
    }

    utils.set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    dataset = dataset.map(utils.tokenize, batched=True)
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
        learning_rate=hyper_params['learning_rate'],
        per_device_train_batch_size=hyper_params['per_device_train_batch_size'],
        per_device_eval_batch_size=hyper_params['per_device_eval_batch_size'],
        num_train_epochs=hyper_params['epochs'],
        weight_decay=hyper_params['weight_decay'],
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
        compute_metrics=utils.compute_metrics,
    )

    COMET_API_KEY = os.getenv("COMET_API_KEY")
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="BERT",
        workspace="france020800"
    )
    experiment.log_parameters(hyper_params)

    print("Fine-tuning...")
    trainer.train()

    print("\nEvaluating...")
    evaluation_results = trainer.evaluate()
    print(evaluation_results)