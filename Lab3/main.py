from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from transformers import EarlyStoppingCallback, DataCollatorWithPadding
from datasets import load_dataset
from comet_ml import Experiment
import evaluate
import os
import utils

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

if __name__ == "__main__":
    set_seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    COMET_API_KEY = os.getenv("COMET_API_KEY")

    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="BERT",
        workspace="france020800"
    )

    hyper_param = {
        "r": 8,
        "epochs": 10,
        "batch_size": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "weight_decay": 0.001,
        "target_modules": ["q_lin", "k_lin", "v_lin", "out_lin"],
        "learning_rate": 2e-5,
        "scheduler_type": "cosine_with_restarts",
        "early_stopping_patience": 5,
        "early_stopping_threshold": 0.001
    }

    experiment.log_parameters(hyper_param)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=hyper_param["early_stopping_patience"],
        early_stopping_threshold=hyper_param["early_stopping_threshold"]
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=hyper_param["r"],
        lora_alpha=hyper_param["lora_alpha"],
        lora_dropout=hyper_param["lora_dropout"],
        bias='none',
        target_modules=hyper_param["target_modules"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to("cuda")

    dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes')
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    dataset = dataset.map(tokenize, batched=True)

    metric = evaluate.load("accuracy")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    token_count = len(dataset['train']) * 128
    train_batch_size = 16
    num_steps = (len(dataset['train']) // train_batch_size) * 4
    warmup_steps = int(0.3 * num_steps)

    training_args = TrainingArguments(
        output_dir="./results_LoRA",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=hyper_param["learning_rate"],
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=hyper_param["batch_size"],
        num_train_epochs=hyper_param["epochs"],
        weight_decay=hyper_param["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine_with_restarts",
        report_to="comet_ml",
        logging_dir="./logs",
        logging_steps=100,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=utils.compute_metrics,
        data_collator=data_collator,
        # callbacks=[early_stopping_callback]
    )

    print("LoRA fine-tuning...")
    trainer.train()

    print("\nEvaluating...")
    evaluation_results = trainer.evaluate()
    print(evaluation_results)