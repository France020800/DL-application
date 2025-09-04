from transformers import pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, get_dataset_split_names
import numpy as np


def main():
    ## Exercise 1.1: Load the Rotten Tomatoes dataset and print its shape and split
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    print(dataset.shape)

    splits = get_dataset_split_names("cornell-movie-review-data/rotten_tomatoes")
    print(splits)

    train_data = dataset['train']
    test_data = dataset["test"]
    val_data = dataset["validation"]
    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    # Positive review
    positive_review = next(item for item in train_data if item['label'] == 1)
    print(positive_review)

    # Negative review
    negative_review = next(item for item in train_data if item['label'] == 0)
    print(negative_review)


    ## Exercise 1.2: Load the Distilbert model and corresponding tokenizer.
    model = AutoModel.from_pretrained('distilbert/distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    # Take the first 3 samples
    samples = train_data['text'][:3]

    tokens = tokenizer(samples, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**tokens)
    print('Model outputs:', outputs)


    ## Exercise 1.3: Create a baseline to evaluate the performance of a SVM classifier on the Rotten Tomatoes dataset using Distillber for feature extraction
    # Split the dataset into train, validation, and test sets
    train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
    val_texts, val_labels = dataset["validation"]["text"], dataset["validation"]["label"]
    test_texts, test_labels = dataset["test"]["text"], dataset["test"]["label"]

    # Initialize the feature extraction pipeline
    feature_extractor = pipeline("feature-extraction", model="distilbert/distilbert-base-uncased", tokenizer="distilbert/distilbert-base-uncased")

    # Extract features for the train, validation, and test sets
    def extract_features(texts):
        features = feature_extractor(texts, padding=True, truncation=True, return_tensors="pt")
        return np.array([f[0][0].numpy() for f in features])

    train_features = extract_features(train_texts)
    val_features = extract_features(val_texts)
    test_features = extract_features(test_texts)

    # Train a classifier (SVM)
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(train_features, train_labels)

    # Evaluate on the validation set
    val_predictions = classifier.predict(val_features)
    print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))
    print("Validation Classification Report:\n", classification_report(val_labels, val_predictions))

    # Evaluate on the test set
    test_predictions = classifier.predict(test_features)
    print("Test Accuracy:", accuracy_score(test_labels, test_predictions))
    print("Test Classification Report:\n", classification_report(test_labels, test_predictions))


if __name__ == '__main__':
    main()