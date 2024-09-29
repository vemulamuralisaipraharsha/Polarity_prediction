import os
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

def load_data(neg_file, pos_file):
    with open(neg_file, 'r', encoding='latin-1') as f_neg, open(pos_file, 'r', encoding='latin-1') as f_pos:
        neg_sentences = f_neg.readlines()
        pos_sentences = f_pos.readlines()
    return neg_sentences, pos_sentences

def split_data(neg_sentences, pos_sentences):
    X_train = neg_sentences[:4000] + pos_sentences[:4000]
    y_train = [0] * 4000 + [1] * 4000

    X_val = neg_sentences[4000:4500] + pos_sentences[4000:4500]
    y_val = [0] * 500 + [1] * 500

    X_test = neg_sentences[4500:] + pos_sentences[4500:]
    y_test = [0] * 831 + [1] * 831

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_svm(X_train, y_train, X_val, y_val):
    vectorizer = CountVectorizer()
    
    model = make_pipeline(vectorizer, SVC(kernel='linear'))

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, val_predictions))

    return model

def test_model(model, X_test, y_test):
    test_predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = f1_score(y_test, test_predictions)
    accuracy = accuracy_score(y_test, test_predictions)

    print("Test Set Performance:")
    print(f"Confusion Matrix: [[TN: {tn}, FP: {fp}], [FN: {fn}, TP: {tp}]]")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return tn, fp, fn, tp, accuracy, precision, recall, f1

if __name__ == "__main__":
    neg_file = 'rt-polarity.neg'
    pos_file = 'rt-polarity.pos'

    neg_sentences, pos_sentences = load_data(neg_file, pos_file)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(neg_sentences, pos_sentences)

    model = train_svm(X_train, y_train, X_val, y_val)

    tn, fp, fn, tp, accuracy, precision, recall, f1 = test_model(model, X_test, y_test)

    print("\n--- Final Test Set Results ---")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
