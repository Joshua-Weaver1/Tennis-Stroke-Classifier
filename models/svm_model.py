import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def create_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size + 1):
        X_windows.append(X[i:i+window_size])
        y_windows.append(y[i+window_size-1])
    return np.array(X_windows), np.array(y_windows)

def calculate_metrics(csv_file_path, window_size=None):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Separate features and target variable
    X = data[['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
    y = data['shot']

    # Check if window_size is provided
    if window_size:
        X, y = create_windows(X, y, window_size)
        X = X.reshape(X.shape[0], -1)  # Flatten windows into 1D arrays

    # Initialize SVM classifier
    svm_classifier = SVC(kernel='rbf', random_state=42)

    # Perform cross-validation
    scores = cross_val_score(svm_classifier, X, y, cv=5)

    # Calculate metrics using classification report
    y_pred = cross_val_predict(svm_classifier, X, y, cv=5)
    metrics_report = classification_report(y, y_pred, output_dict=True)

    # Extract accuracy, recall, precision, and F1-score from the classification report
    accuracy = np.mean(scores)
    recall = metrics_report['weighted avg']['recall']
    precision = metrics_report['weighted avg']['precision']
    f1_score = metrics_report['weighted avg']['f1-score']

    return accuracy, recall, precision, f1_score


# Example usage:
accuracy, precision, recall, f1_score = calculate_metrics("data/all_shot_data.csv")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)