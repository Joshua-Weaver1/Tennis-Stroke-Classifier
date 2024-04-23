import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def create_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size + 1):
        X_windows.append(X[i:i+window_size])
        y_windows.append(y[i+window_size-1])
    return np.array(X_windows), np.array(y_windows)

def calculate_metrics(csv_file_path, k=5, window_size=None):
    """
    Calculate multiple classification metrics for a K-Nearest Neighbors (KNN) model using cross-validation.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - k (int): Number of neighbors for the KNN classifier.
    - window_size (int): Size of the window if using windows for classification.

    Returns:
    - accuracy (float): Mean accuracy across cross-validation folds.
    - recall (float): Weighted average recall.
    - precision (float): Weighted average precision.
    - f1_score (float): Weighted average F1-score.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Separate features and target variable
    X = data[['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
    y = data['shot']

    # Check if window_size is provided
    if window_size:
        X, y = create_windows(X, y, window_size)
        X = X.reshape(X.shape[0], -1)  # Flatten windows into 1D arrays

    # Initialize KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Perform cross-validation
    scores = cross_val_score(knn_classifier, X, y, cv=5)

    # Calculate metrics using classification report
    y_pred = cross_val_predict(knn_classifier, X, y, cv=5)
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