import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def calculate_metrics(csv_file_path, k=5):
    """
    Calculate multiple classification metrics for a K-Nearest Neighbors (KNN) model using cross-validation.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - k (int): Number of neighbors for the KNN classifier.

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
