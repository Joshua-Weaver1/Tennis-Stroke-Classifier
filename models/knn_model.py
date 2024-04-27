import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

# Dictionary to store trained models
trained_models = {}

def create_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size + 1):
        X_windows.append(X[i:i+window_size])
        y_windows.append(y[i+window_size-1])
    return np.array(X_windows), np.array(y_windows)

def calculate_metrics(csv_file_path, k=5, window_size=None, sampling_rate="100Hz"):
    """
    Calculate multiple classification metrics for a K-Nearest Neighbors (KNN) model using cross-validation.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - k (int): Number of neighbors for the KNN classifier.
    - window_size (int): Size of the window if using windows for classification.
    - sampling_rate (str): Sampling rate for selecting training data.

    Returns:
    - accuracy (float): Mean accuracy across cross-validation folds.
    - recall (float): Weighted average recall.
    - precision (float): Weighted average precision.
    - f1_score (float): Weighted average F1-score.
    - correct_guesses (dict): Dictionary containing the number of correct guesses for each class.
    - total_guesses (dict): Dictionary containing the total number of guesses for each class.
    """
    # Check if model with the same parameters is already trained
    model_key = f"k={k}_window={window_size}_sampling_rate={sampling_rate}"
    if model_key in trained_models:
        return trained_models[model_key]

    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Separate features and target variable
    X = data[['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
    y = data['shot']

    # Adjust training data based on sampling rate
    if sampling_rate == "100Hz":
        pass  # Use all rows
    elif sampling_rate == "50Hz":
        X = X.iloc[::2]  # Use every other row
        y = y.iloc[::2]
    elif sampling_rate == "20Hz":
        X = X.iloc[::5]  # Use every 5th row
        y = y.iloc[::5]

    # Check if window_size is provided
    if window_size:
        X, y = create_windows(X, y, window_size)
        X = X.reshape(X.shape[0], -1)  # Flatten windows into 1D arrays

    # Initialize KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Perform cross-validation
    scores = cross_val_score(knn_classifier, X, y, cv=100)

    # Calculate metrics using classification report
    y_pred = cross_val_predict(knn_classifier, X, y, cv=100)
    metrics_report = classification_report(y, y_pred, output_dict=True)

    # Extract accuracy, recall, precision, and F1-score from the classification report
    accuracy = np.mean(scores)
    recall = metrics_report['weighted avg']['recall']
    precision = metrics_report['weighted avg']['precision']
    f1_score = metrics_report['weighted avg']['f1-score']

    # Calculate correct guesses and total guesses for each class
    correct_guesses = {}
    total_guesses = {}
    for label in set(y):
        correct_guesses[label] = np.sum((y == label) & (y_pred == label))
        total_guesses[label] = np.sum(y_pred == label)

    # Store the trained model and metrics in the dictionary
    trained_models[model_key] = (accuracy, recall, precision, f1_score, correct_guesses, total_guesses)

    return accuracy, recall, precision, f1_score, correct_guesses, total_guesses
