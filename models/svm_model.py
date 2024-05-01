import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Dictionary to store trained models
trained_models = {}

def create_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size + 1):
        X_windows.append(X[i:i+window_size])
        y_windows.append(y[i+window_size-1])
    return np.array(X_windows), np.array(y_windows)

def calculate_metrics(csv_file_path, window_size=None, sampling_rate="100Hz"):
    """
    Calculate multiple classification metrics for a Support Vector Machine (SVM) Classifier using cross-validation.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
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
    model_key = f"window={window_size}_sampling_rate={sampling_rate}"
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

    # Initialize SVM classifier
    svm_classifier = SVC(kernel='rbf', random_state=42)

    # Perform cross-validation
    y_pred = cross_val_predict(svm_classifier, X, y, cv=10)

    # Calculate correct and total guesses for each class
    unique_classes = np.unique(y)
    correct_guesses = {label: 0 for label in unique_classes}
    total_guesses = {label: 0 for label in unique_classes}
    for true_label, pred_label in zip(y, y_pred):
        total_guesses[true_label] += 1
        if true_label == pred_label:
            correct_guesses[true_label] += 1

    # Calculate metrics using classification report
    metrics_report = classification_report(y, y_pred, output_dict=True)

    # Extract accuracy, recall, precision, and F1-score from the classification report
    accuracy = metrics_report['accuracy']
    recall = metrics_report['weighted avg']['recall']
    precision = metrics_report['weighted avg']['precision']
    f1_score = metrics_report['weighted avg']['f1-score']

    # Store the trained model in the dictionary
    trained_models[model_key] = (accuracy, recall, precision, f1_score, correct_guesses, total_guesses)

    return accuracy, recall, precision, f1_score, correct_guesses, total_guesses
