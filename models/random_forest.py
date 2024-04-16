import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def calculate_metrics(csv_file_path):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Separate features and target variable
    X = data[['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
    y = data['shot']

    # Initialize random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(rf_classifier, X, y, cv=5)

    # Calculate metrics using classification report
    y_pred = cross_val_predict(rf_classifier, X, y, cv=5)
    metrics_report = classification_report(y, y_pred, output_dict=True)

    # Extract accuracy, recall, precision, and F1-score from the classification report
    accuracy = np.mean(scores)
    recall = metrics_report['weighted avg']['recall']
    precision = metrics_report['weighted avg']['precision']
    f1_score = metrics_report['weighted avg']['f1-score']

    return accuracy, recall, precision, f1_score
