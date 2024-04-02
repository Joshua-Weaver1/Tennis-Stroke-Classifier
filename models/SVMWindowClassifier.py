import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to create sliding windows
def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.array(windows)

# Read the CSV file
data = pd.read_csv("data/AllWristMotion.csv")

# Separate features and target variable
X = data[['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
y = data['shot']

# Set window size
window_size = 10

# Create windows for X and y
X_windows = create_windows(X.values, window_size)
y_windows = create_windows(y.values, window_size)

# Flatten y windows to single labels
y_windows = np.any(y_windows, axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=42)

# Train the classifier
svm_classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Predictions
y_pred = svm_classifier.predict(X_test.reshape(X_test.shape[0], -1))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print Cross-Validation Scores
cv_scores = cross_val_score(svm_classifier, X_windows.reshape(X_windows.shape[0], -1), y_windows, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
