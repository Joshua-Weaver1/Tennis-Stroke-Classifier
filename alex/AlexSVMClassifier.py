import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Read the CSV file
data = pd.read_csv("data/AlexWristMotion.csv")

# Define window size
window_size = 10  # You can adjust this value

# Aggregate data into windows and extract features
def extract_features(window):
    features = []
    for column in window.columns:
        features.append(window[column].mean())  # You can extract other features as needed
    return features

# Select columns of interest
columns_of_interest = ['seconds_elapsed', 'rotationRateX', 'rotationRateY', 'rotationRateZ', 
                       'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 
                       'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ', 
                       'shot']
data = data[columns_of_interest]

X_windows = []
y_windows = []

for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i+window_size]
    X_windows.append(window.mean().values)  # Use mean of window as feature vector
    y_windows.append(data['shot'].iloc[i])  # Assuming each window has the same label

# Convert lists to numpy arrays
X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict the labels for test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Perform cross-validation
cv_scores = cross_val_score(svm_classifier, X_windows, y_windows, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
