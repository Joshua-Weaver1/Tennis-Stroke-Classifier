import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Read the CSV file
try:
    data = pd.read_csv("Data/AlexWristMotion.csv")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()
except Exception as e:
    print("An error occurred:", e)
    exit()

# Check for missing values
if data.isnull().sum().any():
    print("Warning: Missing values detected. Handle them appropriately.")

# Separate features and target variable
X = data[['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
y = data['shot']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn_classifier.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = knn_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))