import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the CSV file
try:
    data = pd.read_csv("Data/JoshWristMotion.csv")
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

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Perform cross-validation
cv_scores = cross_val_score(knn_classifier, X_train, y_train, cv=5)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Predictions
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print Cross Validation Scores
print("Cross Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
