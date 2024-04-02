import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Read the CSV file
data = pd.read_csv("data/AllWristMotion.csv")

# Define window size
window_size = 10  # You can adjust this value

# Aggregate data into windows and extract features
def extract_features(window):
    features = []
    for column in window.columns:
        features.append(window[column].mean())  # You can extract other features as needed
    return features

X_windows = []
y_windows = []
for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i+window_size]
    X_windows.append(extract_features(window.drop('shot', axis=1)))
    y_windows.append(window['shot'].iloc[0])  # Assuming each window has the same label

# Convert lists to numpy arrays
X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

# Separate features and target variable
X = X_windows
y = y_windows

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print Feature Importances
print("Feature Importances:")
for feature, importance in zip(range(X.shape[1]), rf_classifier.feature_importances_):
    print(f"Feature {feature}: {importance}")

# Print Cross-Validation Scores
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))