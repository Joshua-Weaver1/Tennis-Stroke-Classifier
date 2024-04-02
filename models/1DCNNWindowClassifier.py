import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/AllWristMotion.csv')

# Define window size
window_size = 10  # You can adjust this value

# Aggregate data into windows and extract features
X_windows = []
y_windows = []

# Reshape data for 1D CNN input
X = data[['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ', 'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
y = data['shot']

# Define a function to aggregate data into windows
def aggregate_into_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(0, len(X) - window_size + 1, window_size):
        window = X.iloc[i:i+window_size]
        X_windows.append(window.mean().values)
        y_windows.append(y.iloc[i])  # Assuming each window has the same label
    return np.array(X_windows), np.array(y_windows)

# Aggregate data into windows
X_windows, y_windows = aggregate_into_windows(X, y, window_size)

# Normalize features
scaler = StandardScaler()
X_windows = scaler.fit_transform(X_windows)

# Encode target labels
label_encoder = LabelEncoder()
y_windows = label_encoder.fit_transform(y_windows)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

# Initialize lists to store cross-validation scores
cv_scores = []

# Perform cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in k_fold.split(X_windows):
    X_train, X_val = X_windows[train_index], X_windows[test_index]
    y_train, y_val = y_windows[train_index], y_windows[test_index]

    # Build 1D CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on validation set
    _, accuracy = model.evaluate(X_val, y_val)
    cv_scores.append(accuracy)

# Output cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Evaluate the model on test set
_, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_accuracy)

# Plot mean and standard deviation for each shot type
plt.figure(figsize=(12, 6))
for shot_type in range(4):
    shot_data = X_windows[y_windows == shot_type]
    mean_data = np.mean(shot_data, axis=0)
    std_data = np.std(shot_data, axis=0)
    plt.plot(mean_data, label=f'Shot Type {shot_type} (Mean)')
    plt.fill_between(range(mean_data.shape[0]), mean_data - std_data, mean_data + std_data, alpha=0.3)
plt.title('Mean and Standard Deviation for Each Shot Type')
plt.xlabel('Feature Index')
plt.ylabel('Sensor Data')
plt.legend()
plt.grid(True)
plt.show()
