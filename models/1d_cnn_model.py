import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import joblib

# Dictionary to store trained models
trained_models = {}

def preprocess_data(csv_file_path, window_size=200):
    """
    Preprocesses the data from a CSV file by creating windows of data.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - window_size (int): Size of the window for creating data segments.

    Returns:
    - X (numpy.ndarray): Array containing the windowed feature data.
    - y (numpy.ndarray): Array containing the corresponding target labels.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Create windows of data
    windows_X = []
    windows_y = []
    for i in range(0, len(data), window_size):
        window_data = data.iloc[i:i+window_size]
        if len(window_data) == window_size:  # Ensure window is complete
            most_common_shot = window_data['shot'].mode()[0]
            windows_X.append(window_data.drop(columns=['time', 'seconds_elapsed', 'shot']).values)
            windows_y.append(most_common_shot)

    return np.array(windows_X), np.array(windows_y)

def create_cnn_model(input_shape, num_classes):
    """
    Creates a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape (tuple): Shape of the input data.
    - num_classes (int): Number of classes for classification.

    Returns:
    - model (Sequential): CNN model.
    """
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def calculate_metrics(csv_file_path, window_size=200, sampling_rate="100Hz"):
    """
    Calculates classification metrics for a CNN model using preprocessed data.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - window_size (int): Size of the window for creating data segments.
    - sampling_rate (str): Sampling rate for selecting training data.

    Returns:
    - accuracy (float): Accuracy of the model.
    - precision (float): Weighted average precision.
    - recall (float): Weighted average recall.
    - f1_score (float): Weighted average F1-score.
    """
    # Check if model with the same parameters is already trained
    model_key = f"window={window_size}_sampling_rate={sampling_rate}"
    if model_key in trained_models:
        return trained_models[model_key]

    # Preprocess the data with the given window size and sampling rate
    X, y = preprocess_data(csv_file_path, window_size)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Define model
    input_shape = X_train.shape[1:]
    model = create_cnn_model(input_shape, num_classes)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate precision, recall, and F1-score
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    
    # Store the trained model in the dictionary
    trained_models[model_key] = (accuracy, precision, recall, f1_score)

    return accuracy, precision, recall, f1_score