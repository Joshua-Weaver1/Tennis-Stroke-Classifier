import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('data/all_shot_data.csv')

# Separate data by shot type
shot_types = {0: 'No Shot', 1: 'Forehand', 2: 'Backhand', 3: 'Overhead'}
shot_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}

features = ['rotationRateX', 'rotationRateY', 'rotationRateZ', 'gravityX', 'gravityY', 'gravityZ',
            'accelerationX', 'accelerationY', 'accelerationZ', 'quaternionW', 'quaternionX',
            'quaternionY', 'quaternionZ']

for feature in features:
    plt.figure(figsize=(10, 6))
    plt.title(f'{feature} over Time')
    plt.xlabel('Time')
    plt.ylabel(feature)
    
    for shot_type in range(4):
        shot_data = data[data['shot'] == shot_type]
        plt.plot(shot_data['seconds_elapsed'], shot_data[feature], label=shot_types[shot_type], color=shot_colors[shot_type])
    
    plt.legend()
    plt.show()
