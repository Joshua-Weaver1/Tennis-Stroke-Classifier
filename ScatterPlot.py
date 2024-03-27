import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_file = input("Enter the data file: ").lower()

# Read the CSV file into a DataFrame
df = pd.read_csv("Data/" + data_file + "WristMotion.csv")

# Function to plot data based on user input
def plot_data(data_type):
    if data_type == 'acceleration':
        x = df['accelerationX']
        y = df['accelerationY']
        z = df['accelerationZ']
        title = 'Accelerometer Data'
    elif data_type == 'gravity':
        x = df['gravityX']
        y = df['gravityY']
        z = df['gravityZ']
        title = 'Gravity Data'
    elif data_type == 'gyroscope':
        x = df['rotationRateX']
        y = df['rotationRateY']
        z = df['rotationRateZ']
        title = 'Gyroscope Data'
    else:
        print("Invalid input!")
        return

    # Plot the selected data in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Scatter Plot of {title}')

    plt.show()

# Get user input
data_type = input("Enter the data type (acceleration, gravity, gyroscope): ").lower()

# Plot the selected data
plot_data(data_type)