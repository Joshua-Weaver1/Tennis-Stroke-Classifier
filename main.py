import tkinter as tk
from tkinter import messagebox
import os
import importlib

class TennisClassifierGUI:
    """
    Graphical User Interface for the Tennis Shot Classifier application.
    """

    def __init__(self, master, csv_files, models):
        """
        Initialize the GUI.

        Parameters:
        - master: Parent Tkinter window.
        - csv_files (list): List of CSV files.
        - models (list): List of available model names.
        """
        self.master = master
        self.master.title("Tennis Shot Classifier")

        # Create main frame to hold subframes
        self.frame = tk.Frame(self.master, bg="grey", padx=20, pady=20)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Selection Box (Top Left)
        selection_frame = tk.Frame(self.frame, bg="white", padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        selection_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)
        selection_title_frame, selection_content_frame = self.create_half_split_frames(selection_frame)
        self.create_selection_widgets(selection_title_frame, selection_content_frame, csv_files, models)

        # Optional Parameters Box (Bottom Left)
        optional_params_frame = tk.Frame(self.frame, bg="white", padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        optional_params_frame.grid(row=1, column=0, padx=10, pady=10, sticky=tk.NSEW)
        optional_params_title_frame, optional_params_content_frame = self.create_half_split_frames(optional_params_frame)
        self.create_optional_params_widgets(optional_params_title_frame, optional_params_content_frame)

        # Metrics Box (Bottom Right)
        metrics_frame = tk.Frame(self.frame, bg="white", padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        metrics_frame.grid(row=1, column=1, padx=10, pady=10, sticky=tk.NSEW)
        metrics_title_frame, metrics_content_frame = self.create_half_split_frames(metrics_frame)
        self.create_metrics_widgets(metrics_content_frame)

        # Plot Box (Top Right)
        plot_frame = tk.Frame(self.frame, bg="white", padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NSEW)
        plot_title_frame, plot_content_frame = self.create_half_split_frames(plot_frame)
        # You can add plot visualization here

        # Configure grid weights to make frames expand equally
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

        # Add title bars
        self.add_title_bar(selection_title_frame, "Selection", "#015249")
        self.add_title_bar(optional_params_title_frame, "Optional Parameters", "#015249")
        self.add_title_bar(metrics_title_frame, "Metrics", "#015249")
        self.add_title_bar(plot_title_frame, "Plot", "#015249")

        # Initialize result labels
        self.accuracy_label = tk.Label(metrics_content_frame, text="Accuracy: ", bg="white", font=("Arial", 12))
        self.accuracy_label.pack(fill=tk.X, padx=5, pady=(5, 0))
        self.recall_label = tk.Label(metrics_content_frame, text="Recall: ", bg="white", font=("Arial", 12))
        self.recall_label.pack(fill=tk.X, padx=5, pady=5)
        self.precision_label = tk.Label(metrics_content_frame, text="Precision: ", bg="white", font=("Arial", 12))
        self.precision_label.pack(fill=tk.X, padx=5, pady=5)
        self.f1_score_label = tk.Label(metrics_content_frame, text="F1-Score: ", bg="white", font=("Arial", 12))
        self.f1_score_label.pack(fill=tk.X, padx=5, pady=5)

        # Add Exit Button
        exit_button = tk.Button(selection_content_frame, text="Exit", bg="red", fg="white", font=("Arial", 12), command=self.exit_application)
        exit_button.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    def exit_application(self):
        """
        Exits the application.
        """
        self.master.destroy()

    def add_title_bar(self, parent_frame, title, color):
        """
        Adds a title bar to a frame.

        Parameters:
        - parent_frame: Parent frame to add the title bar to.
        - title (str): Title text.
        - color (str): Background color of the title bar.
        """
        title_bar = tk.Label(parent_frame, text=title, bg=color, fg="white", font=("Arial", 12, "bold"))
        title_bar.pack(side="top", fill="x", padx=5, pady=(5, 0))

    def create_half_split_frames(self, parent_frame):
        """
        Creates two frames split vertically in half.

        Parameters:
        - parent_frame: Parent frame to contain the two split frames.

        Returns:
        - title_frame: Top frame for the title bar.
        - content_frame: Bottom frame for content.
        """
        title_frame = tk.Frame(parent_frame, bg="white")
        title_frame.pack(fill=tk.X)
        content_frame = tk.Frame(parent_frame, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True)
        return title_frame, content_frame

    def create_selection_widgets(self, parent_title_frame, parent_content_frame, csv_files, models):
        """
        Creates widgets for selecting CSV file and model.

        Parameters:
        - parent_title_frame: Parent frame for the title bar.
        - parent_content_frame: Parent frame for content.
        - csv_files (list): List of available CSV files.
        - models (list): List of available model names.
        """
        # Entry widget for CSV file selection
        file_entry_label = tk.Label(parent_content_frame, text="Select CSV file:", bg="white", font=("Arial", 12))
        file_entry_label.grid(row=0, column=0, padx=(0, 10), pady=(10, 5), sticky="e")
        self.selected_file = tk.StringVar()
        self.selected_file.set(csv_files[0])  # Set the default value
        file_menu = tk.OptionMenu(parent_content_frame, self.selected_file, *csv_files)
        file_menu.grid(row=0, column=1, padx=(0, 10), pady=(10, 5), sticky="ew")

        # Entry widget for model selection
        model_entry_label = tk.Label(parent_content_frame, text="Select Model:", bg="white", font=("Arial", 12))
        model_entry_label.grid(row=1, column=0, padx=(0, 10), pady=5, sticky="e")
        self.selected_model = tk.StringVar()
        self.selected_model.set(models[0])  # Set the default value
        model_menu = tk.OptionMenu(parent_content_frame, self.selected_model, *models)
        model_menu.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="ew")

        # Run Classification Button
        classify_button = tk.Button(parent_content_frame, text="Run Classification", bg="green", fg="white", font=("Arial", 12), command=self.run_classification)
        classify_button.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    def create_optional_params_widgets(self, parent_title_frame, parent_content_frame):
        """
        Creates widgets for optional parameters (K value, Window Size, Sampling Rate).

        Parameters:
        - parent_title_frame: Parent frame for the title bar.
        - parent_content_frame: Parent frame for content.
        """
        # Entry widget for K value (only shown when KNN model is selected)
        k_entry_label = tk.Label(parent_content_frame, text="Enter K value:", bg="white", font=("Arial", 12))
        k_entry_label.grid(row=0, column=0, padx=(0, 10), pady=(10, 5), sticky="e")
        self.k_entry = tk.Entry(parent_content_frame)
        self.k_entry.grid(row=0, column=1, padx=(0, 10), pady=(10, 5), sticky="ew")

        # Entry widget for Window Size
        window_size_label = tk.Label(parent_content_frame, text="Window Size:", bg="white", font=("Arial", 12))
        window_size_label.grid(row=1, column=0, padx=(0, 10), pady=5, sticky="e")
        self.window_size_entry = tk.Entry(parent_content_frame)
        self.window_size_entry.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="ew")

        # Dropdown menu for selecting sampling rate
        sampling_rate_label = tk.Label(parent_content_frame, text="Select Sampling Rate:", bg="white", font=("Arial", 12))
        sampling_rate_label.grid(row=2, column=0, padx=(0, 10), pady=5, sticky="e")
        self.selected_sampling_rate = tk.StringVar()
        self.selected_sampling_rate.set("100Hz")  # Set the default value
        sampling_rate_menu = tk.OptionMenu(parent_content_frame, self.selected_sampling_rate, "100Hz", "50Hz", "20Hz")
        sampling_rate_menu.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="ew")


    def create_metrics_widgets(self, parent_content_frame):
        """
        Creates widgets for displaying classification metrics.

        Parameters:
        - parent_content_frame: Parent frame for content.
        """
        pass  # Metrics labels are created during initialization

    def run_classification(self):
        """
        Runs the classification process based on user input.
        """
        # Get the selected file name, model name, and values of K and window size
        file_name = self.selected_file.get()
        model_name = self.selected_model.get()
        k_value = self.k_entry.get()
        print("Value of k retrieved from GUI:", k_value)  # Debug print statement
        window_size = self.window_size_entry.get()
        sampling_rate = self.selected_sampling_rate.get()  # Get selected sampling rate

        # Import the selected model dynamically
        model_module = importlib.import_module(f"models.{model_name}")

        # Call the calculate_metrics function from the selected model
        try:
            if model_name == "knn_model":
                if k_value and window_size:
                    k_value = int(k_value)
                    window_size = int(window_size)
                    accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}", k=k_value, window_size=window_size, sampling_rate=sampling_rate)
                elif k_value:
                    k_value = int(k_value)
                    accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}", k=k_value, sampling_rate=sampling_rate)
                elif window_size:
                    window_size = int(window_size)
                    accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}", k=5, window_size=window_size, sampling_rate=sampling_rate)
                else:
                    accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}", sampling_rate=sampling_rate)
            else:
                if window_size:
                    window_size = int(window_size)
                    accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}", window_size=window_size, sampling_rate=sampling_rate)
                else:
                    accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}", sampling_rate=sampling_rate)

            # Update the metric labels
            self.accuracy_label.config(text=f"Accuracy: {accuracy}")
            self.recall_label.config(text=f"Recall: {recall}")
            self.precision_label.config(text=f"Precision: {precision}")
            self.f1_score_label.config(text=f"F1-Score: {f1_score}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


def main():
    # Get list of CSV files in the "data" directory
    csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]

    # Get list of model files in the "models" directory
    models = [f.split(".")[0] for f in os.listdir("models") if f.endswith('.py')]

    root = tk.Tk()
    root.geometry("1000x800")  # Width x Height
    app = TennisClassifierGUI(root, csv_files, models)
    root.mainloop()

if __name__ == "__main__":
    main()
