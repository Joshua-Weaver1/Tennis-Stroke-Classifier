import tkinter as tk
from tkinter import messagebox
import os
import importlib

class TennisClassifierGUI:
    def __init__(self, master, csv_files, models):
        self.master = master
        self.master.title("Tennis Shot Classifier")

        # Create frame to hold widgets
        self.frame = tk.Frame(self.master)
        self.frame.pack()

        # Entry widget for CSV file selection
        self.file_entry_label = tk.Label(self.frame, text="Select CSV file:")
        self.file_entry_label.pack()
        self.selected_file = tk.StringVar(master)
        self.selected_file.set(csv_files[0])  # Set the default value
        self.file_menu = tk.OptionMenu(self.frame, self.selected_file, *csv_files)
        self.file_menu.pack()

        # Entry widget for model selection
        self.model_entry_label = tk.Label(self.frame, text="Select Model:")
        self.model_entry_label.pack()
        self.selected_model = tk.StringVar(master)
        self.selected_model.set(models[0])  # Set the default value
        self.model_menu = tk.OptionMenu(self.frame, self.selected_model, *models)
        self.model_menu.pack()

        # Entry widget for K value (only shown when KNN model is selected)
        self.k_entry_label = tk.Label(self.frame, text="Enter K value:")
        self.k_entry_label.pack()
        self.k_entry = tk.Entry(self.frame)
        self.k_entry.pack()

        # Run Classification Button
        self.classify_button = tk.Button(self.frame, text="Run Classification", command=self.run_classification)
        self.classify_button.pack()

        # Result Label
        self.result_label = tk.Label(self.frame, text="")
        self.result_label.pack()

    def run_classification(self):
        # Get the selected file name, model name, and value of k (if applicable)
        file_name = self.selected_file.get()
        model_name = self.selected_model.get()
        k_value = self.k_entry.get() if model_name == "knn" else None

        # Import the selected model dynamically
        model_module = importlib.import_module(f"models.{model_name}")

        # Call the calculate_metrics function from the selected model
        try:
            if k_value:
                k_value = int(k_value)  # Convert k_value to integer
                accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}", k=k_value)
            else:
                accuracy, recall, precision, f1_score = model_module.calculate_metrics(f"data/{file_name}")

            result_text = f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nF1-Score: {f1_score}"
            self.result_label.config(text=result_text)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    # Get list of CSV files in the "data" directory
    csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]

    # Get list of model files in the "models" directory
    models = [f.split(".")[0] for f in os.listdir("models") if f.endswith('.py') and f != "__init__.py"]

    root = tk.Tk()
    root.geometry("400x500")  # Width x Height
    app = TennisClassifierGUI(root, csv_files, models)
    root.mainloop()

if __name__ == "__main__":
    main()
