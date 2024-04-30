import unittest
from unittest.mock import patch
import svm_model

class TestSVMModel(unittest.TestCase):

    def test_create_windows(self):
        # Test the create_windows function
        print("Testing Windows Creation")
        X = [1, 2, 3, 4, 5]
        y = [0, 1, 0, 1, 0]
        window_size = 3
        X_windows, y_windows = svm_model.create_windows(X, y, window_size)
        self.assertEqual(len(X_windows), 3)
        self.assertEqual(len(y_windows), 3)
        self.assertEqual(X_windows[0].tolist(), [1, 2, 3])

    def test_calculate_metrics(self):
        # Test calculate_metrics returns all values
        print("Testing Results For SVM")
        accuracy, recall, precision, f1_score, correct_guesses, total_guesses = svm_model.calculate_metrics("data/all_shot_data.csv")
        
        self.assertIsNotNone(accuracy, "Accuracy should not be None")
        self.assertIsNotNone(recall, "Recall should not be None")
        self.assertIsNotNone(precision, "Precision should not be None")
        self.assertIsNotNone(f1_score, "F1 Score should not be None")
        self.assertIsNotNone(correct_guesses, "Correct guesses should not be None")
        self.assertIsNotNone(total_guesses, "Total guesses should not be None")

    def test_calculate_metrics_score(self):
        # Evaluate the calculate_metrics function for performance
        print("Evaluating Results For SVM")
        accuracy, recall, precision, f1_score, correct_guesses, total_guesses = svm_model.calculate_metrics("data/all_shot_data.csv")
        
        self.assertTrue(accuracy > 0.69, f"Accuracy is too low: {accuracy}")
        print(f"Accuracy: {accuracy}")
        
        self.assertTrue(recall > 0.69, f"Recall is too low: {recall}")
        print(f"Recall: {recall}")
        
        self.assertTrue(precision > 0.69, f"Precision is too low: {precision}")
        print(f"Precision: {precision}")
        
        self.assertTrue(f1_score > 0.69, f"F1 Score is too low: {f1_score}")
        print(f"F1 Score: {f1_score}")

if __name__ == '__main__':
    unittest.main()
