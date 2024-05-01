import unittest
import cnn_1d_model

class Test1DCNNModel(unittest.TestCase):
    
    def test_create_cnn_model(self):
        # Test Model Creation

        # Define input shape
        input_shape = (100, 13)  # Example input shape
        
        # Define number of classes
        num_classes = 5  # Example number of classes
        
        # Create CNN model
        model = cnn_1d_model.create_cnn_model(input_shape, num_classes)
        
        # Get model output shape
        output_shape = model.output_shape[1]  # Exclude batch dimension
        
        # Check if output shape matches number of classes
        self.assertEqual(output_shape, num_classes)

    def test_calculate_metrics(self):
        # Test calculate_metrics returns all values
        print("Testing Results For 1D CNN")
        accuracy, recall, precision, f1_score, correct_guesses, total_guesses = cnn_1d_model.calculate_metrics("data/all_shot_data.csv")
        
        self.assertIsNotNone(accuracy, "Accuracy should not be None")
        self.assertIsNotNone(recall, "Recall should not be None")
        self.assertIsNotNone(precision, "Precision should not be None")
        self.assertIsNotNone(f1_score, "F1 Score should not be None")
        self.assertIsNotNone(correct_guesses, "Correct guesses should not be None")
        self.assertIsNotNone(total_guesses, "Total guesses should not be None")

    def test_calculate_metrics_score(self):
        # Evaluate the calculate_metrics function for performance
        print("Evaluating Results For 1D CNN")
        accuracy, recall, precision, f1_score, correct_guesses, total_guesses = cnn_1d_model.calculate_metrics("data/all_shot_data.csv")
        
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
