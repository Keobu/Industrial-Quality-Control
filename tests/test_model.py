import unittest
import numpy as np
import os
from src.model import DefectClassifier

class TestModelCore(unittest.TestCase):
    def setUp(self):
        self.classifier = DefectClassifier(kernel='linear')
        self.temp_model_path = "models/test_model_temporary.joblib"

    def tearDown(self):
        if os.path.exists(self.temp_model_path):
            os.remove(self.temp_model_path)

    def test_training_and_prediction_shapes(self):

        X_mock = np.random.rand(4, 50)
        y_mock = np.array([0, 0, 1, 1])
        
        self.classifier.train(X_mock, y_mock)
        preds = self.classifier.predict(X_mock)
        probs = self.classifier.predict_proba(X_mock)
        
        self.assertEqual(len(preds), 4)
        self.assertEqual(probs.shape, (4, 2))

    def test_model_serialization_persistence(self):

        X_mock = np.random.rand(4, 50)
        y_mock = np.array([0, 0, 1, 1])
        
        self.classifier.train(X_mock, y_mock)
        original_preds = self.classifier.predict(X_mock)
        
        # Save and reload
        self.classifier.save_model(self.temp_model_path)
        new_classifier = DefectClassifier()
        new_classifier.load_model(self.temp_model_path)
        
        reloaded_preds = new_classifier.predict(X_mock)
        np.testing.assert_array_equal(original_preds, reloaded_preds)

if __name__ == "__main__":
    unittest.main()