import unittest
import numpy as np
import os
import cv2
from src.preprocessing import IndustrialPreprocessor

class TestIndustrialPipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = IndustrialPreprocessor(target_size=(256, 256))
        # Create a temporary dummy image for unit testing
        self.test_img_path = "test_dummy.png"
        dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 128
        cv2.imwrite(self.test_img_path, dummy_img)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_preprocessing_output_shape_and_range(self):
        # Run pipeline
        processed_img = self.preprocessor.pipeline(self.test_img_path)
        
        # Assertions
        self.assertEqual(processed_img.shape, (256, 256))
        self.assertTrue(np.max(processed_img) <= 1.0)
        self.assertTrue(np.min(processed_img) >= 0.0)
        self.assertEqual(processed_img.dtype, np.float32)

if __name__ == "__main__":
    unittest.main()