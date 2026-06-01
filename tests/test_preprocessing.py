import unittest
import numpy as np
import os
import cv2
from src.preprocessing import IndustrialPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = IndustrialPreprocessor(target_size=(256, 256))
        self.test_img_path = "test_prep_dummy.png"
        # Create a standard gray dummy image
        dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 180
        cv2.imwrite(self.test_img_path, dummy_img)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_output_dimensions_and_type(self):

        processed_img = self.preprocessor.pipeline(self.test_img_path)
        self.assertEqual(processed_img.shape, (256, 256))
        self.assertEqual(processed_img.dtype, np.float32)

    def test_normalization_ranges(self):

        processed_img = self.preprocessor.pipeline(self.test_img_path)
        self.assertTrue(np.max(processed_img) <= 1.0)
        self.assertTrue(np.min(processed_img) >= 0.0)

    def test_missing_file_exception(self):

        with self.assertRaises(FileNotFoundError):
            self.preprocessor.pipeline("this_file_does_not_exist.jpg")

if __name__ == "__main__":
    unittest.main()