import unittest
import numpy as np
import os
import cv2
from src.preprocessing import IndustrialPreprocessor
from src.features import FeatureExtractor

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.preprocessor = IndustrialPreprocessor(target_size=(256, 256))
        self.extractor = FeatureExtractor()
        self.test_img_path = "test_feat_dummy.png"
        dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 180
        cv2.imwrite(self.test_img_path, dummy_img)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_feature_fusion_consistency(self):
        
        processed_img = self.preprocessor.pipeline(self.test_img_path)
        
        classical_feats = self.extractor.extract_classical_features(processed_img)
        deep_feats = self.extractor.extract_deep_features(self.test_img_path)
        combined = self.extractor.combine_features(classical_feats, deep_feats)
        
        expected_length = len(classical_feats) + len(deep_feats)
        self.assertEqual(len(combined), expected_length)
        self.assertTrue(isinstance(combined, np.ndarray))

if __name__ == "__main__":
    unittest.main()