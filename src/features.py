import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        # Classical Vision Configuration (HOG)
        self.hog = cv2.HOGDescriptor(
            _winSize=(256, 256),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        # Deep Learning Backbone Configuration (ResNet18)
        weights = models.ResNet18_Weights.DEFAULT
        self.cnn = models.resnet18(weights=weights)
        # Remove the final classification layer to use it purely as a feature extractor
        self.cnn = torch.nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn.eval()
        
        self.cnn_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_classical_features(self, gray_image):
        # HOG requires 8-bit image format [0, 255]
        img_8bit = (gray_image * 255).astype(np.uint8)
        hog_features = self.hog.compute(img_8bit)
        return hog_features.flatten()

    def extract_deep_features(self, original_image_path):
        img = Image.open(original_image_path).convert('RGB')
        img_tensor = self.cnn_transforms(img).unsqueeze(0)
        
        with torch.no_grad():
            deep_features = self.cnn(img_tensor)
            deep_features = deep_features.squeeze().numpy()
            
        return deep_features

    def combine_features(self, classical_feats, deep_feats):
        # Concatenate handcrafted and learned representations into a single hybrid vector
        return np.concatenate((classical_feats, deep_feats))

if __name__ == "__main__":
    print("Feature extractor module initialized successfully.")