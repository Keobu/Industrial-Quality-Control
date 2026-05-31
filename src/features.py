import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureExtractor:
    def __init__(self):
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
       
        img_8bit = (gray_image * 255).astype(np.uint8)
        h, w = img_8bit.shape
        mid_h, mid_w = h // 2, w // 2
        
        # Divide image into 4 spatial quadrants
        quadrants = [
            img_8bit[0:mid_h, 0:mid_w],   # Top-Left
            img_8bit[0:mid_h, mid_w:w],   # Top-Right
            img_8bit[mid_h:h, 0:mid_w],   # Bottom-Left
            img_8bit[mid_h:h, mid_w:w]    # Bottom-Right
        ]
        
        grid_features = []
        # Local HOG descriptor tailored for 128x128 quadrants
        local_hog = cv2.HOGDescriptor(_winSize=(128,128), _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)
        
        for quad in quadrants:
            quad_resized = cv2.resize(quad, (128, 128))
            feats = local_hog.compute(quad_resized)
            grid_features.append(feats.flatten())
            
        return np.concatenate(grid_features)

    def extract_deep_features(self, original_image_path):
        img = Image.open(original_image_path).convert('RGB')
        img_tensor = self.cnn_transforms(img).unsqueeze(0)
        
        with torch.no_grad():
            deep_features = self.cnn(img_tensor)
            deep_features = deep_features.squeeze().numpy()
            
        return deep_features

    def combine_features(self, classical_feats, deep_feats):
        return np.concatenate((classical_feats, deep_feats))

if __name__ == "__main__":
    print("Feature extractor module initialized successfully.")