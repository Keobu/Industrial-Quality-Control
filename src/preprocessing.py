import cv2
import numpy as np

class IndustrialPreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        # Initialize CLAHE for local contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def resize(self, image):
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def to_grayscale(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def remove_noise(self, image, kernel_size=(5, 5)):
        # Gaussian blur for high-frequency noise reduction
        return cv2.GaussianBlur(image, kernel_size, 0)

    def enhance_contrast(self, image):
        # Apply CLAHE to highlight surface defects like cracks or scratches
        return self.clahe.apply(image)

    def normalize(self, image):
        # Normalize pixel values to [0, 1] range
        return image.astype(np.float32) / 255.0

    def pipeline(self, image_path):
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        img_resized = self.resize(img)
        img_gray = self.to_grayscale(img_resized)
        img_denoised = self.remove_noise(img_gray)
        img_enhanced = self.enhance_contrast(img_denoised)
        img_normalized = self.normalize(img_enhanced)
        
        return img_normalized

if __name__ == "__main__":
    print("Preprocessor module initialized successfully.")