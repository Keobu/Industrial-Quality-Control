import os
import numpy as np
import cv2
from src.preprocessing import IndustrialPreprocessor
from src.features import FeatureExtractor
from src.model import DefectClassifier
from src.evaluate import PerformanceEvaluator

def create_mock_dataset():
    
    print("No data found. Generating a mock industrial dataset for testing...")
    os.makedirs("data/train/normal", exist_ok=True)
    os.makedirs("data/train/defective", exist_ok=True)
    
    for i in range(10):
        # Base image simulating a smooth metallic surface
        img_normal = np.ones((300, 300, 3), dtype=np.uint8) * 180
        cv2.imwrite(f"data/train/normal/sample_{i}.png", img_normal)
        
        # Defective image with a simulated dark scratch line
        img_defect = img_normal.copy()
        cv2.line(img_defect, (50, 50 + i*10), (250, 80 + i*10), (20, 20, 20), 2)
        cv2.imwrite(f"data/train/defective/sample_{i}.png", img_defect)

def load_and_process_dataset(preprocessor, extractor):
    
    X = []
    y = []
    
    categories = {'normal': 0, 'defective': 1}
    base_path = "data/train"
    
    for category, label in categories.items():
        folder_path = os.path.join(base_path, category)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                
                # Run Preprocessing Pipeline (OpenCV)
                processed_img = preprocessor.pipeline(img_path)
                
                # Extract Classical Features (HOG)
                classical_feats = extractor.extract_classical_features(processed_img)
                
                # Extract Deep Features (CNN Backbone)
                deep_feats = extractor.extract_deep_features(img_path)
                
                # Combine representations into a hybrid vector
                combined_vector = extractor.combine_features(classical_feats, deep_feats)
                
                X.append(combined_vector)
                y.append(label)
                
    return np.array(X), np.array(y)

def main():
    print("="*50)
    print("STARTING INDUSTRIAL QUALITY CONTROL PIPELINE")
    print("="*50)
    
    # Ensure data directory is not empty for the demonstration
    if not os.path.exists("data/train") or len(os.listdir("data/train")) == 0:
        create_mock_dataset()
        
    # Initialize pipeline modules
    print("\n[1/4] Initializing pipeline components...")
    preprocessor = IndustrialPreprocessor(target_size=(256, 256))
    extractor = FeatureExtractor()
    classifier = DefectClassifier(kernel='linear', C=1.0)
    
    # Process dataset and build feature matrices
    print("\n[2/4] Processing images and engineering hybrid features...")
    X, y = load_and_process_dataset(preprocessor, extractor)
    print(f"Dataset loaded. Feature matrix shape: {X.shape}")
    
    # Train the core logic classifier
    print("\n[3/4] Training the core classifier (SVM)...")
    classifier.train(X, y)
    classifier.save_model("models/best_svm_model.joblib")
    
    # Run evaluation on the dataset
    print("\n[4/4] Evaluating pipeline performance...")
    predictions = classifier.predict(X)
    
    metrics = PerformanceEvaluator.compute_metrics(y, predictions)
    print(f"-> Accuracy:  {metrics['accuracy']:.4f}")
    print(f"-> Precision: {metrics['precision']:.4f}")
    print(f"-> Recall:    {metrics['recall']:.4f}")
    print(f"-> F1-Score:  {metrics['f1_score']:.4f}")
    
    # Generate charts for the technical documentation
    PerformanceEvaluator.plot_confusion_matrix(y, predictions, "confusion_matrix.png")
    
    print("\n"+"="*50)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()