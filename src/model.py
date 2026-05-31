from sklearn.svm import SVC
import joblib
import os

class DefectClassifier:
    def __init__(self, kernel='linear', C=1.0):
        # Initialize SVM classifier to handle the hybrid feature vector
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved successfully to {filepath}")

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        self.model = joblib.load(filepath)
        print(f"Model loaded successfully from {filepath}")

if __name__ == "__main__":
    print("Core logic classifier module initialized.")