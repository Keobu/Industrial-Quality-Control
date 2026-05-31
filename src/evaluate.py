import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class PerformanceEvaluator:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', pos_label=1),
            'recall': recall_score(y_true, y_pred, average='binary', pos_label=1),
            'f1_score': f1_score(y_true, y_pred, average='binary', pos_label=1)
        }
        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Defective'], 
                    yticklabels=['Normal', 'Defective'])
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Quality Control - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix chart saved to {output_path}")

if __name__ == "__main__":
    print("Evaluation module initialized.")