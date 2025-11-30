"""
Classification Metrics: Detailed metrics display for decision trees
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class ClassificationMetrics:
    """
    Calculate and display comprehensive classification performance metrics.
    
    This class provides detailed evaluation metrics for classification models,
    including overall metrics (accuracy, precision, recall, F1-score), per-class
    metrics, confusion matrix, and a detailed classification report.
    """
    
    def __init__(self, y_true, y_pred, class_names=None):
        """
        Initialize the ClassificationMetrics calculator with true and predicted labels.
        
        Parameters:
        -----------
        y_true : array-like of shape (n_samples,)
            True/actual class labels from the dataset
        y_pred : array-like of shape (n_samples,)
            Predicted class labels from the model
        class_names : list, optional
            Human-readable names for each class (e.g., ['No Diabetes', 'Diabetes']).
            If None, automatically generates names like ['Class 0', 'Class 1', ...]
        """
        # Convert to numpy arrays for consistent processing
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        # Generate class names if not provided
        # Uses unique values from true labels to ensure all classes are represented
        self.class_names = class_names if class_names else [f'Class {i}' for i in np.unique(y_true)]
    
    def display_metrics(self):
        """
        Calculate and display comprehensive classification metrics.
        
        This method computes and prints:
        - Overall metrics (accuracy, precision, recall, F1-score)
        - Per-class metrics for each class
        - Confusion matrix showing prediction vs actual counts
        - Detailed classification report with support counts
        
        Returns:
        --------
        dict : Dictionary containing all calculated metrics:
            - 'accuracy': float
                Overall accuracy score
            - 'precision': float
                Weighted average precision
            - 'recall': float
                Weighted average recall
            - 'f1_score': float
                Weighted average F1-score
            - 'confusion_matrix': numpy.ndarray
                2D array showing true vs predicted class counts
            - 'per_class_metrics': pandas.DataFrame
                DataFrame with precision, recall, F1-score for each class
        """
        # Display header
        print("\n" + "="*80)
        print("CLASSIFICATION METRICS")
        print("="*80)
        
        # Calculate overall metrics
        # Accuracy: proportion of correct predictions
        accuracy = accuracy_score(self.y_true, self.y_pred)
        # Precision: proportion of positive predictions that are correct
        # 'weighted': accounts for class imbalance by weighting by support
        # zero_division=0: returns 0 if division by zero occurs
        precision = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        # Recall: proportion of actual positives that were correctly identified
        recall = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        # F1-score: harmonic mean of precision and recall (balances both metrics)
        f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        # Display overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Calculate per-class metrics
        # average=None returns metrics for each class separately
        print(f"\nPer-Class Metrics:")
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        # Create a DataFrame for better visualization of per-class metrics
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class
        })
        print(metrics_df.to_string(index=False))
        
        # Calculate and display confusion matrix
        # Confusion matrix shows: rows = true labels, columns = predicted labels
        cm = confusion_matrix(self.y_true, self.y_pred)
        print(f"\nConfusion Matrix:")
        # Print header row with predicted class names
        print("                Predicted")
        print("              ", end="")
        # Truncate class names to 8 characters for formatting
        for i, name in enumerate(self.class_names):
            print(f"  {name[:8]:<8}", end="")
        print()
        # Print each row with true class name and prediction counts
        for i, (true_name, row) in enumerate(zip(self.class_names, cm)):
            print(f"True {true_name[:8]:<8}", end="")
            for val in row:
                print(f"  {val:8d}", end="")
            print()
        
        # Display detailed classification report
        # Includes precision, recall, F1-score, and support for each class
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_true, self.y_pred, 
                                    target_names=self.class_names, zero_division=0))
        
        # Return all metrics as a dictionary for programmatic access
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'per_class_metrics': metrics_df
        }


def display_classification_metrics(y_true, y_pred, class_names=None):
    """
    Convenience function to display classification metrics without instantiating the class.
    
    This is a wrapper function that creates a ClassificationMetrics instance and
    immediately displays the metrics. Useful for quick evaluation without needing
    to manage the metrics object.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True/actual class labels from the dataset
    y_pred : array-like of shape (n_samples,)
        Predicted class labels from the model
    class_names : list, optional
        Human-readable names for each class. If None, auto-generates names.
        
    Returns:
    --------
    dict : Dictionary containing all calculated metrics.
        Same structure as ClassificationMetrics.display_metrics()
    """
    # Create metrics calculator instance
    metrics = ClassificationMetrics(y_true, y_pred, class_names)
    # Calculate and display metrics, return results
    return metrics.display_metrics()
