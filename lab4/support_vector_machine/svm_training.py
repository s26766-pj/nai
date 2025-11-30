"""
SVM Training: Support Vector Machine with Different Kernels
Handles training of SVM models with various kernel functions and parameters
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SVMTrainer:
    """Handles training of SVM models with different kernels and parameters"""
    
    def __init__(self, X_train, X_test, y_train, y_test, scale_features=True):
        """
        Initialize SVM trainer
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Test features
        y_train : array-like
            Training labels
        y_test : array-like
            Test labels
        scale_features : bool, optional
            Whether to scale features (recommended for SVM). Default: True
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scale_features = scale_features
        
        # Scale features if requested
        if scale_features:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_test_scaled = self.scaler.transform(X_test)
        else:
            self.scaler = None
            self.X_train_scaled = X_train
            self.X_test_scaled = X_test
    
    def train_svm_models(self, kernels_config=None, random_state=42):
        """
        Train SVM models with different kernels and parameters
        
        Parameters:
        -----------
        kernels_config : dict, optional
            Dictionary with kernel configurations. If None, uses default configurations.
            Format: {
                'kernel_name': [
                    {'C': value, 'gamma': value, ...},
                    ...
                ]
            }
        random_state : int, optional
            Random state for reproducibility. Default: 42
            
        Returns:
        --------
        dict : Dictionary containing results for each kernel and configuration
            Structure: {kernel_name: {config_key: {metrics...}}}
        """
        if kernels_config is None:
            kernels_config = self._get_default_kernels_config()
        
        results = {}
        
        for kernel_name, configs in kernels_config.items():
            print(f"\n{'='*60}")
            print(f"Training SVM with {kernel_name.upper()} kernel")
            print(f"{'='*60}")
            
            results[kernel_name] = {}
            
            for i, config in enumerate(configs):
                config_key = f"config_{i+1}"
                
                # Create SVM model
                svm = SVC(
                    kernel=kernel_name,
                    random_state=random_state,
                    probability=True,  # Enable probability estimates
                    **config
                )
                
                # Train model
                svm.fit(self.X_train_scaled, self.y_train)
                
                # Predictions
                y_train_pred = svm.predict(self.X_train_scaled)
                y_test_pred = svm.predict(self.X_test_scaled)
                
                # Metrics
                train_acc = accuracy_score(self.y_train, y_train_pred)
                test_acc = accuracy_score(self.y_test, y_test_pred)
                precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
                
                # Support vectors info
                n_support_vectors = svm.n_support_
                total_support_vectors = len(svm.support_)
                
                # Store results
                results[kernel_name][config_key] = {
                    'model': svm,
                    'config': config,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'n_support_vectors': total_support_vectors,
                    'n_support_vectors_per_class': n_support_vectors,
                    'y_test_pred': y_test_pred
                }
                
                # Display config and results
                config_str = ', '.join([f"{k}={v}" for k, v in config.items()])
                print(f"\nConfiguration {i+1}: {config_str}")
                print(f"  Support Vectors: {total_support_vectors}")
                print(f"  Train Accuracy: {train_acc:.4f}")
                print(f"  Test Accuracy: {test_acc:.4f}")
                print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return results
    
    def _get_default_kernels_config(self):
        """Get default kernel configurations for testing"""
        return {
            'linear': [
                {'C': 0.1},
                {'C': 1.0},
                {'C': 10.0},
                {'C': 100.0}
            ],
            'poly': [
                {'C': 1.0, 'gamma': 'scale', 'degree': 2},
                {'C': 1.0, 'gamma': 'scale', 'degree': 3},
                {'C': 1.0, 'gamma': 'auto', 'degree': 3},
                {'C': 10.0, 'gamma': 'scale', 'degree': 3}
            ],
            'rbf': [
                {'C': 0.1, 'gamma': 'scale'},
                {'C': 1.0, 'gamma': 'scale'},
                {'C': 1.0, 'gamma': 'auto'},
                {'C': 10.0, 'gamma': 'scale'},
                {'C': 1.0, 'gamma': 0.01},
                {'C': 1.0, 'gamma': 0.1}
            ],
            'sigmoid': [
                {'C': 1.0, 'gamma': 'scale'},
                {'C': 1.0, 'gamma': 'auto'},
                {'C': 10.0, 'gamma': 'scale'},
                {'C': 1.0, 'gamma': 0.01}
            ]
        }


def train_svm_models(X_train, X_test, y_train, y_test, kernels_config=None, scale_features=True, random_state=42):
    """
    Convenience function to train SVM models
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    kernels_config : dict, optional
        Dictionary with kernel configurations
    scale_features : bool, optional
        Whether to scale features. Default: True
    random_state : int, optional
        Random state for reproducibility. Default: 42
        
    Returns:
    --------
    dict : Dictionary containing results for each kernel and configuration
    """
    trainer = SVMTrainer(X_train, X_test, y_train, y_test, scale_features=scale_features)
    return trainer.train_svm_models(kernels_config=kernels_config, random_state=random_state)
