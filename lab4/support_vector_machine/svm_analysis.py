"""
SVM Analysis: Support Vector Machine Analysis
Analyzes SVM models with different kernels and parameters
"""

import pandas as pd
import numpy as np
import os

# Import training module
from .svm_training import SVMTrainer

# Import visualization module
from .svm_visualization import plot_svm_analysis

# Import metrics module
from helpers.classification_metrics import ClassificationMetrics

# Import data visualization
from helpers.data_visualization import plot_data_overview


class SVMAnalyzer:
    """Comprehensive SVM analysis with different kernels"""
    
    def __init__(self, dataset_name, X, y, feature_names, X_train, X_test, y_train, y_test, scale_features=True):
        self.dataset_name = dataset_name
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scale_features = scale_features
        self.results = {}
        self.trainer = SVMTrainer(X_train, X_test, y_train, y_test, scale_features=scale_features)
        
    def load_and_prepare_data(self):
        """Display data preparation summary"""
        print(f"\n{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"Shape: {self.X.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: {np.unique(self.y)}")
        print(f"Class distribution:\n{pd.Series(self.y).value_counts()}")
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Feature scaling: {'Enabled' if self.scale_features else 'Disabled'}")
    
    def train_svm_models(self, kernels_config=None):
        """Train SVM models using the training module"""
        self.results = self.trainer.train_svm_models(kernels_config=kernels_config)
    
    def analyze_kernel_performance(self):
        """Analyze the performance of different kernels"""
        print(f"\n{'='*60}")
        print("KERNEL PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        kernel_summary = []
        
        for kernel_name, configs in self.results.items():
            for config_key, result in configs.items():
                config_str = ', '.join([f"{k}={v}" for k, v in result['config'].items()])
                kernel_summary.append({
                    'Kernel': kernel_name,
                    'Configuration': config_str,
                    'Test_Accuracy': result['test_accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1_score'],
                    'Support_Vectors': result['n_support_vectors']
                })
        
        summary_df = pd.DataFrame(kernel_summary)
        summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
        
        print("\nKernel Performance Summary (sorted by Test Accuracy):")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def plot_analysis(self, summary_df):
        """Create visualizations using the visualization module"""
        return plot_svm_analysis(
            dataset_name=self.dataset_name,
            results=self.results,
            y_test=self.y_test,
            summary_df=summary_df
        )
    
    def plot_decision_boundary(self, kernel_name='rbf', config_key=None, plot_3d=True):
        """
        Visualize the SVM decision boundary (hyperplane) in 2D and/or 3D space.
        
        This method creates visualizations showing how the SVM separates classes
        using PCA to project high-dimensional data to 2D and/or 3D. Shows the decision
        boundary, support vectors, and data points.
        
        Parameters:
        -----------
        kernel_name : str, optional
            Kernel to visualize ('linear', 'poly', 'rbf', 'sigmoid').
            Default: 'rbf'
        config_key : str, optional
            Configuration key. If None, uses best configuration for that kernel.
        plot_3d : bool, optional
            If True, creates both 2D and 3D visualizations. If False, only 2D.
            Default: True
            
        Returns:
        --------
        str or tuple : 
            If plot_3d=False: Filepath of 2D visualization
            If plot_3d=True: Tuple of (2D_filepath, 3D_filepath)
        """
        from .svm_visualization import SVMVisualizer
        
        # Create visualizer instance
        visualizer = SVMVisualizer(
            dataset_name=self.dataset_name,
            results=self.results,
            y_test=self.y_test,
            summary_df=None  # Not needed for decision boundary plot
        )
        
        # Get the scaler used during training
        scaler = self.trainer.scaler if self.scale_features else None
        
        # Generate and return the decision boundary plot(s)
        return visualizer.plot_decision_boundary(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            kernel_name=kernel_name,
            config_key=config_key,
            scaler=scaler,
            plot_3d=plot_3d
        )
    
    def display_classification_metrics(self, kernel_name, config_key=None):
        """
        Display detailed classification metrics for a specific model
        
        Parameters:
        -----------
        kernel_name : str
            Name of the kernel ('linear', 'poly', 'rbf', 'sigmoid')
        config_key : str, optional
            Configuration key. If None, uses best configuration for that kernel
        """
        if config_key is None:
            # Find best configuration for this kernel
            best_acc = 0
            best_key = None
            for key, result in self.results[kernel_name].items():
                if result['test_accuracy'] > best_acc:
                    best_acc = result['test_accuracy']
                    best_key = key
            config_key = best_key
        
        result = self.results[kernel_name][config_key]
        y_pred = result['y_test_pred']
        
        # Determine class names
        unique_classes = np.unique(self.y)
        if len(unique_classes) == 2:
            class_names = ['Class 0', 'Class 1']
        else:
            class_names = [f'Class {i}' for i in unique_classes]
        
        metrics = ClassificationMetrics(self.y_test, y_pred, class_names)
        return metrics.display_metrics()
    
    def visualize_sample_data(self, n_samples=100):
        """Visualize sample data from the dataset"""
        return plot_data_overview(
            dataset_name=self.dataset_name,
            X=self.X,
            y=self.y,
            feature_names=self.feature_names,
            n_samples=n_samples
        )
    
    def predict_sample(self, sample_data, kernel_name='rbf', config_key=None):
        """
        Make predictions on sample input data
        
        Parameters:
        -----------
        sample_data : array-like or dict
            Sample data to predict. Can be:
            - Array/list of feature values
            - Dictionary with feature names as keys
        kernel_name : str
            Kernel to use ('linear', 'poly', 'rbf', 'sigmoid')
        config_key : str, optional
            Configuration key. If None, uses best configuration for that kernel
            
        Returns:
        --------
        dict : Prediction results
        """
        # Find best configuration if not specified
        if config_key is None:
            best_acc = 0
            best_key = None
            for key, result in self.results[kernel_name].items():
                if result['test_accuracy'] > best_acc:
                    best_acc = result['test_accuracy']
                    best_key = key
            config_key = best_key
        
        model = self.results[kernel_name][config_key]['model']
        
        # Convert sample data to array
        if isinstance(sample_data, dict):
            sample_array = np.array([sample_data.get(feat, 0) for feat in self.feature_names])
        else:
            sample_array = np.array(sample_data)
        
        # Reshape if needed
        if sample_array.ndim == 1:
            sample_array = sample_array.reshape(1, -1)
        
        # Scale if features were scaled during training
        if self.scale_features:
            sample_array = self.trainer.scaler.transform(sample_array)
        
        # Make prediction
        prediction = model.predict(sample_array)[0]
        prediction_proba = model.predict_proba(sample_array)[0]
        
        # Format results
        unique_classes = np.unique(self.y)
        class_names = [f'Class {i}' for i in unique_classes]
        
        result = {
            'predicted_class': int(prediction),
            'class_probabilities': {
                class_names[i]: float(prob) for i, prob in enumerate(prediction_proba)
            },
            'model_info': {
                'kernel': kernel_name,
                'config_key': config_key,
                'config': self.results[kernel_name][config_key]['config'],
                'accuracy': self.results[kernel_name][config_key]['test_accuracy']
            }
        }
        
        return result
    
    def generate_kernel_summary(self, summary_df):
        """Generate summary report about kernel functions"""
        report = f"""
{'='*80}
SVM KERNEL FUNCTIONS ANALYSIS SUMMARY
Dataset: {self.dataset_name}
{'='*80}

1. OVERVIEW
-----------
This analysis evaluates Support Vector Machine (SVM) classifiers using
different kernel functions and their parameters on the {self.dataset_name} dataset.

2. KERNEL FUNCTIONS TESTED
---------------------------
"""
        # Get best result for each kernel
        kernel_best = {}
        for kernel_name in self.results.keys():
            best_acc = 0
            best_config = None
            for config_key, result in self.results[kernel_name].items():
                if result['test_accuracy'] > best_acc:
                    best_acc = result['test_accuracy']
                    best_config = (config_key, result)
            kernel_best[kernel_name] = best_config
        
        for kernel_name, (config_key, result) in kernel_best.items():
            config_str = ', '.join([f"{k}={v}" for k, v in result['config'].items()])
            report += f"""
{kernel_name.upper()} Kernel:
  Best Configuration: {config_str}
  Test Accuracy: {result['test_accuracy']:.4f}
  Precision: {result['precision']:.4f}
  Recall: {result['recall']:.4f}
  F1-Score: {result['f1_score']:.4f}
  Support Vectors: {result['n_support_vectors']}
"""
        
        report += f"""
3. KERNEL COMPARISON TABLE
---------------------------
{summary_df.to_string(index=False)}

4. KERNEL FUNCTION DESCRIPTIONS
-------------------------------

4.1 Linear Kernel
-----------------
Formula: K(x, y) = x^T * y
Parameters: C (regularization parameter)

Characteristics:
- Simplest kernel function
- Creates linear decision boundaries
- Fast training and prediction
- Good for linearly separable data
- C parameter controls margin width vs. classification errors

Effect on Results:
"""
        linear_best = kernel_best.get('linear', (None, {}))
        if linear_best[1]:
            report += f"- Best Linear Accuracy: {linear_best[1]['test_accuracy']:.4f}\n"
            report += f"- Lower C values create wider margins but may misclassify more points\n"
            report += f"- Higher C values create narrower margins but fit training data better\n"
        
        report += f"""
4.2 Polynomial Kernel
---------------------
Formula: K(x, y) = (gamma * x^T * y + coef0)^degree
Parameters: C, gamma, degree, coef0

Characteristics:
- Can model non-linear relationships
- Degree controls complexity (higher = more complex)
- Computationally expensive for high degrees
- Can overfit with high degree values

Effect on Results:
"""
        poly_best = kernel_best.get('poly', (None, {}))
        if poly_best[1]:
            report += f"- Best Polynomial Accuracy: {poly_best[1]['test_accuracy']:.4f}\n"
            report += f"- Degree 2: Quadratic decision boundaries\n"
            report += f"- Degree 3: Cubic decision boundaries (most common)\n"
            report += f"- Higher degrees can capture more complex patterns but risk overfitting\n"
        
        report += f"""
4.3 RBF (Radial Basis Function) Kernel
----------------------------------------
Formula: K(x, y) = exp(-gamma * ||x - y||^2)
Parameters: C, gamma

Characteristics:
- Most popular kernel for non-linear problems
- Creates smooth, curved decision boundaries
- gamma controls influence of individual training examples
- 'scale': gamma = 1 / (n_features * X.var())
- 'auto': gamma = 1 / n_features
- Small gamma: wider influence, smoother boundaries
- Large gamma: narrower influence, more complex boundaries

Effect on Results:
"""
        rbf_best = kernel_best.get('rbf', (None, {}))
        if rbf_best[1]:
            report += f"- Best RBF Accuracy: {rbf_best[1]['test_accuracy']:.4f}\n"
            report += f"- Generally performs well on non-linear datasets\n"
            report += f"- gamma='scale' often works better than 'auto'\n"
            report += f"- C parameter balances margin width and classification errors\n"
        
        report += f"""
4.4 Sigmoid Kernel
------------------
Formula: K(x, y) = tanh(gamma * x^T * y + coef0)
Parameters: C, gamma, coef0

Characteristics:
- Similar to neural network activation function
- Less commonly used than RBF
- Can be sensitive to parameter choices
- May not be positive definite for all parameters

Effect on Results:
"""
        sigmoid_best = kernel_best.get('sigmoid', (None, {}))
        if sigmoid_best[1]:
            report += f"- Best Sigmoid Accuracy: {sigmoid_best[1]['test_accuracy']:.4f}\n"
            report += f"- Often performs worse than RBF for most datasets\n"
            report += f"- Requires careful parameter tuning\n"
        
        report += f"""
5. KEY FINDINGS
---------------

5.1 Best Performing Kernel
"""
        # Find overall best
        overall_best = max(kernel_best.items(), key=lambda x: x[1][1]['test_accuracy'] if x[1][1] else 0)
        report += f"""
Overall Best: {overall_best[0].upper()} Kernel
  Accuracy: {overall_best[1][1]['test_accuracy']:.4f}
  Configuration: {', '.join([f"{k}={v}" for k, v in overall_best[1][1]['config'].items()])}
"""
        
        report += f"""
5.2 Parameter Effects
---------------------
C Parameter (Regularization):
- Low C: Wider margin, more misclassifications allowed, simpler model
- High C: Narrower margin, fewer misclassifications, more complex model
- Optimal C depends on dataset characteristics

Gamma Parameter (RBF, Poly, Sigmoid):
- Low gamma: Wider influence radius, smoother decision boundaries
- High gamma: Narrower influence radius, more complex boundaries
- 'scale' often better than 'auto' for feature-scaled data

Degree Parameter (Polynomial):
- Degree 2: Quadratic boundaries
- Degree 3: Cubic boundaries (most common)
- Higher degrees: More complex but risk overfitting

5.3 Support Vectors
------------------
"""
        for kernel_name, (config_key, result) in kernel_best.items():
            if result:
                report += f"- {kernel_name.upper()}: {result['n_support_vectors']} support vectors\n"
        
        report += f"""
Support vectors are the data points that define the decision boundary.
Fewer support vectors generally indicate a simpler, more generalizable model.

6. RECOMMENDATIONS
-----------------
"""
        if overall_best[0] == 'rbf':
            report += """
1. RBF kernel is recommended for this dataset
2. Start with C=1.0 and gamma='scale' as default
3. Tune C and gamma using grid search or cross-validation
4. Consider feature scaling (already applied in this analysis)
"""
        elif overall_best[0] == 'linear':
            report += """
1. Linear kernel works well, suggesting data may be nearly linearly separable
2. Try RBF kernel for potentially better non-linear performance
3. Tune C parameter for optimal margin width
"""
        else:
            report += f"""
1. {overall_best[0].upper()} kernel shows best performance
2. Consider trying RBF kernel with different gamma values
3. Feature scaling is important for kernel-based methods
4. Use cross-validation for parameter selection
"""
        
        report += f"""
7. CONCLUSION
-------------
Different kernel functions create different decision boundaries and have
varying computational costs. The choice of kernel and parameters significantly
affects classification performance. For this dataset, the {overall_best[0].upper()}
kernel with appropriate parameters provides the best results.

{'='*80}
"""
        
        # Save report
        report_dir = 'report'
        os.makedirs(report_dir, exist_ok=True)
        
        filename = f"{self.dataset_name.replace(' ', '_')}_SVM_Kernel_Summary.txt"
        filepath = os.path.join(report_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"\nKernel summary saved as: {filepath}")
        
        return report
