"""
Decision Tree Analysis: Entropy vs GINI Index
Research on Banknote Authentication and Pima Indians Diabetes Datasets

Based on:
- Analysis of Depth of Entropy and GINI Index Based Decision Trees for Predicting Diabetes
- Entropy-Based Decision Trees (ID3, C4.5, C5.0)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Import training module
from .decision_tree_training import DecisionTreeTrainer

# Import visualization module
from .decision_tree_visualization import plot_analysis
from helpers.classification_metrics import ClassificationMetrics
from helpers.data_visualization import plot_data_overview


class DecisionTreeAnalyzer:
    """Comprehensive decision tree analysis with entropy and GINI index"""
    
    def __init__(self, dataset_name, X, y, feature_names, X_train, X_test, y_train, y_test):
        self.dataset_name = dataset_name
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.trainer = DecisionTreeTrainer(X_train, X_test, y_train, y_test)
        
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
    
    def train_decision_trees(self, max_depths=None):
        """Train decision trees using the training module"""
        self.results = self.trainer.train_decision_trees(max_depths=max_depths)
    
    def analyze_depth_impact(self):
        """Analyze the impact of tree depth on performance"""
        print(f"\n{'='*60}")
        print("DEPTH IMPACT ANALYSIS")
        print(f"{'='*60}")
        
        depths = []
        train_accs_entropy = []
        test_accs_entropy = []
        train_accs_gini = []
        test_accs_gini = []
        depths_actual_entropy = []
        depths_actual_gini = []
        
        for key in sorted(self.results['entropy'].keys(), 
                         key=lambda x: float('inf') if 'unlimited' in x else int(x.split('_')[1])):
            if 'unlimited' in key:
                depth_val = 999  # For sorting
            else:
                depth_val = int(key.split('_')[1])
            
            depths.append(depth_val)
            
            # Entropy results
            ent_result = self.results['entropy'][key]
            train_accs_entropy.append(ent_result['train_accuracy'])
            test_accs_entropy.append(ent_result['test_accuracy'])
            depths_actual_entropy.append(ent_result['depth'])
            
            # GINI results
            gini_result = self.results['gini'][key]
            train_accs_gini.append(gini_result['train_accuracy'])
            test_accs_gini.append(gini_result['test_accuracy'])
            depths_actual_gini.append(gini_result['depth'])
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Max_Depth': depths,
            'Entropy_Train_Acc': train_accs_entropy,
            'Entropy_Test_Acc': test_accs_entropy,
            'GINI_Train_Acc': train_accs_gini,
            'GINI_Test_Acc': test_accs_gini,
            'Entropy_Actual_Depth': depths_actual_entropy,
            'GINI_Actual_Depth': depths_actual_gini
        })
        
        print("\nDepth vs Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_analysis(self, comparison_df):
        """Create visualizations using the visualization module"""
        return plot_analysis(
            dataset_name=self.dataset_name,
            results=self.results,
            y_test=self.y_test,
            comparison_df=comparison_df
        )
    
    def display_classification_metrics(self, criterion='entropy', depth_key=None, comparison_df=None):
        """
        Display detailed classification metrics for a specific model
        
        Parameters:
        -----------
        criterion : str
            'entropy' or 'gini'
        depth_key : str, optional
            Depth key (e.g., 'depth_5', 'unlimited'). If None, uses best model
        comparison_df : pandas.DataFrame, optional
            Pre-computed comparison DataFrame to avoid recomputation
        """
        if depth_key is None:
            # Find best model
            if comparison_df is None:
                # Compute comparison without printing
                depths = []
                for key in sorted(self.results['entropy'].keys(), 
                                 key=lambda x: float('inf') if 'unlimited' in x else int(x.split('_')[1])):
                    if 'unlimited' in key:
                        depth_val = 999
                    else:
                        depth_val = int(key.split('_')[1])
                    depths.append(depth_val)
                
                comparison_df = pd.DataFrame({
                    'Max_Depth': depths,
                    'Entropy_Test_Acc': [self.results['entropy'][k]['test_accuracy'] 
                                        for k in sorted(self.results['entropy'].keys(), 
                                                       key=lambda x: float('inf') if 'unlimited' in x else int(x.split('_')[1]))],
                    'GINI_Test_Acc': [self.results['gini'][k]['test_accuracy'] 
                                     for k in sorted(self.results['gini'].keys(), 
                                                    key=lambda x: float('inf') if 'unlimited' in x else int(x.split('_')[1]))]
                })
            
            if criterion == 'entropy':
                best_idx = comparison_df['Entropy_Test_Acc'].idxmax()
                best_key = list(self.results['entropy'].keys())[best_idx]
            else:
                best_idx = comparison_df['GINI_Test_Acc'].idxmax()
                best_key = list(self.results['gini'].keys())[best_idx]
            depth_key = best_key
        
        result = self.results[criterion][depth_key]
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
    
    def predict_sample(self, sample_data, criterion='entropy', depth_key=None, comparison_df=None):
        """
        Make predictions on sample input data
        
        Parameters:
        -----------
        sample_data : array-like or dict
            Sample data to predict. Can be:
            - Array/list of feature values
            - Dictionary with feature names as keys
        criterion : str
            'entropy' or 'gini'
        depth_key : str, optional
            Depth key. If None, uses best model
        comparison_df : pandas.DataFrame, optional
            Pre-computed comparison DataFrame to avoid recomputation
            
        Returns:
        --------
        dict : Prediction results
        """
        # Find best model if depth_key not specified
        if depth_key is None:
            if comparison_df is None:
                # Compute comparison without printing
                depths = []
                for key in sorted(self.results['entropy'].keys(), 
                                 key=lambda x: float('inf') if 'unlimited' in x else int(x.split('_')[1])):
                    if 'unlimited' in key:
                        depth_val = 999
                    else:
                        depth_val = int(key.split('_')[1])
                    depths.append(depth_val)
                
                comparison_df = pd.DataFrame({
                    'Max_Depth': depths,
                    'Entropy_Test_Acc': [self.results['entropy'][k]['test_accuracy'] 
                                        for k in sorted(self.results['entropy'].keys(), 
                                                       key=lambda x: float('inf') if 'unlimited' in x else int(x.split('_')[1]))],
                    'GINI_Test_Acc': [self.results['gini'][k]['test_accuracy'] 
                                     for k in sorted(self.results['gini'].keys(), 
                                                    key=lambda x: float('inf') if 'unlimited' in x else int(x.split('_')[1]))]
                })
            
            if criterion == 'entropy':
                best_idx = comparison_df['Entropy_Test_Acc'].idxmax()
                best_key = list(self.results['entropy'].keys())[best_idx]
            else:
                best_idx = comparison_df['GINI_Test_Acc'].idxmax()
                best_key = list(self.results['gini'].keys())[best_idx]
            depth_key = best_key
        
        model = self.results[criterion][depth_key]['model']
        
        # Convert sample data to array
        if isinstance(sample_data, dict):
            # Convert dict to array in correct feature order
            sample_array = np.array([sample_data.get(feat, 0) for feat in self.feature_names])
        else:
            sample_array = np.array(sample_data)
        
        # Reshape if needed
        if sample_array.ndim == 1:
            sample_array = sample_array.reshape(1, -1)
        
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
                'criterion': criterion,
                'depth_key': depth_key,
                'accuracy': self.results[criterion][depth_key]['test_accuracy']
            }
        }
        
        return result
    
    def generate_report(self, comparison_df):
        """Generate comprehensive research report"""
        report = f"""
{'='*80}
RESEARCH REPORT: Decision Tree Analysis - Entropy vs GINI Index
Dataset: {self.dataset_name}
{'='*80}

1. DATASET OVERVIEW
-------------------
- Total Samples: {len(self.X)}
- Features: {len(self.feature_names)}
- Training Samples: {len(self.X_train)}
- Test Samples: {len(self.X_test)}
- Classes: {np.unique(self.y)}

2. METHODOLOGY
-------------
This analysis compares Decision Tree algorithms using two splitting criteria:
- Entropy (Information Gain): Based on Shannon's entropy
- GINI Index: Measures impurity

Both criteria were tested with varying maximum depths to analyze:
- Impact of tree depth on performance
- Overfitting behavior
- Optimal depth for each criterion

3. KEY FINDINGS
---------------

3.1 Best Performance
"""
        # Find best models
        best_entropy_idx = comparison_df['Entropy_Test_Acc'].idxmax()
        best_gini_idx = comparison_df['GINI_Test_Acc'].idxmax()
        
        best_entropy_key = list(self.results['entropy'].keys())[best_entropy_idx]
        best_gini_key = list(self.results['gini'].keys())[best_gini_idx]
        
        ent_best = self.results['entropy'][best_entropy_key]
        gini_best = self.results['gini'][best_gini_key]
        
        report += f"""
Entropy (Best):
  - Max Depth Limit: {best_entropy_key}
  - Actual Depth: {ent_best['depth']}
  - Test Accuracy: {ent_best['test_accuracy']:.4f}
  - Precision: {ent_best['precision']:.4f}
  - Recall: {ent_best['recall']:.4f}
  - F1-Score: {ent_best['f1_score']:.4f}
  - Nodes: {ent_best['n_nodes']}, Leaves: {ent_best['n_leaves']}

GINI Index (Best):
  - Max Depth Limit: {best_gini_key}
  - Actual Depth: {gini_best['depth']}
  - Test Accuracy: {gini_best['test_accuracy']:.4f}
  - Precision: {gini_best['precision']:.4f}
  - Recall: {gini_best['recall']:.4f}
  - F1-Score: {gini_best['f1_score']:.4f}
  - Nodes: {gini_best['n_nodes']}, Leaves: {gini_best['n_leaves']}

3.2 Depth Analysis
"""
        report += f"""
The analysis shows how tree depth affects model performance:

{comparison_df.to_string(index=False)}

Key Observations:
- Optimal depth varies between Entropy and GINI
- Overfitting increases with depth (Train Acc - Test Acc gap)
- Both criteria show similar patterns but may differ in optimal depth
"""
        
        # Determine winner
        if ent_best['test_accuracy'] > gini_best['test_accuracy']:
            winner = "Entropy"
            diff = ent_best['test_accuracy'] - gini_best['test_accuracy']
        else:
            winner = "GINI Index"
            diff = gini_best['test_accuracy'] - ent_best['test_accuracy']
        
        report += f"""

3.3 Comparison Summary
----------------------
Winner: {winner} (Difference: {diff:.4f})

4. CONCLUSIONS
--------------
Based on the analysis of {self.dataset_name}:

1. Both Entropy and GINI Index are effective splitting criteria
2. Optimal tree depth is critical for preventing overfitting
3. The best criterion may depend on the specific dataset characteristics
4. Deeper trees tend to overfit, showing higher train accuracy but lower test accuracy

5. RECOMMENDATIONS
------------------
1. Use cross-validation to determine optimal depth
2. Monitor train-test accuracy gap to detect overfitting
3. Consider pruning techniques for deeper trees
4. Test both criteria and select based on validation performance

{'='*80}
Report generated successfully.
{'='*80}
"""
        
        # Create report folder if it doesn't exist
        report_dir = 'report'
        os.makedirs(report_dir, exist_ok=True)
        
        # Save report
        filename = f"{self.dataset_name.replace(' ', '_')}_report.txt"
        filepath = os.path.join(report_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"\nReport saved as: {filepath}")
        
        return report
