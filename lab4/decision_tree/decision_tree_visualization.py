"""
Decision Tree Visualization: Entropy vs GINI Index
Handles all plotting and visualization functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DecisionTreeVisualizer:
    """Handles visualization of decision tree analysis results"""
    
    def __init__(self, dataset_name, results, y_test, comparison_df):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        results : dict
            Dictionary containing results from DecisionTreeAnalyzer
        y_test : array-like
            True test labels
        comparison_df : pandas.DataFrame
            DataFrame with depth comparison results
        """
        self.dataset_name = dataset_name
        self.results = results
        self.y_test = y_test
        self.comparison_df = comparison_df
    
    def plot_analysis(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Depth vs Accuracy
        ax1 = plt.subplot(2, 3, 1)
        max_depths = self.comparison_df['Max_Depth'].values
        max_depths = [d if d != 999 else 25 for d in max_depths]  # Replace 999 with 25 for visualization
        
        ax1.plot(max_depths, self.comparison_df['Entropy_Train_Acc'], 'o-', label='Entropy (Train)', linewidth=2)
        ax1.plot(max_depths, self.comparison_df['Entropy_Test_Acc'], 's--', label='Entropy (Test)', linewidth=2)
        ax1.plot(max_depths, self.comparison_df['GINI_Train_Acc'], 'o-', label='GINI (Train)', linewidth=2)
        ax1.plot(max_depths, self.comparison_df['GINI_Test_Acc'], 's--', label='GINI (Test)', linewidth=2)
        ax1.set_xlabel('Max Depth', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(f'{self.dataset_name}\nAccuracy vs Max Depth', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Actual Depth Comparison
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(max_depths, self.comparison_df['Entropy_Actual_Depth'], 'o-', label='Entropy', linewidth=2, markersize=8)
        ax2.plot(max_depths, self.comparison_df['GINI_Actual_Depth'], 's--', label='GINI', linewidth=2, markersize=8)
        ax2.set_xlabel('Max Depth Limit', fontsize=12)
        ax2.set_ylabel('Actual Tree Depth', fontsize=12)
        ax2.set_title('Actual Tree Depth vs Max Depth Limit', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Overfitting Analysis (Train vs Test)
        ax3 = plt.subplot(2, 3, 3)
        overfit_entropy = self.comparison_df['Entropy_Train_Acc'] - self.comparison_df['Entropy_Test_Acc']
        overfit_gini = self.comparison_df['GINI_Train_Acc'] - self.comparison_df['GINI_Test_Acc']
        ax3.plot(max_depths, overfit_entropy, 'o-', label='Entropy Overfitting', linewidth=2)
        ax3.plot(max_depths, overfit_gini, 's--', label='GINI Overfitting', linewidth=2)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Max Depth', fontsize=12)
        ax3.set_ylabel('Train Acc - Test Acc', fontsize=12)
        ax3.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Best Model Comparison
        ax4 = plt.subplot(2, 3, 4)
        best_entropy_idx = self.comparison_df['Entropy_Test_Acc'].idxmax()
        best_gini_idx = self.comparison_df['GINI_Test_Acc'].idxmax()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        entropy_vals = [
            self.results['entropy'][list(self.results['entropy'].keys())[best_entropy_idx]]['test_accuracy'],
            self.results['entropy'][list(self.results['entropy'].keys())[best_entropy_idx]]['precision'],
            self.results['entropy'][list(self.results['entropy'].keys())[best_entropy_idx]]['recall'],
            self.results['entropy'][list(self.results['entropy'].keys())[best_entropy_idx]]['f1_score']
        ]
        gini_vals = [
            self.results['gini'][list(self.results['gini'].keys())[best_gini_idx]]['test_accuracy'],
            self.results['gini'][list(self.results['gini'].keys())[best_gini_idx]]['precision'],
            self.results['gini'][list(self.results['gini'].keys())[best_gini_idx]]['recall'],
            self.results['gini'][list(self.results['gini'].keys())[best_gini_idx]]['f1_score']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax4.bar(x - width/2, entropy_vals, width, label='Entropy', alpha=0.8)
        ax4.bar(x + width/2, gini_vals, width, label='GINI', alpha=0.8)
        ax4.set_xlabel('Metrics', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Best Model Performance Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Confusion Matrix - Entropy
        ax5 = plt.subplot(2, 3, 5)
        best_entropy_key = list(self.results['entropy'].keys())[best_entropy_idx]
        cm_entropy = confusion_matrix(
            self.y_test, 
            self.results['entropy'][best_entropy_key]['y_test_pred']
        )
        sns.heatmap(cm_entropy, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_title('Confusion Matrix - Entropy (Best Model)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('True Label')
        ax5.set_xlabel('Predicted Label')
        
        # 6. Confusion Matrix - GINI
        ax6 = plt.subplot(2, 3, 6)
        best_gini_key = list(self.results['gini'].keys())[best_gini_idx]
        cm_gini = confusion_matrix(
            self.y_test, 
            self.results['gini'][best_gini_key]['y_test_pred']
        )
        sns.heatmap(cm_gini, annot=True, fmt='d', cmap='Greens', ax=ax6)
        ax6.set_title('Confusion Matrix - GINI (Best Model)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('True Label')
        ax6.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Create report folder if it doesn't exist
        report_dir = 'report'
        os.makedirs(report_dir, exist_ok=True)
        
        filename = f'{self.dataset_name.replace(" ", "_")}_analysis.png'
        filepath = os.path.join(report_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {filepath}")
        plt.close()
        
        return filepath


def plot_analysis(dataset_name, results, y_test, comparison_df):
    """
    Convenience function to create visualizations
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    results : dict
        Dictionary containing results from DecisionTreeAnalyzer
    y_test : array-like
        True test labels
    comparison_df : pandas.DataFrame
        DataFrame with depth comparison results
        
    Returns:
    --------
    str : Filename of saved visualization
    """
    visualizer = DecisionTreeVisualizer(dataset_name, results, y_test, comparison_df)
    return visualizer.plot_analysis()
