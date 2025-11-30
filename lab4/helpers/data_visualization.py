"""
Data Visualization: Sample data visualization for datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DataVisualizer:
    """
    Create exploratory data analysis (EDA) visualizations for datasets.
    
    This class generates comprehensive visualizations to understand dataset
    characteristics before model training, including class distribution,
    feature distributions, and correlation matrices.
    """
    
    def __init__(self, dataset_name, X, y, feature_names):
        """
        Initialize the DataVisualizer with dataset information.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset (used in plot titles and filenames)
        X : array-like of shape (n_samples, n_features)
            Feature matrix containing all input features
        y : array-like of shape (n_samples,)
            Target vector containing class labels
        feature_names : list
            List of feature names corresponding to columns in X
        """
        # Store dataset name for titles and file naming
        self.dataset_name = dataset_name
        # Convert to numpy arrays for consistent processing
        self.X = np.array(X)
        self.y = np.array(y)
        # Store feature names for labeling plots
        self.feature_names = feature_names
        # Create pandas DataFrame for easier data manipulation
        # This makes it easier to filter by class and access features
        self.df = pd.DataFrame(X, columns=feature_names)
        # Add target column to DataFrame for easy filtering by class
        self.df['Target'] = y
    
    def plot_data_overview(self, n_samples=100):
        """
        Create comprehensive exploratory data analysis (EDA) visualizations.
        
        This method generates multiple plots to understand the dataset:
        - Class distribution: shows if classes are balanced or imbalanced
        - Feature distributions: histograms showing feature values by class
        - Correlation matrix: shows relationships between features
        
        Parameters:
        -----------
        n_samples : int, optional
            Maximum number of samples to use for visualization.
            For large datasets, sampling helps reduce computation time.
            Default: 100
            
        Returns:
        --------
        str : Filepath where the visualization was saved
        """
        # Sample data if dataset is too large
        # This speeds up visualization for large datasets while maintaining randomness
        if len(self.df) > n_samples:
            df_sample = self.df.sample(n=n_samples, random_state=42)
        else:
            df_sample = self.df
        
        # Create figure with 2 rows and 3 columns of subplots
        # Large figure size (20x12) for better readability
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Class Distribution
        # Shows how many samples belong to each class (important for detecting imbalance)
        ax1 = plt.subplot(2, 3, 1)
        # Count occurrences of each class and sort by class index
        class_counts = pd.Series(self.y).value_counts().sort_index()
        # Create bar chart showing count for each class
        ax1.bar(range(len(class_counts)), class_counts.values, alpha=0.7)
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'{self.dataset_name}\nClass Distribution', fontsize=14, fontweight='bold')
        # Set x-axis ticks and labels to show class numbers
        ax1.set_xticks(range(len(class_counts)))
        ax1.set_xticklabels([f'Class {i}' for i in class_counts.index])
        ax1.grid(True, alpha=0.3, axis='y')  # Add grid for easier reading
        
        # Plot 2-5: Feature Distributions (first 4 features)
        # Shows how feature values are distributed, separated by class
        # This helps identify which features might be good predictors
        n_features_to_plot = min(4, len(self.feature_names))
        for i, feature in enumerate(self.feature_names[:n_features_to_plot]):
            # Create subplot for this feature
            ax = plt.subplot(2, 3, 2 + i)
            # Plot histogram for each class separately
            for class_val in np.unique(self.y):
                # Filter data for this class
                class_data = self.df[self.df['Target'] == class_val][feature]
                # Create histogram with transparency for overlapping classes
                ax.hist(class_data, alpha=0.6, label=f'Class {class_val}', bins=20)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution: {feature}', fontsize=12, fontweight='bold')
            ax.legend()  # Show legend to distinguish classes
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Correlation Matrix
        # Shows how features are related to each other
        # High correlation might indicate redundant features
        if len(self.feature_names) > 1:
            ax_corr = plt.subplot(2, 3, 6)
            # Calculate Pearson correlation coefficient between all feature pairs
            corr_matrix = self.df[self.feature_names].corr()
            # Create heatmap with correlation values
            # coolwarm colormap: blue for negative, red for positive correlation
            # center=0: white at zero correlation
            # annot=True: show correlation values in each cell
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax_corr, square=True)
            ax_corr.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        
        # Adjust subplot spacing to prevent overlap
        plt.tight_layout()
        
        # Create report folder if it doesn't exist
        # This is where all generated visualizations are saved
        report_dir = 'report'
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate filename by replacing spaces with underscores
        filename = f'{self.dataset_name.replace(" ", "_")}_data_overview.png'
        filepath = os.path.join(report_dir, filename)
        # Save figure with high resolution (300 DPI) for publication quality
        # bbox_inches='tight' ensures no labels are cut off
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nData visualization saved as: {filepath}")
        # Close figure to free memory
        plt.close()
        
        return filepath


def plot_data_overview(dataset_name, X, y, feature_names, n_samples=100):
    """
    Convenience function to create data visualization without instantiating the class.
    
    This is a wrapper function that creates a DataVisualizer instance and
    immediately generates the visualization. Useful for quick EDA without
    needing to manage the visualizer object.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (used in plot titles and filenames)
    X : array-like of shape (n_samples, n_features)
        Feature matrix containing all input features
    y : array-like of shape (n_samples,)
        Target vector containing class labels
    feature_names : list
        List of feature names corresponding to columns in X
    n_samples : int, optional
        Maximum number of samples to use for visualization.
        Default: 100
        
    Returns:
    --------
    str : Filepath where the visualization was saved
    """
    # Create visualizer instance with provided data
    visualizer = DataVisualizer(dataset_name, X, y, feature_names)
    # Generate and save visualization, return filepath
    return visualizer.plot_data_overview(n_samples=n_samples)
