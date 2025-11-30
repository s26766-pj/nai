"""
SVM Visualization: Support Vector Machine Visualization
Handles all plotting and visualization functionality for SVM analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SVMVisualizer:
    """Handles visualization of SVM analysis results"""
    
    def __init__(self, dataset_name, results, y_test, summary_df):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        results : dict
            Dictionary containing results from SVMAnalyzer
        y_test : array-like
            True test labels
        summary_df : pandas.DataFrame
            DataFrame with kernel performance summary
        """
        self.dataset_name = dataset_name
        self.results = results
        self.y_test = y_test
        self.summary_df = summary_df
    
    def plot_analysis(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Kernel Performance Comparison (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        kernel_perf = self.summary_df.groupby('Kernel')['Test_Accuracy'].max().sort_values(ascending=False)
        ax1.bar(range(len(kernel_perf)), kernel_perf.values, alpha=0.7)
        ax1.set_xlabel('Kernel', fontsize=12)
        ax1.set_ylabel('Best Test Accuracy', fontsize=12)
        ax1.set_title(f'{self.dataset_name}\nBest Kernel Performance', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(kernel_perf)))
        ax1.set_xticklabels(kernel_perf.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(kernel_perf.values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. All Configurations Performance
        ax2 = plt.subplot(2, 3, 2)
        colors = {'linear': 'blue', 'poly': 'green', 'rbf': 'red', 'sigmoid': 'orange'}
        for kernel in self.results.keys():
            configs = []
            accs = []
            for config_key, result in self.results[kernel].items():
                config_str = f"{config_key}"
                configs.append(config_str)
                accs.append(result['test_accuracy'])
            ax2.scatter(range(len(accs)), accs, label=kernel, alpha=0.6, s=100, c=colors.get(kernel, 'gray'))
        ax2.set_xlabel('Configuration Index', fontsize=12)
        ax2.set_ylabel('Test Accuracy', fontsize=12)
        ax2.set_title('All Kernel Configurations Performance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Support Vectors Comparison
        ax3 = plt.subplot(2, 3, 3)
        kernel_sv = {}
        for kernel in self.results.keys():
            sv_counts = [result['n_support_vectors'] for result in self.results[kernel].values()]
            kernel_sv[kernel] = np.mean(sv_counts)
        
        kernels = list(kernel_sv.keys())
        sv_counts = list(kernel_sv.values())
        ax3.bar(range(len(kernels)), sv_counts, alpha=0.7, color=[colors.get(k, 'gray') for k in kernels])
        ax3.set_xlabel('Kernel', fontsize=12)
        ax3.set_ylabel('Average Support Vectors', fontsize=12)
        ax3.set_title('Average Support Vectors per Kernel', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(kernels)))
        ax3.set_xticklabels(kernels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Best Model Metrics Comparison
        ax4 = plt.subplot(2, 3, 4)
        best_per_kernel = {}
        for kernel in self.results.keys():
            best = max(self.results[kernel].items(), key=lambda x: x[1]['test_accuracy'])
            best_per_kernel[kernel] = best[1]
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (kernel, result) in enumerate(best_per_kernel.items()):
            offset = (i - len(best_per_kernel)/2 + 0.5) * width
            vals = [
                result['test_accuracy'],
                result['precision'],
                result['recall'],
                result['f1_score']
            ]
            ax4.bar(x + offset, vals, width, label=kernel, alpha=0.8, color=colors.get(kernel, 'gray'))
        
        ax4.set_xlabel('Metrics', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Best Model Performance by Kernel', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Confusion Matrix - Best Overall Model
        ax5 = plt.subplot(2, 3, 5)
        best_overall = None
        best_kernel = None
        best_config = None
        best_acc = 0
        
        for kernel in self.results.keys():
            for config_key, result in self.results[kernel].items():
                if result['test_accuracy'] > best_acc:
                    best_acc = result['test_accuracy']
                    best_overall = result
                    best_kernel = kernel
                    best_config = config_key
        
        cm = confusion_matrix(self.y_test, best_overall['y_test_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_title(f'Confusion Matrix - {best_kernel.upper()} (Best Model)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('True Label')
        ax5.set_xlabel('Predicted Label')
        
        # 6. Parameter Effect Analysis (for RBF kernel if available)
        ax6 = plt.subplot(2, 3, 6)
        if 'rbf' in self.results:
            c_values = []
            gamma_values = []
            accuracies = []
            
            for config_key, result in self.results['rbf'].items():
                config = result['config']
                c_val = config.get('C', 1.0)
                gamma_val = config.get('gamma', 'scale')
                
                # Convert gamma to numeric if possible
                if isinstance(gamma_val, str):
                    gamma_num = 0.1 if gamma_val == 'auto' else 0.01  # Approximate
                else:
                    gamma_num = gamma_val
                
                c_values.append(c_val)
                gamma_values.append(gamma_num)
                accuracies.append(result['test_accuracy'])
            
            scatter = ax6.scatter(c_values, gamma_values, c=accuracies, s=100, alpha=0.6, cmap='viridis')
            ax6.set_xlabel('C Parameter', fontsize=12)
            ax6.set_ylabel('Gamma Parameter', fontsize=12)
            ax6.set_title('RBF Kernel: Parameter Effect on Accuracy', fontsize=14, fontweight='bold')
            ax6.set_xscale('log')
            ax6.set_yscale('log')
            plt.colorbar(scatter, ax=ax6, label='Test Accuracy')
        else:
            # If no RBF, show kernel comparison table
            ax6.axis('off')
            table_data = []
            for kernel in self.results.keys():
                best = max(self.results[kernel].items(), key=lambda x: x[1]['test_accuracy'])
                table_data.append([
                    kernel,
                    f"{best[1]['test_accuracy']:.4f}",
                    f"{best[1]['f1_score']:.4f}",
                    best[1]['n_support_vectors']
                ])
            
            table = ax6.table(cellText=table_data,
                            colLabels=['Kernel', 'Accuracy', 'F1-Score', 'SV'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax6.set_title('Best Configuration per Kernel', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Create report folder if it doesn't exist
        report_dir = 'report'
        os.makedirs(report_dir, exist_ok=True)
        
        filename = f'{self.dataset_name.replace(" ", "_")}_SVM_analysis.png'
        filepath = os.path.join(report_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nSVM visualization saved as: {filepath}")
        plt.close()
        
        return filepath
    
    def plot_decision_boundary_3d(self, X_train, X_test, y_train, y_test, kernel_name='rbf', config_key=None, scaler=None):
        """
        Visualize the SVM decision boundary (hyperplane) in 3D space.
        
        This method uses PCA to project high-dimensional data to 3D for visualization.
        It shows the decision boundary as a 3D surface, support vectors, and data points.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_train_samples, n_features)
            Training feature matrix
        X_test : array-like of shape (n_test_samples, n_features)
            Test feature matrix
        y_train : array-like of shape (n_train_samples,)
            Training labels
        y_test : array-like of shape (n_test_samples,)
            Test labels
        kernel_name : str, optional
            Kernel to visualize. Default: 'rbf'
        config_key : str, optional
            Configuration key. If None, uses best configuration for that kernel
        scaler : sklearn.preprocessing.StandardScaler, optional
            Scaler used during training. If provided, applies same scaling.
            
        Returns:
        --------
        str : Filepath where the visualization was saved
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
        
        # Get the trained model
        model = self.results[kernel_name][config_key]['model']
        config = self.results[kernel_name][config_key]['config']
        
        # Combine train and test data for PCA
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.hstack([y_train, y_test])
        
        # Apply scaling if scaler is provided (same as used during training)
        if scaler is not None:
            X_combined_scaled = scaler.transform(X_combined)
        else:
            X_combined_scaled = X_combined
        
        # Use PCA to reduce to 3D for visualization
        # This projects high-dimensional data to 3D while preserving maximum variance
        pca = PCA(n_components=3, random_state=42)
        X_3d = pca.fit_transform(X_combined_scaled)
        
        # Split back into train and test
        n_train = len(X_train)
        X_train_3d = X_3d[:n_train]
        X_test_3d = X_3d[n_train:]
        
        # Train a new SVM on the 3D projected data with same parameters
        from sklearn.svm import SVC
        svm_3d = SVC(
            kernel=kernel_name,
            random_state=42,
            probability=True,
            **config
        )
        svm_3d.fit(X_train_3d, y_train)
        
        # Create a 3D mesh grid to plot the decision boundary
        # Define the range of the plot
        h = 0.1  # Step size in the mesh (smaller for 3D)
        x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
        y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
        z_min, z_max = X_3d[:, 2].min() - 0.5, X_3d[:, 2].max() + 0.5
        
        # Create mesh grid for the first two dimensions
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # For each point in the mesh, find the z value that creates the decision boundary
        # We'll sample z values and predict
        zz_samples = np.linspace(z_min, z_max, 20)
        
        # Create the 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot training points
        ax.scatter(X_train_3d[y_train == 0, 0], X_train_3d[y_train == 0, 1], X_train_3d[y_train == 0, 2],
                  c='blue', marker='o', s=30, alpha=0.6, label='Train Class 0', edgecolors='black', linewidths=0.5)
        ax.scatter(X_train_3d[y_train == 1, 0], X_train_3d[y_train == 1, 1], X_train_3d[y_train == 1, 2],
                  c='red', marker='s', s=30, alpha=0.6, label='Train Class 1', edgecolors='black', linewidths=0.5)
        
        # Plot test points
        ax.scatter(X_test_3d[y_test == 0, 0], X_test_3d[y_test == 0, 1], X_test_3d[y_test == 0, 2],
                  c='lightblue', marker='^', s=60, alpha=0.9, label='Test Class 0', edgecolors='black', linewidths=1)
        ax.scatter(X_test_3d[y_test == 1, 0], X_test_3d[y_test == 1, 1], X_test_3d[y_test == 1, 2],
                  c='orange', marker='v', s=60, alpha=0.9, label='Test Class 1', edgecolors='black', linewidths=1)
        
        # Highlight support vectors
        support_vectors_3d = X_train_3d[svm_3d.support_]
        ax.scatter(support_vectors_3d[:, 0], support_vectors_3d[:, 1], support_vectors_3d[:, 2],
                  s=300, facecolors='none', edgecolors='yellow', linewidths=2.5,
                  label=f'Support Vectors ({len(support_vectors_3d)})')
        
        # Create decision boundary surface
        # Sample points on a grid and predict their class
        # Then create an isosurface at the decision boundary
        zz_mid = (z_min + z_max) / 2
        zz = np.full_like(xx, zz_mid)
        
        # Predict for points in the mesh at the middle z level
        mesh_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        Z = svm_3d.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Create a surface showing the decision boundary
        # Use a colormap to show different regions
        surf = ax.plot_surface(xx, yy, zz, facecolors=plt.cm.RdYlBu(Z), 
                              alpha=0.3, linewidth=0, antialiased=True)
        
        # For better visualization, create multiple slices at different z levels
        for z_level in np.linspace(z_min, z_max, 5):
            zz_slice = np.full_like(xx, z_level)
            mesh_slice = np.c_[xx.ravel(), yy.ravel(), zz_slice.ravel()]
            Z_slice = svm_3d.predict(mesh_slice)
            Z_slice = Z_slice.reshape(xx.shape)
            ax.contour(xx, yy, Z_slice, zdir='z', offset=z_level, 
                      colors='black', alpha=0.3, linewidths=0.5)
        
        # Set labels and title
        variance_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}% variance)', fontsize=11)
        ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}% variance)', fontsize=11)
        ax.set_zlabel(f'PC3 ({variance_explained[2]*100:.1f}% variance)', fontsize=11)
        config_str = ', '.join([f"{k}={v}" for k, v in config.items()])
        ax.set_title(f'{self.dataset_name}\nSVM Decision Boundary 3D ({kernel_name.upper()} Kernel)\n{config_str}', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=9)
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Save the plot
        report_dir = 'report'
        os.makedirs(report_dir, exist_ok=True)
        filename = f'{self.dataset_name.replace(" ", "_")}_SVM_decision_boundary_3d_{kernel_name}.png'
        filepath = os.path.join(report_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n3D Decision boundary visualization saved as: {filepath}")
        plt.close()
        
        return filepath
    
    def plot_decision_boundary(self, X_train, X_test, y_train, y_test, kernel_name='rbf', config_key=None, scaler=None, plot_3d=True):
        """
        Visualize the SVM decision boundary (hyperplane) in 2D and/or 3D space.
        
        Since the datasets have multiple dimensions, this method uses PCA to project
        the data to 2D (and optionally 3D) for visualization. It shows:
        - Decision boundary (hyperplane) separating the classes
        - Support vectors (data points that define the boundary)
        - Training and test data points
        - Margin boundaries (for linear kernel in 2D)
        
        Parameters:
        -----------
        X_train : array-like of shape (n_train_samples, n_features)
            Training feature matrix
        X_test : array-like of shape (n_test_samples, n_features)
            Test feature matrix
        y_train : array-like of shape (n_train_samples,)
            Training labels
        y_test : array-like of shape (n_test_samples,)
            Test labels
        kernel_name : str, optional
            Kernel to visualize. Default: 'rbf'
        config_key : str, optional
            Configuration key. If None, uses best configuration for that kernel
        scaler : sklearn.preprocessing.StandardScaler, optional
            Scaler used during training. If provided, applies same scaling.
        plot_3d : bool, optional
            If True, creates both 2D and 3D visualizations. If False, only 2D.
            Default: True
            
        Returns:
        --------
        str or tuple : 
            If plot_3d=False: Filepath of 2D visualization
            If plot_3d=True: Tuple of (2D_filepath, 3D_filepath)
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
        
        # Get the trained model
        model = self.results[kernel_name][config_key]['model']
        
        # Combine train and test data for PCA
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.hstack([y_train, y_test])
        
        # Apply scaling if scaler is provided (same as used during training)
        if scaler is not None:
            X_combined_scaled = scaler.transform(X_combined)
        else:
            X_combined_scaled = X_combined
        
        # Use PCA to reduce to 2D for visualization
        # This projects high-dimensional data to 2D while preserving maximum variance
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_combined_scaled)
        
        # Split back into train and test
        n_train = len(X_train)
        X_train_2d = X_2d[:n_train]
        X_test_2d = X_2d[n_train:]
        
        # Train a new SVM on the 2D projected data with same parameters
        # This allows us to visualize the decision boundary in 2D
        from sklearn.svm import SVC
        config = self.results[kernel_name][config_key]['config']
        svm_2d = SVC(
            kernel=kernel_name,
            random_state=42,
            probability=True,
            **config
        )
        svm_2d.fit(X_train_2d, y_train)
        
        # Create a mesh grid to plot the decision boundary
        # Define the range of the plot
        h = 0.02  # Step size in the mesh
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Predict for each point in the mesh
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the decision boundary and regions
        # Contour plot shows the decision regions
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        # Contour lines show the decision boundary
        ax.contour(xx, yy, Z, colors='black', alpha=0.6, linewidths=1.5, linestyles='solid')
        
        # Plot training points
        scatter1 = ax.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1],
                            c='blue', marker='o', s=50, alpha=0.7, label='Train Class 0', edgecolors='black')
        scatter2 = ax.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1],
                            c='red', marker='s', s=50, alpha=0.7, label='Train Class 1', edgecolors='black')
        
        # Plot test points
        scatter3 = ax.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1],
                            c='lightblue', marker='^', s=80, alpha=0.9, label='Test Class 0', edgecolors='black', linewidths=1.5)
        scatter4 = ax.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1],
                            c='orange', marker='v', s=80, alpha=0.9, label='Test Class 1', edgecolors='black', linewidths=1.5)
        
        # Highlight support vectors
        support_vectors_2d = X_train_2d[svm_2d.support_]
        ax.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                  s=200, facecolors='none', edgecolors='yellow', linewidths=2,
                  label=f'Support Vectors ({len(support_vectors_2d)})', zorder=10)
        
        # Add margin lines for linear kernel
        if kernel_name == 'linear':
            # Get the separating hyperplane
            w = svm_2d.coef_[0]
            a = -w[0] / w[1]
            xx_line = np.linspace(x_min, x_max)
            yy_line = a * xx_line - (svm_2d.intercept_[0]) / w[1]
            
            # Plot the separating line
            ax.plot(xx_line, yy_line, 'k-', linewidth=2, label='Decision Boundary')
            
            # Plot margin boundaries
            margin = 1 / np.sqrt(np.sum(svm_2d.coef_[0] ** 2))
            yy_down = yy_line - np.sqrt(1 + a ** 2) * margin
            yy_up = yy_line + np.sqrt(1 + a ** 2) * margin
            
            ax.plot(xx_line, yy_down, 'k--', linewidth=1.5, alpha=0.7, label='Margin')
            ax.plot(xx_line, yy_up, 'k--', linewidth=1.5, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
        config_str = ', '.join([f"{k}={v}" for k, v in config.items()])
        ax.set_title(f'{self.dataset_name}\nSVM Decision Boundary ({kernel_name.upper()} Kernel)\n{config_str}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        report_dir = 'report'
        os.makedirs(report_dir, exist_ok=True)
        filename = f'{self.dataset_name.replace(" ", "_")}_SVM_decision_boundary_{kernel_name}.png'
        filepath = os.path.join(report_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n2D Decision boundary visualization saved as: {filepath}")
        plt.close()
        
        # Also create 3D visualization if requested
        if plot_3d:
            filepath_3d = self.plot_decision_boundary_3d(
                X_train, X_test, y_train, y_test, 
                kernel_name, config_key, scaler
            )
            return filepath, filepath_3d
        
        return filepath


def plot_svm_analysis(dataset_name, results, y_test, summary_df):
    """
    Convenience function to create SVM visualizations
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    results : dict
        Dictionary containing results from SVMAnalyzer
    y_test : array-like
        True test labels
    summary_df : pandas.DataFrame
        DataFrame with kernel performance summary
        
    Returns:
    --------
    str : Filename of saved visualization
    """
    visualizer = SVMVisualizer(dataset_name, results, y_test, summary_df)
    return visualizer.plot_analysis()
