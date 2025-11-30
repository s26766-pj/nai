"""
Decision Tree Training: Entropy vs GINI Index
Handles training of decision trees with different criteria and depths
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DecisionTreeTrainer:
    """
    Handles training of decision trees with different splitting criteria and depths.
    
    This class trains multiple decision tree models using different splitting criteria
    (Entropy and GINI Index) and various maximum depth values to analyze how these
    parameters affect model performance, overfitting, and tree complexity.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the DecisionTreeTrainer with training and test data.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_train_samples, n_features)
            Training feature matrix used to train the decision tree models
        X_test : array-like of shape (n_test_samples, n_features)
            Test feature matrix used to evaluate model performance
        y_train : array-like of shape (n_train_samples,)
            Training target labels (class values)
        y_test : array-like of shape (n_test_samples,)
            Test target labels (class values) for evaluation
        """
        # Store training data for model training
        self.X_train = X_train
        # Store test data for model evaluation
        self.X_test = X_test
        # Store training labels for supervised learning
        self.y_train = y_train
        # Store test labels for performance evaluation
        self.y_test = y_test
    
    def train_decision_trees(self, max_depths=None, random_state=42):
        """
        Train decision trees with different splitting criteria and maximum depths.
        
        This method trains multiple decision tree models to compare:
        - Entropy vs GINI Index splitting criteria
        - Different maximum tree depths to analyze overfitting
        - Model performance metrics (accuracy, precision, recall, F1-score)
        - Tree complexity (depth, number of nodes, number of leaves)
        
        Parameters:
        -----------
        max_depths : list, optional
            List of maximum depths to test. Each value limits the tree depth.
            Use None for unlimited depth. Default: [3, 5, 7, 10, 15, 20, None]
        random_state : int, optional
            Random seed for reproducibility. Ensures same results across runs.
            Default: 42
            
        Returns:
        --------
        dict : Nested dictionary containing results for each criterion and depth.
            Structure: {
                'entropy': {
                    'depth_3': {model, metrics, tree_properties, ...},
                    'depth_5': {...},
                    ...
                },
                'gini': {
                    'depth_3': {...},
                    ...
                }
            }
        """
        # Set default depths if not provided
        # None means unlimited depth (tree grows until all leaves are pure)
        if max_depths is None:
            max_depths = [3, 5, 7, 10, 15, 20, None]  # None = no limit
        
        # Define the two splitting criteria to compare
        # 'entropy': Uses information gain (ID3, C4.5, C5.0 algorithm style)
        # 'gini': Uses GINI impurity (CART algorithm style)
        criteria = ['entropy', 'gini']
        
        # Initialize results dictionary with empty dicts for each criterion
        results = {criterion: {} for criterion in criteria}
        
        # Iterate through each splitting criterion
        for criterion in criteria:
            # Display header for current criterion
            print(f"\n{'='*60}")
            print(f"Training with {criterion.upper()} criterion")
            print(f"{'='*60}")
            
            # Iterate through each maximum depth value
            for depth in max_depths:
                # Create a key for this depth configuration
                # 'unlimited' for None, 'depth_X' for specific depths
                depth_key = 'unlimited' if depth is None else f'depth_{depth}'
                
                # Create and configure the decision tree classifier
                clf = DecisionTreeClassifier(
                    criterion=criterion,      # Splitting criterion (entropy or gini)
                    max_depth=depth,          # Maximum tree depth (None = unlimited)
                    random_state=random_state, # Seed for reproducibility
                    min_samples_split=2,      # Minimum samples required to split a node
                    min_samples_leaf=1        # Minimum samples required in a leaf node
                )
                
                # Train the decision tree on the training data
                # The algorithm recursively splits nodes to maximize information gain
                clf.fit(self.X_train, self.y_train)
                
                # Make predictions on both training and test sets
                # Training predictions help detect overfitting
                y_train_pred = clf.predict(self.X_train)
                # Test predictions are used for final performance evaluation
                y_test_pred = clf.predict(self.X_test)
                
                # Calculate performance metrics
                # Training accuracy: how well the model fits the training data
                train_acc = accuracy_score(self.y_train, y_train_pred)
                # Test accuracy: how well the model generalizes to unseen data
                test_acc = accuracy_score(self.y_test, y_test_pred)
                # Precision: proportion of positive predictions that are correct
                # 'weighted' accounts for class imbalance
                precision = precision_score(self.y_test, y_test_pred, average='weighted')
                # Recall: proportion of actual positives that were correctly identified
                recall = recall_score(self.y_test, y_test_pred, average='weighted')
                # F1-score: harmonic mean of precision and recall
                f1 = f1_score(self.y_test, y_test_pred, average='weighted')
                
                # Extract tree structure properties
                # Actual depth: the maximum depth the tree actually reached
                actual_depth = clf.tree_.max_depth
                # Number of nodes: total decision nodes in the tree
                n_nodes = clf.tree_.node_count
                # Number of leaves: terminal nodes (final predictions)
                n_leaves = clf.tree_.n_leaves
                
                # Store all results for this configuration
                results[criterion][depth_key] = {
                    'model': clf,                    # Trained model for later use
                    'train_accuracy': train_acc,     # Training set accuracy
                    'test_accuracy': test_acc,        # Test set accuracy
                    'precision': precision,           # Precision score
                    'recall': recall,                 # Recall score
                    'f1_score': f1,                  # F1-score
                    'depth': actual_depth,            # Actual tree depth
                    'n_nodes': n_nodes,              # Total number of nodes
                    'n_leaves': n_leaves,            # Number of leaf nodes
                    'y_test_pred': y_test_pred       # Test predictions for confusion matrix
                }
                
                # Display results for this configuration
                print(f"\nMax Depth: {depth_key}")
                print(f"  Actual Depth: {actual_depth}")
                print(f"  Nodes: {n_nodes}, Leaves: {n_leaves}")
                print(f"  Train Accuracy: {train_acc:.4f}")
                print(f"  Test Accuracy: {test_acc:.4f}")
                print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Return all results for analysis and visualization
        return results


def train_decision_trees(X_train, X_test, y_train, y_test, max_depths=None, random_state=42):
    """
    Convenience function to train decision trees without instantiating the class.
    
    This is a wrapper function that creates a DecisionTreeTrainer instance and
    calls its train_decision_trees method. Useful for quick training without
    needing to manage the trainer object.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_train_samples, n_features)
        Training feature matrix
    X_test : array-like of shape (n_test_samples, n_features)
        Test feature matrix
    y_train : array-like of shape (n_train_samples,)
        Training target labels
    y_test : array-like of shape (n_test_samples,)
        Test target labels
    max_depths : list, optional
        List of maximum depths to test. None = no limit.
        Default: [3, 5, 7, 10, 15, 20, None]
    random_state : int, optional
        Random seed for reproducibility. Default: 42
        
    Returns:
    --------
    dict : Dictionary containing results for each criterion and depth.
        Same structure as DecisionTreeTrainer.train_decision_trees()
    """
    # Create trainer instance with provided data
    trainer = DecisionTreeTrainer(X_train, X_test, y_train, y_test)
    # Train models and return results
    return trainer.train_decision_trees(max_depths=max_depths, random_state=random_state)
