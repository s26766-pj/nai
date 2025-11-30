"""
Dataset 1 Loader: Banknote Authentication Dataset
Loads and prepares the banknote authentication dataset for analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_banknote_dataset(data_path='data/dataset1/banknote-authentication.csv'):
    """
    Load and prepare the Banknote Authentication dataset for machine learning analysis.
    
    This function reads the CSV file containing banknote authentication data, extracts
    features and target labels, and splits the data into training and testing sets.
    The dataset contains image features extracted from banknote images to classify
    them as authentic or fake.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the CSV file containing the banknote authentication dataset.
        Default: 'data/dataset1/banknote-authentication.csv'
        
    Returns:
    --------
    dict : Dictionary containing the following keys:
        - 'X': numpy.ndarray
            Feature matrix with shape (n_samples, n_features) containing all samples
        - 'y': numpy.ndarray
            Target vector with shape (n_samples,) containing class labels (0 or 1)
        - 'feature_names': list
            List of feature names: ['Variance', 'Skewness', 'Curtosis', 'Entropy']
        - 'dataset_name': str
            Name of the dataset: "Banknote Authentication"
        - 'X_train': numpy.ndarray
            Training feature matrix (70% of data)
        - 'X_test': numpy.ndarray
            Test feature matrix (30% of data)
        - 'y_train': numpy.ndarray
            Training target vector
        - 'y_test': numpy.ndarray
            Test target vector
    """
    # Display header for dataset loading
    print("\n" + "="*60)
    print("Loading Dataset 1: Banknote Authentication")
    print("="*60)
    
    # Read the CSV file into a pandas DataFrame
    # header=None indicates the file has no column headers
    df = pd.read_csv(data_path, header=None)
    
    # Extract features (all columns except the last one) and convert to numpy array
    # The last column contains the target labels
    X = df.iloc[:, :-1].values
    # Extract target labels (last column) and convert to numpy array
    y = df.iloc[:, -1].values
    
    # Define feature names based on the dataset documentation
    # These are image features extracted from wavelet transforms
    feature_names = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    
    # Set the dataset name for identification in reports and visualizations
    dataset_name = "Banknote Authentication"
    
    # Display dataset information to the user
    print(f"Dataset loaded successfully!")
    print(f"Shape: {X.shape}")  # Show number of samples and features
    print(f"Features: {len(feature_names)}")
    print(f"  - {', '.join(feature_names)}")
    print(f"Classes: {np.unique(y)}")  # Show unique class labels
    print(f"Class distribution:")
    # Display count of each class to check for class imbalance
    print(pd.Series(y).value_counts().to_string())
    
    # Split the dataset into training and testing sets
    # test_size=0.3: 30% for testing, 70% for training
    # random_state=42: Ensures reproducible splits across runs
    # stratify=y: Maintains the same class distribution in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Display split information
    print(f"\nData split:")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Return all data and metadata in a dictionary for easy access
    return {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'dataset_name': dataset_name,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def get_dataset_info():
    """
    Get metadata and information about the Banknote Authentication dataset.
    
    This function returns a dictionary containing descriptive information about
    the dataset, including feature descriptions and target variable explanation.
    Useful for documentation and understanding the dataset structure.
    
    Returns:
    --------
    dict : Dictionary containing dataset metadata:
        - 'name': str
            Dataset name
        - 'description': str
            Brief description of what the dataset is used for
        - 'features': list
            Detailed descriptions of each feature
        - 'target': str
            Description of the target variable and its values
        - 'source': str
            Path to the source data file
    """
    return {
        'name': 'Banknote Authentication',
        'description': 'Banknote authentication using image features',
        'features': [
            'Variance of Wavelet Transformed image',
            'Skewness of Wavelet Transformed image',
            'Curtosis of Wavelet Transformed image',
            'Entropy of image'
        ],
        'target': 'Class (0 = authentic, 1 = fake)',
        'source': 'data/dataset1/banknote-authentication.csv'
    }


if __name__ == "__main__":
    # Test the loader
    data = load_banknote_dataset()
    print("\n" + "="*60)
    print("Dataset loaded successfully!")
    print("="*60)
    print(f"\nAvailable keys: {list(data.keys())}")
