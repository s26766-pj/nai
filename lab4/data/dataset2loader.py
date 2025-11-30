"""
Dataset 2 Loader: Pima Indians Diabetes Dataset
Loads and prepares the Pima Indians Diabetes dataset for analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_diabetes_dataset(data_path='data/dataset2/pima-indians-diabetes.csv'):
    """
    Load and prepare the Pima Indians Diabetes dataset for machine learning analysis.
    
    This function reads the CSV file containing diabetes prediction data for Pima
    Indian women, extracts features and target labels, checks for missing values
    (represented as zeros), and splits the data into training and testing sets.
    
    Note: This dataset may contain zero values in certain features that represent
    missing data rather than actual zero measurements. These are kept as-is for
    this analysis but should be handled appropriately in production scenarios.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the CSV file containing the Pima Indians Diabetes dataset.
        Default: 'data/dataset2/pima-indians-diabetes.csv'
        
    Returns:
    --------
    dict : Dictionary containing the following keys:
        - 'X': numpy.ndarray
            Feature matrix with shape (n_samples, n_features) containing all samples
        - 'y': numpy.ndarray
            Target vector with shape (n_samples,) containing class labels (0 or 1)
        - 'feature_names': list
            List of 8 feature names related to health measurements
        - 'dataset_name': str
            Name of the dataset: "Pima Indians Diabetes"
        - 'X_train': numpy.ndarray
            Training feature matrix (70% of data)
        - 'X_test': numpy.ndarray
            Test feature matrix (30% of data)
        - 'y_train': numpy.ndarray
            Training target vector
        - 'y_test': numpy.ndarray
            Test target vector
        - 'missing_info': dict
            Dictionary mapping feature names to count of zero values (potential missing data)
    """
    # Display header for dataset loading
    print("\n" + "="*60)
    print("Loading Dataset 2: Pima Indians Diabetes")
    print("="*60)
    
    # Read the CSV file into a pandas DataFrame
    # header=None indicates the file has no column headers
    df = pd.read_csv(data_path, header=None)
    
    # Extract features (all columns except the last one) and convert to numpy array
    # The last column contains the target labels (diabetes diagnosis)
    X = df.iloc[:, :-1].values
    # Extract target labels (last column) and convert to numpy array
    # 0 = no diabetes, 1 = diabetes
    y = df.iloc[:, -1].values
    
    # Define feature names based on the dataset documentation
    # These represent various health measurements for diabetes prediction
    feature_names = [
        'Pregnancies',              # Number of times pregnant
        'Glucose',                  # Plasma glucose concentration
        'BloodPressure',            # Diastolic blood pressure
        'SkinThickness',           # Triceps skinfold thickness
        'Insulin',                  # 2-Hour serum insulin
        'BMI',                      # Body mass index
        'DiabetesPedigreeFunction', # Diabetes pedigree function
        'Age'                       # Age in years
    ]
    
    # Set the dataset name for identification in reports and visualizations
    dataset_name = "Pima Indians Diabetes"
    
    # Display dataset information to the user
    print(f"Dataset loaded successfully!")
    print(f"Shape: {X.shape}")  # Show number of samples and features
    print(f"Features: {len(feature_names)}")
    print(f"  - {', '.join(feature_names)}")
    print(f"Classes: {np.unique(y)}")  # Show unique class labels
    print(f"Class distribution:")
    # Display count of each class to check for class imbalance
    print(pd.Series(y).value_counts().to_string())
    
    # Important note about missing data
    # In this dataset, zero values in certain features (like Glucose, BloodPressure)
    # may represent missing data rather than actual zero measurements
    # For this analysis, we keep them as-is, but in production they should be handled
    print("\nNote: Some features may contain 0 values representing missing data")
    print("For this analysis, these values are kept as-is")
    
    # Check for potential missing values by counting zeros in each feature
    # This helps identify which features have suspicious zero values
    missing_info = {}
    for i, name in enumerate(feature_names):
        # Count how many zero values exist in this feature
        zero_count = np.sum(X[:, i] == 0)
        if zero_count > 0:
            # Store the count for features that have zeros
            missing_info[name] = zero_count
    
    # Display information about features with zero values
    if missing_info:
        print(f"\nFeatures with zero values (potential missing data):")
        for name, count in missing_info.items():
            # Calculate and display percentage of zeros
            percentage = count/len(X)*100
            print(f"  - {name}: {count} zeros ({percentage:.1f}%)")
    
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
    
    # Return all data, metadata, and missing value information
    return {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'dataset_name': dataset_name,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'missing_info': missing_info  # Include missing value information
    }


def get_dataset_info():
    """
    Get metadata and information about the Pima Indians Diabetes dataset.
    
    This function returns a dictionary containing descriptive information about
    the dataset, including detailed feature descriptions and target variable
    explanation. Also includes a note about potential missing values.
    
    Returns:
    --------
    dict : Dictionary containing dataset metadata:
        - 'name': str
            Dataset name
        - 'description': str
            Brief description of what the dataset is used for
        - 'features': list
            Detailed descriptions of each of the 8 features
        - 'target': str
            Description of the target variable and its values
        - 'source': str
            Path to the source data file
        - 'note': str
            Important note about missing values in the dataset
    """
    return {
        'name': 'Pima Indians Diabetes',
        'description': 'Diabetes prediction for Pima Indian women',
        'features': [
            'Number of times pregnant',
            'Plasma glucose concentration (2 hours in oral glucose tolerance test)',
            'Diastolic blood pressure (mm Hg)',
            'Triceps skinfold thickness (mm)',
            '2-Hour serum insulin (mu U/ml)',
            'Body mass index (weight in kg/(height in m)^2)',
            'Diabetes pedigree function',
            'Age (years)'
        ],
        'target': 'Class variable (0 = no diabetes, 1 = diabetes)',
        'source': 'data/dataset2/pima-indians-diabetes.csv',
        'note': 'Some features may contain 0 values representing missing data'
    }


if __name__ == "__main__":
    # Test the loader
    data = load_diabetes_dataset()
    print("\n" + "="*60)
    print("Dataset loaded successfully!")
    print("="*60)
    print(f"\nAvailable keys: {list(data.keys())}")
