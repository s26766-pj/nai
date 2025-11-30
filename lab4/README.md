# Decision Tree and SVM Analysis: Entropy vs GINI Index & Kernel Functions

## Overview
This project analyzes two datasets using:
1. **Decision Tree algorithms** with Entropy and GINI Index splitting criteria
2. **Support Vector Machine (SVM)** with different kernel functions and parameters

## Datasets
1. **Banknote Authentication Dataset**
   - 4 features: Variance, Skewness, Curtosis, Entropy
   - Binary classification (authentic vs fake banknotes)
   - 1,372 samples

2. **Pima Indians Diabetes Dataset**
   - 8 features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
   - Binary classification (diabetes positive/negative)
   - 768 samples

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Project Structure

```
lab4/
├── data/
│   ├── dataset1loader.py          # Banknote dataset loader
│   ├── dataset2loader.py          # Diabetes dataset loader
│   └── dataset1/, dataset2/       # CSV data files
├── decision_tree/
│   ├── decision_tree_training.py   # Decision tree training
│   ├── decision_tree_analysis.py  # Decision tree analysis
│   └── decision_tree_visualization.py  # Decision tree plots
├── support_vector_machine/
│   ├── svm_training.py            # SVM training with kernels
│   ├── svm_analysis.py            # SVM analysis
│   └── svm_visualization.py       # SVM plots
├── helpers/
│   ├── classification_metrics.py  # Detailed metrics display
│   └── data_visualization.py      # Sample data visualization
├── main.py                        # Main entry point
├── report/                        # Generated reports and visualizations
└── requirements.txt               # Python dependencies
```

## Features

### Decision Tree Analysis
- **Entropy vs GINI Index** comparison
- **Depth analysis** (3, 5, 7, 10, 15, 20, unlimited)
- **Overfitting detection**
- **Performance metrics** (Accuracy, Precision, Recall, F1-Score)
- **Confusion matrices**
- **Sample predictions**

### SVM Analysis
- **Multiple kernel functions**: Linear, Polynomial, RBF, Sigmoid
- **Parameter variations**: Different C, gamma, degree values
- **Kernel performance comparison**
- **Support vector analysis**
- **Detailed classification metrics**
- **Sample predictions**

## Output Files

All files are saved in the `report/` folder:

### Decision Tree Analysis
- `{dataset}_data_overview.png` - Sample data visualization
- `{dataset}_analysis.png` - Decision tree analysis plots
- `{dataset}_report.txt` - Research report

### SVM Analysis
- `{dataset}_SVM_analysis.png` - SVM kernel comparison plots
- `{dataset}_SVM_Kernel_Summary.txt` - Detailed kernel analysis summary

### Documentation
- `KERNEL_FUNCTIONS_SUMMARY.md` - Summary of how kernel functions affect classification results
- `report/SVM_Kernel_Functions_Summary.md` - Comprehensive guide on kernel functions

## Kernel Functions Summary

See `KERNEL_FUNCTIONS_SUMMARY.md` in the repository root for a concise summary of:
- How different kernel functions affect classification results
- Parameter effects on model performance
- Experimental findings from both datasets
- Recommendations for kernel and parameter selection

See `report/SVM_Kernel_Functions_Summary.md` for detailed information about:
- How each kernel function works
- Parameter effects on classification results
- When to use each kernel
- Best practices for parameter tuning

## Key Findings

### Decision Trees
- Optimal tree depth varies between Entropy and GINI
- Both criteria show similar performance patterns
- Deeper trees tend to overfit

### SVM Kernels
- **RBF kernel** typically performs best for non-linear problems
- **Linear kernel** works well for linearly separable data
- **Feature scaling** is critical for SVM performance
- Parameter selection significantly affects results

## Research References

1. **Entropy-Based Decision Trees** (M.Ulqinaku, A.Ktona)
   - Overview of Shannon's entropy
   - ID3, C4.5, C5.0 algorithms

2. **Analysis of Depth of Entropy and GINI Index Based Decision Trees for Predicting Diabetes**
   - Depth analysis for diabetes prediction
   - Comparison of entropy and GINI index
