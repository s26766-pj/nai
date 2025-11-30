"""
Main Entry Point: Decision Tree Analysis
Orchestrates the complete analysis pipeline for both datasets
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Import dataset loaders
from data.dataset1loader import load_banknote_dataset
from data.dataset2loader import load_diabetes_dataset

# Import analysis modules
from decision_tree.decision_tree_analysis import DecisionTreeAnalyzer
from support_vector_machine.svm_analysis import SVMAnalyzer


def main():
    """Main analysis function"""
    # Create report folder if it doesn't exist
    report_dir = 'report'
    os.makedirs(report_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("DECISION TREE ANALYSIS: ENTROPY vs GINI INDEX")
    print("="*80)
    
    # Dataset 1: Banknote Authentication
    print("\n" + "="*80)
    print("DATASET 1: Banknote Authentication")
    print("="*80)
    
    data1 = load_banknote_dataset()
    analyzer1 = DecisionTreeAnalyzer(
        dataset_name=data1['dataset_name'],
        X=data1['X'],
        y=data1['y'],
        feature_names=data1['feature_names'],
        X_train=data1['X_train'],
        X_test=data1['X_test'],
        y_train=data1['y_train'],
        y_test=data1['y_test']
    )
    analyzer1.load_and_prepare_data()
    
    # Visualize sample data
    print("\n" + "="*60)
    print("SAMPLE DATA VISUALIZATION")
    print("="*60)
    analyzer1.visualize_sample_data()
    
    analyzer1.train_decision_trees()
    comparison1 = analyzer1.analyze_depth_impact()
    analyzer1.plot_analysis(comparison1)
    
    # Display detailed classification metrics
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS - ENTROPY (Best Model)")
    print("="*60)
    analyzer1.display_classification_metrics(criterion='entropy', comparison_df=comparison1)
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS - GINI (Best Model)")
    print("="*60)
    analyzer1.display_classification_metrics(criterion='gini', comparison_df=comparison1)
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    sample_indices = [0, 10, 20]  # Sample from test set
    for idx in sample_indices:
        sample = data1['X_test'][idx]
        print(f"\nSample {idx + 1} (True class: {data1['y_test'][idx]}):")
        print(f"  Features: {dict(zip(data1['feature_names'], sample))}")
        
        pred_entropy = analyzer1.predict_sample(sample, criterion='entropy', comparison_df=comparison1)
        pred_gini = analyzer1.predict_sample(sample, criterion='gini', comparison_df=comparison1)
        
        print(f"  Entropy Model Prediction: Class {pred_entropy['predicted_class']} "
              f"(Probabilities: {pred_entropy['class_probabilities']})")
        print(f"  GINI Model Prediction: Class {pred_gini['predicted_class']} "
              f"(Probabilities: {pred_gini['class_probabilities']})")
    
    analyzer1.generate_report(comparison1)
    
    # Dataset 2: Pima Indians Diabetes
    print("\n" + "="*80)
    print("DATASET 2: Pima Indians Diabetes")
    print("="*80)
    
    data2 = load_diabetes_dataset()
    analyzer2 = DecisionTreeAnalyzer(
        dataset_name=data2['dataset_name'],
        X=data2['X'],
        y=data2['y'],
        feature_names=data2['feature_names'],
        X_train=data2['X_train'],
        X_test=data2['X_test'],
        y_train=data2['y_train'],
        y_test=data2['y_test']
    )
    analyzer2.load_and_prepare_data()
    
    # Visualize sample data
    print("\n" + "="*60)
    print("SAMPLE DATA VISUALIZATION")
    print("="*60)
    analyzer2.visualize_sample_data()
    
    analyzer2.train_decision_trees()
    comparison2 = analyzer2.analyze_depth_impact()
    analyzer2.plot_analysis(comparison2)
    
    # Display detailed classification metrics
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS - ENTROPY (Best Model)")
    print("="*60)
    analyzer2.display_classification_metrics(criterion='entropy', comparison_df=comparison2)
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS - GINI (Best Model)")
    print("="*60)
    analyzer2.display_classification_metrics(criterion='gini', comparison_df=comparison2)
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    sample_indices = [0, 10, 20]  # Sample from test set
    for idx in sample_indices:
        sample = data2['X_test'][idx]
        print(f"\nSample {idx + 1} (True class: {data2['y_test'][idx]}):")
        print(f"  Features: {dict(zip(data2['feature_names'], sample))}")
        
        pred_entropy = analyzer2.predict_sample(sample, criterion='entropy', comparison_df=comparison2)
        pred_gini = analyzer2.predict_sample(sample, criterion='gini', comparison_df=comparison2)
        
        print(f"  Entropy Model Prediction: Class {pred_entropy['predicted_class']} "
              f"(Probabilities: {pred_entropy['class_probabilities']})")
        print(f"  GINI Model Prediction: Class {pred_gini['predicted_class']} "
              f"(Probabilities: {pred_gini['class_probabilities']})")
    
    analyzer2.generate_report(comparison2)
    
    # ========================================================================
    # SVM ANALYSIS
    # ========================================================================
    print("\n\n" + "="*80)
    print("SUPPORT VECTOR MACHINE (SVM) ANALYSIS")
    print("="*80)
    
    # Dataset 1: Banknote Authentication - SVM
    print("\n" + "="*80)
    print("SVM ANALYSIS - DATASET 1: Banknote Authentication")
    print("="*80)
    
    svm_analyzer1 = SVMAnalyzer(
        dataset_name=data1['dataset_name'],
        X=data1['X'],
        y=data1['y'],
        feature_names=data1['feature_names'],
        X_train=data1['X_train'],
        X_test=data1['X_test'],
        y_train=data1['y_train'],
        y_test=data1['y_test'],
        scale_features=True
    )
    svm_analyzer1.load_and_prepare_data()
    svm_analyzer1.train_svm_models()
    summary1 = svm_analyzer1.analyze_kernel_performance()
    svm_analyzer1.plot_analysis(summary1)
    
    # Visualize decision boundary for best kernel
    best_kernel1 = summary1.iloc[0]['Kernel']
    print("\n" + "="*60)
    print(f"VISUALIZING DECISION BOUNDARY - {best_kernel1.upper()} KERNEL")
    print("="*60)
    svm_analyzer1.plot_decision_boundary(kernel_name=best_kernel1)
    
    # Display detailed classification metrics for best kernel
    best_kernel1 = summary1.iloc[0]['Kernel']
    print("\n" + "="*60)
    print(f"DETAILED CLASSIFICATION METRICS - {best_kernel1.upper()} (Best Kernel)")
    print("="*60)
    svm_analyzer1.display_classification_metrics(kernel_name=best_kernel1)
    
    # Sample predictions with SVM
    print("\n" + "="*60)
    print("SVM SAMPLE PREDICTIONS")
    print("="*60)
    sample_indices = [0, 10, 20]
    for idx in sample_indices:
        sample = data1['X_test'][idx]
        print(f"\nSample {idx + 1} (True class: {data1['y_test'][idx]}):")
        print(f"  Features: {dict(zip(data1['feature_names'], sample))}")
        
        pred_rbf = svm_analyzer1.predict_sample(sample, kernel_name='rbf')
        pred_linear = svm_analyzer1.predict_sample(sample, kernel_name='linear')
        
        print(f"  RBF Model Prediction: Class {pred_rbf['predicted_class']} "
              f"(Probabilities: {pred_rbf['class_probabilities']})")
        print(f"  Linear Model Prediction: Class {pred_linear['predicted_class']} "
              f"(Probabilities: {pred_linear['class_probabilities']})")
    
    svm_analyzer1.generate_kernel_summary(summary1)
    
    # Dataset 2: Pima Indians Diabetes - SVM
    print("\n" + "="*80)
    print("SVM ANALYSIS - DATASET 2: Pima Indians Diabetes")
    print("="*80)
    
    svm_analyzer2 = SVMAnalyzer(
        dataset_name=data2['dataset_name'],
        X=data2['X'],
        y=data2['y'],
        feature_names=data2['feature_names'],
        X_train=data2['X_train'],
        X_test=data2['X_test'],
        y_train=data2['y_train'],
        y_test=data2['y_test'],
        scale_features=True
    )
    svm_analyzer2.load_and_prepare_data()
    svm_analyzer2.train_svm_models()
    summary2 = svm_analyzer2.analyze_kernel_performance()
    svm_analyzer2.plot_analysis(summary2)
    
    # Visualize decision boundary for best kernel
    best_kernel2 = summary2.iloc[0]['Kernel']
    print("\n" + "="*60)
    print(f"VISUALIZING DECISION BOUNDARY - {best_kernel2.upper()} KERNEL")
    print("="*60)
    svm_analyzer2.plot_decision_boundary(kernel_name=best_kernel2)
    
    # Display detailed classification metrics for best kernel
    best_kernel2 = summary2.iloc[0]['Kernel']
    print("\n" + "="*60)
    print(f"DETAILED CLASSIFICATION METRICS - {best_kernel2.upper()} (Best Kernel)")
    print("="*60)
    svm_analyzer2.display_classification_metrics(kernel_name=best_kernel2)
    
    # Sample predictions with SVM
    print("\n" + "="*60)
    print("SVM SAMPLE PREDICTIONS")
    print("="*60)
    sample_indices = [0, 10, 20]
    for idx in sample_indices:
        sample = data2['X_test'][idx]
        print(f"\nSample {idx + 1} (True class: {data2['y_test'][idx]}):")
        print(f"  Features: {dict(zip(data2['feature_names'], sample))}")
        
        pred_rbf = svm_analyzer2.predict_sample(sample, kernel_name='rbf')
        pred_linear = svm_analyzer2.predict_sample(sample, kernel_name='linear')
        
        print(f"  RBF Model Prediction: Class {pred_rbf['predicted_class']} "
              f"(Probabilities: {pred_rbf['class_probabilities']})")
        print(f"  Linear Model Prediction: Class {pred_linear['predicted_class']} "
              f"(Probabilities: {pred_linear['class_probabilities']})")
    
    svm_analyzer2.generate_kernel_summary(summary2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files (saved in '{report_dir}/' folder):")
    print("\nDecision Tree Analysis:")
    print(f"  - {report_dir}/Banknote_Authentication_data_overview.png")
    print(f"  - {report_dir}/Banknote_Authentication_analysis.png")
    print(f"  - {report_dir}/Banknote_Authentication_report.txt")
    print(f"  - {report_dir}/Pima_Indians_Diabetes_data_overview.png")
    print(f"  - {report_dir}/Pima_Indians_Diabetes_analysis.png")
    print(f"  - {report_dir}/Pima_Indians_Diabetes_report.txt")
    print("\nSVM Analysis:")
    print(f"  - {report_dir}/Banknote_Authentication_SVM_analysis.png")
    print(f"  - {report_dir}/Banknote_Authentication_SVM_decision_boundary_*.png (2D)")
    print(f"  - {report_dir}/Banknote_Authentication_SVM_decision_boundary_3d_*.png (3D)")
    print(f"  - {report_dir}/Banknote_Authentication_SVM_Kernel_Summary.txt")
    print(f"  - {report_dir}/Pima_Indians_Diabetes_SVM_analysis.png")
    print(f"  - {report_dir}/Pima_Indians_Diabetes_SVM_decision_boundary_*.png (2D)")
    print(f"  - {report_dir}/Pima_Indians_Diabetes_SVM_decision_boundary_3d_*.png (3D)")
    print(f"  - {report_dir}/Pima_Indians_Diabetes_SVM_Kernel_Summary.txt")


if __name__ == "__main__":
    main()
