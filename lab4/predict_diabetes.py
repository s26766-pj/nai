"""Command-line prediction tool for Pima Indians Diabetes.

Allows predicting diabetes (yes/no) for a single patient using
either Decision Tree or SVM models trained on the full dataset pipeline.

Usage examples (from repository root `lab4/`):

    # Decision Tree (entropy)
    python predict_diabetes.py --model dt \
        --pregnancies 6 --glucose 148 --bloodpressure 72 --skinthickness 35 \
        --insulin 0 --bmi 33.6 --diabetespedigreefunction 0.627 --age 50

    # SVM (RBF)
    python predict_diabetes.py --model svm --kernel rbf \
        --pregnancies 6 --glucose 148 --bloodpressure 72 --skinthickness 35 \
        --insulin 0 --bmi 33.6 --diabetespedigreefunction 0.627 --age 50
"""

import argparse

from data.dataset2loader import load_diabetes_dataset
from decision_tree.decision_tree_analysis import DecisionTreeAnalyzer
from support_vector_machine.svm_analysis import SVMAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for diabetes prediction."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict Pima Indians Diabetes (0 = no diabetes, 1 = diabetes) "
            "using Decision Tree or SVM."
        )
    )

    parser.add_argument(
        "--model",
        choices=["dt", "svm"],
        required=True,
        help="Model type to use: 'dt' for Decision Tree, 'svm' for Support Vector Machine.",
    )

    # Decision Tree options
    parser.add_argument(
        "--criterion",
        choices=["entropy", "gini"],
        default="entropy",
        help=(
            "Decision Tree splitting criterion when --model=dt. "
            "Ignored for SVM. Default: entropy."
        ),
    )

    # SVM options
    parser.add_argument(
        "--kernel",
        choices=["linear", "poly", "rbf", "sigmoid"],
        default="rbf",
        help=(
            "SVM kernel to use when --model=svm. "
            "Ignored for Decision Tree. Default: rbf."
        ),
    )

    # Feature values for Pima Indians Diabetes
    parser.add_argument("--pregnancies", type=float, required=True, help="Number of times pregnant")
    parser.add_argument("--glucose", type=float, required=True, help="Plasma glucose concentration")
    parser.add_argument("--bloodpressure", type=float, required=True, help="Diastolic blood pressure")
    parser.add_argument("--skinthickness", type=float, required=True, help="Triceps skinfold thickness")
    parser.add_argument("--insulin", type=float, required=True, help="2-Hour serum insulin")
    parser.add_argument("--bmi", type=float, required=True, help="Body mass index")
    parser.add_argument(
        "--diabetespedigreefunction",
        type=float,
        required=True,
        help="Diabetes pedigree function",
    )
    parser.add_argument("--age", type=float, required=True, help="Age in years")

    return parser.parse_args()


def build_sample_dict(args: argparse.Namespace) -> dict:
    """Build a feature dictionary in the correct order for the analyzer."""
    return {
        "Pregnancies": args.pregnancies,
        "Glucose": args.glucose,
        "BloodPressure": args.bloodpressure,
        "SkinThickness": args.skinthickness,
        "Insulin": args.insulin,
        "BMI": args.bmi,
        "DiabetesPedigreeFunction": args.diabetespedigreefunction,
        "Age": args.age,
    }


def run_decision_tree_prediction(sample: dict, criterion: str) -> None:
    """Train Decision Tree models and print prediction for the given sample."""
    data = load_diabetes_dataset()

    analyzer = DecisionTreeAnalyzer(
        dataset_name=data["dataset_name"],
        X=data["X"],
        y=data["y"],
        feature_names=data["feature_names"],
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
    )

    analyzer.load_and_prepare_data()
    analyzer.train_decision_trees()
    comparison_df = analyzer.analyze_depth_impact()

    result = analyzer.predict_sample(sample_data=sample, criterion=criterion, comparison_df=comparison_df)

    print("\n" + "=" * 60)
    print("DECISION TREE PREDICTION (Pima Indians Diabetes)")
    print("=" * 60)
    print(f"Input features: {sample}")

    label = "no_diabetes" if result["predicted_class"] == 0 else "diabetes"
    print(f"Predicted class: {result['predicted_class']} ({label})")
    print("Class probabilities:")
    for cls, prob in result["class_probabilities"].items():
        print(f"  {cls}: {prob:.4f}")

    info = result["model_info"]
    print("\nModel info:")
    print(f"  Criterion: {info['criterion']}")
    print(f"  Depth key: {info['depth_key']}")
    print(f"  Test accuracy (for this model): {info['accuracy']:.4f}")


def run_svm_prediction(sample: dict, kernel: str) -> None:
    """Train SVM models and print prediction for the given sample."""
    data = load_diabetes_dataset()

    analyzer = SVMAnalyzer(
        dataset_name=data["dataset_name"],
        X=data["X"],
        y=data["y"],
        feature_names=data["feature_names"],
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        scale_features=True,
    )

    analyzer.load_and_prepare_data()
    analyzer.train_svm_models()

    result = analyzer.predict_sample(sample_data=sample, kernel_name=kernel)

    print("\n" + "=" * 60)
    print("SVM PREDICTION (Pima Indians Diabetes)")
    print("=" * 60)
    print(f"Input features: {sample}")

    label = "no_diabetes" if result["predicted_class"] == 0 else "diabetes"
    print(f"Predicted class: {result['predicted_class']} ({label})")
    print("Class probabilities:")
    for cls, prob in result["class_probabilities"].items():
        print(f"  {cls}: {prob:.4f}")

    info = result["model_info"]
    print("\nModel info:")
    print(f"  Kernel: {info['kernel']}")
    print(f"  Config key: {info['config_key']}")
    print(f"  Config: {info['config']}")
    print(f"  Test accuracy (for this model): {info['accuracy']:.4f}")


def main() -> None:
    """Entry point for the diabetes prediction CLI."""
    args = parse_args()
    sample = build_sample_dict(args)

    if args.model == "dt":
        run_decision_tree_prediction(sample, criterion=args.criterion)
    else:
        run_svm_prediction(sample, kernel=args.kernel)


if __name__ == "__main__":
    main()
