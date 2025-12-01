"""Command-line prediction tool for Banknote Authentication.

Allows predicting a single banknote as authentic/fake using either
Decision Tree or SVM models trained on the full dataset pipeline.

Usage examples (from repository root `lab4/`):

    # Decision Tree (entropy)
    python predict_banknote.py --model dt \
        --variance 3.6216 --skewness 8.6661 --curtosis -2.8073 --entropy -0.44699

    # Decision Tree (gini)
    python predict_banknote.py --model dt --criterion gini \
        --variance 3.6216 --skewness 8.6661 --curtosis -2.8073 --entropy -0.44699

    # SVM (RBF kernel)
    python predict_banknote.py --model svm --kernel rbf \
        --variance 3.6216 --skewness 8.6661 --curtosis -2.8073 --entropy -0.44699
"""

import argparse

from data.dataset1loader import load_banknote_dataset
from decision_tree.decision_tree_analysis import DecisionTreeAnalyzer
from support_vector_machine.svm_analysis import SVMAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for banknote prediction."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict Banknote Authentication (authentic vs fake) "
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

    # Feature values for banknote dataset
    parser.add_argument("--variance", type=float, required=True, help="Variance of Wavelet Transformed image")
    parser.add_argument("--skewness", type=float, required=True, help="Skewness of Wavelet Transformed image")
    parser.add_argument("--curtosis", type=float, required=True, help="Curtosis of Wavelet Transformed image")
    parser.add_argument("--entropy", type=float, required=True, help="Entropy of image")

    return parser.parse_args()


def build_sample_dict(args: argparse.Namespace) -> dict:
    """Build a feature dictionary in the correct order for the analyzer."""
    return {
        "Variance": args.variance,
        "Skewness": args.skewness,
        "Curtosis": args.curtosis,
        "Entropy": args.entropy,
    }


def run_decision_tree_prediction(sample: dict, criterion: str) -> None:
    """Train Decision Tree models and print prediction for the given sample."""
    data = load_banknote_dataset()

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

    # Minimal processing: we only need models and comparison to select best depth
    analyzer.load_and_prepare_data()
    analyzer.train_decision_trees()
    comparison_df = analyzer.analyze_depth_impact()

    result = analyzer.predict_sample(sample_data=sample, criterion=criterion, comparison_df=comparison_df)

    print("\n" + "=" * 60)
    print("DECISION TREE PREDICTION (Banknote Authentication)")
    print("=" * 60)
    print(f"Input features: {sample}")

    label = "authentic" if result["predicted_class"] == 0 else "fake"
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
    data = load_banknote_dataset()

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
    print("SVM PREDICTION (Banknote Authentication)")
    print("=" * 60)
    print(f"Input features: {sample}")

    label = "authentic" if result["predicted_class"] == 0 else "fake"
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
    """Entry point for the banknote prediction CLI."""
    args = parse_args()
    sample = build_sample_dict(args)

    if args.model == "dt":
        run_decision_tree_prediction(sample, criterion=args.criterion)
    else:
        run_svm_prediction(sample, kernel=args.kernel)


if __name__ == "__main__":
    main()
