"""
install dependencies:
pip install -r requirements.txt

run gui.py to start the application:
python gui.py

AUTHORS: Kamil Suchomski and Kamil Koniak

PROBLEM: The task was to create a recommendation engine for movies and series based on the data collected from survey.

SOLUTION: We used Item based Collaborative Filtering with Cosine Similarity algorithm to create the recommendation engine.
We also used IMDb API to get the data about the movies and series, to display the posters, ratings, genres, duration, and descriptions.
"""


"""Helper functions for the recommendation system."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class DatasetSplit:
    """Container for train/test dataset splits."""
    train: pd.DataFrame
    test: pd.DataFrame


def load_datasets(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load movies and ratings CSVs located under the given base directory."""
    movies = pd.read_csv(base_dir / "data" / "movies.csv")
    ratings = pd.read_csv(base_dir / "data" / "ratings.csv")
    return movies, ratings


def stratified_user_split(
    ratings: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42
) -> DatasetSplit:
    """Perform an 80/20-like split for each user preserving at least one rating per side."""
    rng = np.random.default_rng(random_state)
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for user_id, group in ratings.groupby("userId"):
        user_ratings = group.sort_values("movieId")  # deterministic order before sampling
        n_ratings = len(user_ratings)

        if n_ratings < 2:
            # not enough data to split, send all to train
            train_parts.append(user_ratings)
            continue

        n_test = max(1, int(round(n_ratings * test_ratio)))
        n_test = min(n_test, n_ratings - 1)  # keep at least one in train
        test_indices = rng.choice(n_ratings, size=n_test, replace=False)
        mask = np.zeros(n_ratings, dtype=bool)
        mask[test_indices] = True

        test_parts.append(user_ratings.iloc[mask])
        train_parts.append(user_ratings.iloc[~mask])

    train_df = pd.concat(train_parts).sort_values(["userId", "movieId"]).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(["userId", "movieId"]).reset_index(drop=True)
    return DatasetSplit(train=train_df, test=test_df)


def evaluate_model(model, test_ratings: pd.DataFrame) -> Dict[str, float]:
    """Compute MAE and RMSE on the held-out test set."""
    if test_ratings.empty:
        return {"rmse": float("nan"), "mae": float("nan")}

    predictions: List[float] = []
    truths: List[float] = []

    for row in test_ratings.itertuples(index=False):
        pred = model.predict(int(row.userId), int(row.movieId))
        predictions.append(pred)
        truths.append(float(row.rating))

    rmse = math.sqrt(mean_squared_error(truths, predictions))
    mae = mean_absolute_error(truths, predictions)
    return {"rmse": rmse, "mae": mae}


def format_recommendations_with_metadata(
    recs: Dict[str, List[Tuple[int, float]]],
    movie_lookup: Dict[int, str],
    imdb_lookup: Dict[int, str],
) -> Dict[str, List[Dict[str, object]]]:
    """Convert movieId recommendations into structures containing title, imdbId, and score."""
    formatted: Dict[str, List[Dict[str, object]]] = {}
    for key, items in recs.items():
        formatted[key] = [
            {
                "movieId": mid,
                "title": movie_lookup.get(mid, f"Movie {mid}"),
                "imdbId": imdb_lookup.get(mid, "N/A"),
                "score": score,
            }
            for mid, score in items
        ]
    return formatted


def plot_input_data(movies: pd.DataFrame, ratings: pd.DataFrame, split: DatasetSplit) -> None:
    """
    Visualize input data: ratings distribution, movies by genre/type/production, ratings per user.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Input Data Analysis", fontsize=16, fontweight="bold")
    
    # 1. Rating distribution
    axes[0, 0].hist(ratings["rating"], bins=10, edgecolor="black", alpha=0.7, color="skyblue")
    axes[0, 0].set_xlabel("Rating")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Rating Distribution")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Ratings per user
    ratings_per_user = ratings.groupby("userId")["rating"].count()
    axes[0, 1].bar(range(len(ratings_per_user)), ratings_per_user.values, color="lightcoral")
    axes[0, 1].set_xlabel("User ID")
    axes[0, 1].set_ylabel("Number of Ratings")
    axes[0, 1].set_title("Ratings per User")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    # 3. Movies by type
    if "type" in movies.columns:
        type_counts = movies["type"].value_counts()
        axes[0, 2].pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%", startangle=90)
        axes[0, 2].set_title("Movies by Type")
    
    # 4. Top genres (extract from pipe-separated genres)
    genre_col = None
    for col in ["genres", "generes"]:
        if col in movies.columns:
            genre_col = col
            break
    
    if genre_col:
        all_genres = []
        for genre_str in movies[genre_col].fillna(""):
            all_genres.extend([g.strip() for g in str(genre_str).split("|") if g.strip()])
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        axes[1, 0].barh(range(len(genre_counts)), genre_counts.values, color="lightgreen")
        axes[1, 0].set_yticks(range(len(genre_counts)))
        axes[1, 0].set_yticklabels(genre_counts.index)
        axes[1, 0].set_xlabel("Count")
        axes[1, 0].set_title("Top 10 Genres")
        axes[1, 0].grid(True, alpha=0.3, axis="x")
    
    # 5. Movies by production country (top 10)
    if "production" in movies.columns:
        prod_counts = movies["production"].value_counts().head(10)
        axes[1, 1].bar(range(len(prod_counts)), prod_counts.values, color="plum")
        axes[1, 1].set_xticks(range(len(prod_counts)))
        axes[1, 1].set_xticklabels(prod_counts.index, rotation=45, ha="right")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Top 10 Production Countries")
        axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    # 6. Train/Test split visualization
    train_size = len(split.train)
    test_size = len(split.test)
    axes[1, 2].bar(["Train", "Test"], [train_size, test_size], color=["steelblue", "orange"])
    axes[1, 2].set_ylabel("Number of Ratings")
    axes[1, 2].set_title(f"Train/Test Split\n({train_size} / {test_size})")
    axes[1, 2].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("lab3/input_data_analysis.png", dpi=150, bbox_inches="tight")
    print("Input data plots saved to: lab3/input_data_analysis.png")
    plt.close()


def plot_output_data(
    model, test_ratings: pd.DataFrame, metrics: Dict[str, float], recommendations: Dict
) -> None:
    """
    Visualize output data: prediction accuracy, recommendation scores, error distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Output Data Analysis", fontsize=16, fontweight="bold")
    
    # 1. Prediction vs Actual (scatter plot)
    predictions = []
    actuals = []
    for row in test_ratings.itertuples(index=False):
        pred = model.predict(int(row.userId), int(row.movieId))
        predictions.append(pred)
        actuals.append(float(row.rating))
    
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, color="steelblue", s=30)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    axes[0, 0].set_xlabel("Actual Rating")
    axes[0, 0].set_ylabel("Predicted Rating")
    axes[0, 0].set_title(f"Prediction vs Actual\nRMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    errors = [abs(a - p) for a, p in zip(actuals, predictions)]
    axes[0, 1].hist(errors, bins=15, edgecolor="black", alpha=0.7, color="coral")
    axes[0, 1].set_xlabel("Absolute Error")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Prediction Error Distribution")
    axes[0, 1].axvline(np.mean(errors), color="red", linestyle="--", lw=2, label=f"Mean: {np.mean(errors):.2f}")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Recommendation scores distribution
    all_rec_scores: List[float] = []
    all_anti_scores: List[float] = []
    for user_recs in recommendations.values():
        if "recommendations" in user_recs:
            all_rec_scores.extend([item["score"] for item in user_recs["recommendations"]])
        if "anti_recommendations" in user_recs:
            all_anti_scores.extend([item["score"] for item in user_recs["anti_recommendations"]])
    
    if all_rec_scores and all_anti_scores:
        axes[1, 0].hist(all_rec_scores, bins=15, alpha=0.6, label="Recommendations", color="green", edgecolor="black")
        axes[1, 0].hist(all_anti_scores, bins=15, alpha=0.6, label="Anti-recommendations", color="red", edgecolor="black")
        axes[1, 0].set_xlabel("Predicted Score")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Recommendation Scores Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Prediction accuracy by rating value
    rating_bins = {}
    for actual, pred in zip(actuals, predictions):
        rating = int(actual)
        if rating not in rating_bins:
            rating_bins[rating] = []
        rating_bins[rating].append(abs(actual - pred))
    
    if rating_bins:
        bin_labels = sorted(rating_bins.keys())
        bin_errors = [np.mean(rating_bins[r]) for r in bin_labels]
        axes[1, 1].bar(bin_labels, bin_errors, color="mediumpurple", edgecolor="black")
        axes[1, 1].set_xlabel("Actual Rating")
        axes[1, 1].set_ylabel("Mean Absolute Error")
        axes[1, 1].set_title("Prediction Error by Rating Value")
        axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("lab3/output_data_analysis.png", dpi=150, bbox_inches="tight")
    print("Output data plots saved to: lab3/output_data_analysis.png")
    plt.close()

