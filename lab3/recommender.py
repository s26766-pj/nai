"""
install dependencies:
pip install -r requirements.txt

run gui.py to start the application:
python gui.py

AUTHORS: Kamil Suchomski and Kamil Koniak

PROBLEM: The task was to create a recommendation engine for movies and series based on the data collected from survey.

SOLUTION: We used Item based Collaborative Filtering with Cosine Similarity algorithm to create the recommendation engine.
We used content-based features (genres, type, production) to improve the recommendations.
We also used IMDb API to get the data about the movies and series, to display the posters, ratings, genres, duration, and descriptions.
We are showing the recommendations for each user in the GUI (TKinter).
"""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from helper import (
    DatasetSplit,
    evaluate_model,
    format_recommendations_with_metadata,
    load_datasets,
    plot_input_data,
    plot_output_data,
    stratified_user_split,
)

# https://en.wikipedia.org/wiki/Collaborative_filtering
# https://en.wikipedia.org/wiki/Cosine_similarity
class ItemBasedCollaborativeFilteringWithCosineSimilarity:
    """Hybrid item-based collaborative filtering with content-based features (genres, type, production)."""

    def __init__(
        self,
        top_k_similar: int = 30,
        min_similarity: float = 0.01,
        clip_range: Tuple[float, float] = (1.0, 10.0),
        content_weight: float = 0.3,
    ) -> None:
        self.top_k_similar = top_k_similar
        self.min_similarity = min_similarity
        self.clip_range = clip_range
        self.content_weight = content_weight  # Weight for content-based similarity (genres, type, production)

        self.user_means: Dict[int, float] = {}
        self.global_mean: float = math.nan
        self.user_item_matrix: pd.DataFrame | None = None
        self.item_similarity: pd.DataFrame | None = None
        self.fitted: bool = False

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> None:
        """Fit the model on ratings and optionally movies (for content-based features)."""
        if ratings.empty:
            raise ValueError("Cannot fit ItemBasedCollaborativeFilteringWithCosineSimilarity on an empty ratings dataframe.")

        self.global_mean = float(ratings["rating"].mean())
        self.user_means = ratings.groupby("userId")["rating"].mean().to_dict()

        centered = ratings.copy()
        centered["rating_centered"] = centered["rating"] - centered["userId"].map(self.user_means)

        user_item = (
            centered.pivot(index="userId", columns="movieId", values="rating_centered")
            .fillna(0.0)
            .sort_index()
        )
        self.user_item_matrix = user_item

        # Compute rating-based cosine similarity between items
        item_matrix = user_item.to_numpy().T
        rating_similarity = cosine_similarity(item_matrix)
        movie_ids = user_item.columns

        # Compute content-based similarity if movies provided
        if movies is not None and self.content_weight > 0.0:
            content_similarity = self._compute_content_similarity(movies, movie_ids)
            
            # Combine rating-based and content-based similarities
            rating_weight = 1.0 - self.content_weight
            hybrid_similarity = (
                rating_weight * rating_similarity +
                self.content_weight * content_similarity.values
            )
            self.item_similarity = pd.DataFrame(hybrid_similarity, index=movie_ids, columns=movie_ids)
        else:
            # Pure collaborative filtering (no content features)
            self.item_similarity = pd.DataFrame(rating_similarity, index=movie_ids, columns=movie_ids)
        
        self.fitted = True

    def _compute_content_similarity(
        self, movies: pd.DataFrame, movie_ids: pd.Index
    ) -> pd.DataFrame:
        """
        Compute content-based similarity using genres, type, and production.
        
        Creates feature vectors for each movie:
        - Genres: multi-hot encoding (pipe-separated)
        - Type: one-hot encoding (movie/series)
        - Production: one-hot encoding (country codes)
        
        Then computes cosine similarity between feature vectors.
        """
        # Filter movies to only those in rating matrix
        movies_filtered = movies[movies["movieId"].isin(movie_ids)].copy()
        
        # Get all unique genres
        genre_col = None
        for col in ["genres", "generes"]:
            if col in movies_filtered.columns:
                genre_col = col
                break
        
        if genre_col is None:
            # No genres: return identity matrix
            n = len(movie_ids)
            return pd.DataFrame(np.eye(n), index=movie_ids, columns=movie_ids)
        
        # Extract genres (pipe-separated)
        genre_lists = movies_filtered[genre_col].fillna("").str.split("|")
        all_genres = set()
        for genre_list in genre_lists:
            all_genres.update([g.strip() for g in genre_list if g.strip()])
        all_genres = sorted(all_genres)
        
        # Get unique types and production countries
        all_types = sorted(movies_filtered["type"].dropna().unique()) if "type" in movies_filtered.columns else []
        all_productions = sorted(movies_filtered["production"].dropna().unique()) if "production" in movies_filtered.columns else []
        
        # Create feature matrix
        n_features = len(all_genres) + len(all_types) + len(all_productions)
        feature_matrix = np.zeros((len(movies_filtered), n_features), dtype=float)
        
        # Use enumerate to get sequential index (0, 1, 2, ...) instead of DataFrame index
        for row_idx, (_, row) in enumerate(movies_filtered.iterrows()):
            feature_idx = 0
            
            # Genre features (multi-hot)
            movie_genres = set()
            if pd.notna(row[genre_col]):
                movie_genres = set([g.strip() for g in str(row[genre_col]).split("|") if g.strip()])
            for genre in all_genres:
                feature_matrix[row_idx, feature_idx] = 1.0 if genre in movie_genres else 0.0
                feature_idx += 1
            
            # Type feature (one-hot)
            if "type" in movies_filtered.columns and pd.notna(row["type"]):
                for type_val in all_types:
                    feature_matrix[row_idx, feature_idx] = 1.0 if row["type"] == type_val else 0.0
                    feature_idx += 1
            
            # Production feature (one-hot)
            if "production" in movies_filtered.columns and pd.notna(row["production"]):
                for prod_val in all_productions:
                    feature_matrix[row_idx, feature_idx] = 1.0 if row["production"] == prod_val else 0.0
                    feature_idx += 1
        
        # Compute cosine similarity
        content_sim = cosine_similarity(feature_matrix)
        
        # Create DataFrame aligned with movie_ids
        content_df = pd.DataFrame(
            content_sim,
            index=movies_filtered["movieId"].values,
            columns=movies_filtered["movieId"].values,
        )
        
        # Reindex to match movie_ids order
        content_df = content_df.reindex(index=movie_ids, columns=movie_ids, fill_value=0.0)
        
        return content_df

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-item pair."""
        self._ensure_fitted()
        assert self.user_item_matrix is not None
        assert self.item_similarity is not None

        if user_id not in self.user_item_matrix.index:
            baseline = self.global_mean if not math.isnan(self.global_mean) else 0.0
            return self._clip(baseline)

        if movie_id not in self.item_similarity.index:
            baseline = self.user_means.get(user_id, self.global_mean)
            return self._clip(baseline)

        user_vector = self.user_item_matrix.loc[user_id]
        rated_mask = user_vector != 0.0
        if not rated_mask.any():
            baseline = self.user_means.get(user_id, self.global_mean)
            return self._clip(baseline)

        rated_items = user_vector.index[rated_mask]
        similarities = self.item_similarity.loc[movie_id, rated_items]

        # Filter by minimum similarity threshold
        if self.min_similarity > 0.0:
            keep_mask = similarities.abs() >= self.min_similarity
            similarities = similarities[keep_mask]
            rated_items = rated_items[keep_mask]

        if similarities.empty:
            baseline = self.user_means.get(user_id, self.global_mean)
            return self._clip(baseline)

        if self.top_k_similar and len(similarities) > self.top_k_similar:
            top_indices = similarities.abs().sort_values(ascending=False).index[: self.top_k_similar]
            similarities = similarities[top_indices]
            rated_items = top_indices

        ratings_centered = user_vector.loc[rated_items].to_numpy()
        sims = similarities.to_numpy()

        numerator = np.dot(sims, ratings_centered)
        denominator = np.abs(sims).sum()
        baseline = self.user_means.get(user_id, self.global_mean)

        if denominator == 0.0 or np.isnan(numerator):
            return self._clip(baseline)

        prediction = baseline + numerator / denominator
        return self._clip(prediction)

    def recommend(
        self,
        user_id: int,
        seen_items: Iterable[int],
        n_recommendations: int = 5,
        n_anti_recommendations: int = 5,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Return top/bottom-N recommendations for unseen items."""
        self._ensure_fitted()
        assert self.item_similarity is not None

        seen_set = set(seen_items)
        candidate_ids = [mid for mid in self.item_similarity.index if mid not in seen_set]

        scores: List[Tuple[int, float]] = []
        for movie_id in candidate_ids:
            score = self.predict(user_id, movie_id)
            scores.append((movie_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = scores[:n_recommendations]
        anti_recommendations = sorted(scores, key=lambda x: x[1])[:n_anti_recommendations]

        return {
            "recommendations": recommendations,
            "anti_recommendations": anti_recommendations,
        }

    def _clip(self, value: float) -> float:
        low, high = self.clip_range
        if low is None or high is None:
            return value
        return float(np.clip(value, low, high))

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("ItemBasedCollaborativeFilteringWithCosineSimilarity must be fitted before usage.")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    movies, ratings = load_datasets(base_dir)
    users_df = pd.read_csv(base_dir / "data" / "users.csv")
    links_df = pd.read_csv(base_dir / "data" / "links.csv")
    user_lookup = users_df.set_index("userId")["userName"].to_dict()
    imdb_lookup = links_df.set_index("movieId")["imdbId"].to_dict()
    split = stratified_user_split(ratings, test_ratio=0.2, random_state=42)

    # Plot input data analysis
    print("Generating input data visualizations...")
    plot_input_data(movies, ratings, split)
    print()

    # Initialize model with content_weight=0.3 (30% content features, 70% ratings)
    model = ItemBasedCollaborativeFilteringWithCosineSimilarity(
        top_k_similar=25, min_similarity=0.05, content_weight=0.3
    )
    # Pass both ratings and movies to enable content-based features (genres, type, production)
    model.fit(split.train, movies=movies)

    metrics = evaluate_model(model, split.test)
    print("Evaluation metrics (80/20 split):")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE : {metrics['mae']:.4f}")
    print()

    # Generate recommendations for all users
    movie_lookup = movies.set_index("movieId")["title"].to_dict()
    users = sorted(ratings["userId"].unique())
    all_recommendations: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    
    for user_id in users:
        user_name = user_lookup.get(user_id, f"User {user_id}")
        seen_items = ratings.loc[ratings["userId"] == user_id, "movieId"]
        recs = model.recommend(
            user_id=user_id,
            seen_items=seen_items,
            n_recommendations=5,
            n_anti_recommendations=5,
        )
        formatted = format_recommendations_with_metadata(recs, movie_lookup, imdb_lookup)
        all_recommendations[user_name] = formatted
        print(f"{user_name} recommendations:")
        for item in formatted["recommendations"]:
            print(f"  [LIKE] {item['title']} (IMDB: {item['imdbId']}, predicted {item['score']:.2f})")
        for item in formatted["anti_recommendations"]:
            print(f"  [DISLIKE] {item['title']} (IMDB: {item['imdbId']}, predicted {item['score']:.2f})")
        print()

    # Plot output data analysis
    print("Generating output data visualizations...")
    plot_output_data(model, split.test, metrics, all_recommendations)
    print()


if __name__ == "__main__":
    main()

