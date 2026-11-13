"""Content-Based Filtering model.

Feature text = Thai genre names (genres_text) + overview.
- TF-IDF vectorization (TfidfVectorizer, max_features=5000).
- Item-item similarity via cosine distance on the TF-IDF vectors, retrieved
  with a brute-force NearestNeighbors index (top-k) so the full dense
  9841x9841 matrix is never materialized.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from recommender.preprocessing import config


def build_feature_text(df):
    """Combine genres_text + overview, safely handling missing/invalid values."""
    genres = df["genres_text"].fillna("").astype(str)
    overview = df["overview"].fillna("").astype(str)
    return (genres + " " + overview).str.strip()


class ContentBasedModel:
    def __init__(self, movies, max_features=5000):
        self.movies = movies.reset_index(drop=True)
        texts = build_feature_text(self.movies)
        valid = texts.str.len() > 0
        self.movies = self.movies[valid].reset_index(drop=True)
        self.texts = texts[valid].reset_index(drop=True)
        self.max_features = max_features

        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf = self.vectorizer.fit_transform(self.texts)
        self.nn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute").fit(
            self.tfidf
        )

        self.ids = self.movies["id"].tolist()
        self.id_to_index = {int(mid): i for i, mid in enumerate(self.ids)}
        self.n_dropped_empty = int((~valid).sum())

    @property
    def n_movies(self):
        return len(self.ids)

    @property
    def n_features(self):
        return int(self.tfidf.shape[1])

    def matrix_dimensions(self):
        """Return (tfidf_rows, tfidf_cols, pairwise_n, pairwise_n)."""
        n = self.n_movies
        return int(self.tfidf.shape[0]), int(self.tfidf.shape[1]), n, n

    def similar_movies(self, movie_id, top_n=10):
        """Top-n most similar movies to `movie_id`, excluding the source itself."""
        if int(movie_id) not in self.id_to_index:
            raise KeyError(f"movie_id {movie_id} not in the CBF model")
        idx = self.id_to_index[int(movie_id)]
        dists, indices = self.nn.kneighbors(self.tfidf[idx], n_neighbors=top_n + 1)
        results = []
        for d, i in zip(dists[0], indices[0]):
            if int(self.ids[i]) == int(movie_id):
                continue
            results.append({
                "source_movie_id": int(movie_id),
                "movie_id": int(self.ids[i]),
                "title": self.movies.iloc[i]["title"],
                "genres_text": self.movies.iloc[i]["genres_text"],
                "similarity": round(float(1.0 - d), 6),
            })
            if len(results) >= top_n:
                break
        return results

    def save(self, path):
        """Persist vectorizer + tfidf + ids (+ nn) for reuse by hybrid/backend."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "tfidf": self.tfidf,
                "ids": self.ids,
                "nn": self.nn,
                "max_features": self.max_features,
            },
            path,
        )
        return path
