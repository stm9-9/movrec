"""Improved CBF (v2): separate genre one-hot + overview character n-gram TF-IDF.

Compared against CBF v1 (whitespace TF-IDF over combined text) as a baseline.
"""
import json

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def load_genre_ids(movies):
    """List of genre-id sets, aligned with `movies` row order."""
    sets = []
    for g in movies["genre_ids"]:
        g = (g or "").strip()
        if g in ("", "[]"):
            sets.append(set())
            continue
        try:
            sets.append(set(int(x) for x in json.loads(g)))
        except (ValueError, TypeError, json.JSONDecodeError):
            sets.append(set())
    return sets


def build_genre_onehot(movies, genre_ids_sorted):
    """Binary multi-hot genre matrix (n x k), L2-normalized per row."""
    genre_sets = load_genre_ids(movies)
    idx = {g: i for i, g in enumerate(genre_ids_sorted)}
    rows, cols = [], []
    for r, gs in enumerate(genre_sets):
        for g in gs:
            if g in idx:
                rows.append(r)
                cols.append(idx[g])
    data = np.ones(len(rows), dtype=np.float64)
    X = sp.csr_matrix((data, (rows, cols)), shape=(len(movies), len(idx)))
    return normalize(X, norm="l2")


def build_overview_tfidf(movies, max_features=5000, analyzer="char", ngram_range=(3, 5)):
    """Character n-gram TF-IDF over the overview text (Thai-safe, no segmenter)."""
    overview = movies["overview"].fillna("").astype(str)
    vec = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram_range, max_features=max_features
    )
    return vec, vec.fit_transform(overview)


class TopKSimilarity:
    """Brute-force cosine kNN over a sparse feature matrix."""

    def __init__(self, feature_matrix, ids):
        self.feature_matrix = feature_matrix
        self.ids = [int(i) for i in ids]
        self.id_to_index = {i: k for k, i in enumerate(self.ids)}
        self.nn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute").fit(
            feature_matrix
        )

    def similar(self, movie_id, top_n=10):
        """Top-n similar movies, excluding the source itself."""
        idx = self.id_to_index[int(movie_id)]
        dists, indices = self.nn.kneighbors(
            self.feature_matrix[idx], n_neighbors=top_n + 1
        )
        out = []
        for d, i in zip(dists[0], indices[0]):
            if int(self.ids[i]) == int(movie_id):
                continue
            out.append({"movie_id": int(self.ids[i]), "similarity": round(float(1.0 - d), 6)})
            if len(out) >= top_n:
                break
        return out


def genre_jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def content_cosine(content_tfidf, id_to_index, src_id, rec_id):
    """Cosine similarity of overview char n-gram TF-IDF vectors (L2-normalized)."""
    i = id_to_index[int(src_id)]
    j = id_to_index[int(rec_id)]
    return float(content_tfidf[i].dot(content_tfidf[j].T).toarray()[0, 0])
