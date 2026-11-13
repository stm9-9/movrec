"""Auxiliary K-Means user clustering (K=2) per thesis Ch.3.

This is reported as a descriptive grouping of users and is NOT used by SVD.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_users(ratings_long, n_clusters=2, random_state=42):
    """Cluster users on their (user-mean-imputed, standardized) rating vectors."""
    pivot = ratings_long.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    )
    # Impute missing ratings with the user's own mean rating.
    pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(pivot.values)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    result = pd.DataFrame({"user_id": pivot.index.tolist(), "cluster": labels})
    return result, km, pivot
