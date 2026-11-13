"""Model-serving inference layer for the Streamlit webapp (Stage 5.1).

DESIGN NOTE
-----------
- CF  : item-item cosine approximation on SVD B's item factors (q_i). It is used
  because a new webapp user has no fitted user factor p_u (Surprise SVD predict()
  needs p_u, which exists only for the 180 training users).
- CBF : the full CBF v2 feature matrix [w·genre_onehot | overview_tfidf] is used, but
  the cold-start query is genre-only (overview block = 0), so scores are genre-driven.
- Hybrid therefore APPROXIMATES, but does not exactly reproduce, Stage 4's Hybrid A
  (Stage 4 used Surprise SVD predict() for existing users and a rating-weighted CBF
  profile over the full feature matrix).

Wraps the three scoring sources behind one reusable API so the UI never touches raw
model objects:
  - CF  : SVD B — item-item cosine on item factors (CF approximation for new users)
  - CBF : CBF v2 — full genre+overview feature matrix (genre-only query at cold start)
  - Popularity : train mean rating (the 15 survey movies)

All user-facing text lives in the UI layer, not here.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommender.collaborative.svd_model import SVDRecommender  # noqa: E402
from recommender.hybrid.hybrid_model import minmax_normalize  # noqa: E402

MOVIES_CLEAN = ROOT / "data" / "processed" / "movies_clean.csv"
SVD_PATH = ROOT / "data" / "processed" / "svd_model.joblib"
CBF_V2_PATH = ROOT / "data" / "processed" / "cbf_model_v2.joblib"
POP_PATH = ROOT / "data" / "processed" / "movie_popularity_train.csv"
GENRES_JSON = ROOT / "data" / "raw" / "genres.json"

DEFAULT_ALPHA = 0.3  # Hybrid A (selected in Stage 4)


@lru_cache(maxsize=1)
def _movies():
    return pd.read_csv(MOVIES_CLEAN, keep_default_na=False)


@lru_cache(maxsize=1)
def _svd():
    return SVDRecommender.load(SVD_PATH)


@lru_cache(maxsize=1)
def _cbf():
    return joblib.load(CBF_V2_PATH)


@lru_cache(maxsize=1)
def _pop():
    return pd.read_csv(POP_PATH)


@lru_cache(maxsize=1)
def _popularity_map():
    """movie_id -> TMDB popularity (secondary sort key for deterministic ties)."""
    m = _movies()
    return {int(r.id): float(r.popularity) for r in m.itertuples()}


@lru_cache(maxsize=1)
def _genres():
    with open(GENRES_JSON, encoding="utf-8") as f:
        return json.load(f)


def all_movies():
    """All catalog movie ids (for CBF / popularity)."""
    return _movies()["id"].astype(int).tolist()


def known_movies():
    """Movie ids with rating history in train (the only ids SVD can score)."""
    return sorted(_svd().trained_items)


def id_to_title(movie_id):
    m = _movies()
    row = m[m["id"] == int(movie_id)]
    return row.iloc[0]["title"] if len(row) else str(movie_id)


def genre_labels():
    """genre id -> Thai name mapping."""
    return {int(g["id"]): g["name"] for g in _genres()}


def _cf_raw_scores(user_rated):
    """CF APPROXIMATION for new users: item-item cosine on SVD B's item factors.

    This is NOT Surprise SVD's predict() (μ + b_u + b_i + q_i^T p_u), because that
    requires a fitted user latent factor p_u, which only exists for the 180 training
    users. A webapp user is new, so we fall back to item-item CF over the learned
    item factors q_i:
        score(j) = sum_i rating_i * max(0, cos(q_i, q_j)) / sum_i max(0, cos(q_i, q_j))
    Already-rated movies are excluded.
    """
    svd = _svd()
    Q = normalize(np.asarray(svd.algo.qi, dtype=float), norm="l2")
    S = Q @ Q.T  # item-item cosine similarity

    inner_to_raw = {i: int(svd.trainset.to_raw_iid(i)) for i in svd.trainset.all_items()}
    raw_to_inner = {v: k for k, v in inner_to_raw.items()}
    rated = [(raw_to_inner[int(m)], float(r)) for m, r in user_rated.items() if int(m) in raw_to_inner]

    rated_ids = {int(m) for m in user_rated}
    out = {}
    for j in range(len(inner_to_raw)):
        mid = inner_to_raw[j]
        if mid in rated_ids:
            continue
        num = den = 0.0
        for i, r in rated:
            sim = max(0.0, float(S[i, j]))
            num += sim * r
            den += sim
        out[mid] = num / den if den > 0 else float(svd.global_mean)
    return out


def _cbf_raw_scores(preferred_genres):
    """CBF v2 similarity over the full catalog, using the FULL v2 feature matrix.

    Reconstructs the Stage 2.1 selected model X = [w*genre_onehot | overview_tfidf]
    (rows L2-normalized) and queries it with a genre-only vector (overview part = 0).
    With no overview seed, the overview component contributes 0, so the ranking equals
    genre similarity — but the full v2 representation is used faithfully, not just the
    genre one-hot.
    """
    cbf = _cbf()
    weight = float(cbf["weight"])
    X = sp.hstack([weight * cbf["genre_onehot"], cbf["overview_tfidf"]]).tocsr()
    Xn = normalize(X, norm="l2")
    ids = [int(i) for i in cbf["ids"]]

    gsorted = sorted(int(g["id"]) for g in _genres())
    n_genres = len(gsorted)
    col = {g: i for i, g in enumerate(gsorted)}
    qg = np.zeros(n_genres, dtype=float)
    for g in preferred_genres:
        if int(g) in col:
            qg[col[int(g)]] = 1.0
    nq = float(np.linalg.norm(qg))
    if nq > 0:
        qg = qg / nq

    # Full query: genre part populated, overview part zero.
    q = np.zeros(Xn.shape[1], dtype=float)
    q[:n_genres] = qg

    scores = np.asarray((Xn @ q)).ravel()
    return {ids[i]: float(scores[i]) for i in range(len(ids))}


def _rank(scored, top_k):
    """Sort by (score desc, TMDB popularity desc) and return top_k [(id, score)].

    The secondary key makes the order deterministic and sensible even when many
    movies share the same score (e.g. exact genre matches in CBF).
    """
    pop = _popularity_map()
    items = sorted(scored.items(), key=lambda kv: (-kv[1], -pop.get(kv[0], 0.0)))
    return items[:top_k]


def recommend_cf(user_rated, top_k=10):
    """Ranked list [(movie_id, score)] — CF approximation for new users.

    Uses item-item cosine on SVD B's item latent factors (q_i); NOT Surprise's
    predict() (which needs a fitted p_u and only exists for the 180 training users).
    """
    return _rank(_cf_raw_scores(user_rated), top_k)


def recommend_cbf(preferred_genres, top_k=10):
    """Ranked list [(movie_id, score)] using the full CBF v2 feature matrix."""
    return _rank(_cbf_raw_scores(preferred_genres), top_k)


def recommend_popular(top_k=10):
    """Ranked list [(movie_id, score)] from train mean rating (popularity)."""
    pop = _pop().sort_values(["train_mean_rating", "train_count"], ascending=False)
    return [(int(r.movie_id), float(r.train_mean_rating)) for r in pop.head(top_k).itertuples()]


def recommend_hybrid(user_rated, preferred_genres, alpha=DEFAULT_ALPHA, top_k=10):
    """Min-max normalize CF and CBF per user, blend alpha*CF + (1-alpha)*CBF.

    Fallbacks: no ratings -> CBF only; no genres -> CF only; neither -> popularity.
    """
    has_ratings = bool(user_rated)
    has_genres = bool(preferred_genres)
    if not has_ratings and not has_genres:
        return recommend_popular(top_k)
    if not has_ratings:
        return recommend_cbf(preferred_genres, top_k)
    if not has_genres:
        return recommend_cf(user_rated, top_k)

    cf_raw = _cf_raw_scores(user_rated)
    cbf_raw = _cbf_raw_scores(preferred_genres)
    catalog = set(cbf_raw) | set(cf_raw)
    cf_full = {m: cf_raw.get(m, 0.0) for m in catalog}
    cbf_full = {m: cbf_raw.get(m, 0.0) for m in catalog}
    cf_n = minmax_normalize(cf_full)
    cbf_n = minmax_normalize(cbf_full)

    beta = 1.0 - alpha
    rated_ids = {int(m) for m in user_rated}
    scored = {
        m: alpha * cf_n[m] + beta * cbf_n[m]
        for m in catalog
        if int(m) not in rated_ids
    }
    return _rank(scored, top_k)
