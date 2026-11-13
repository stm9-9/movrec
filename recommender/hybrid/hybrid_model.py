"""Hybrid recommender combining CF (SVD) and CBF v2.

Blend: HybridScore = alpha * CF_norm + beta * CBF_norm, alpha + beta = 1.
Both CF (predicted rating) and CBF (content similarity) are min-max normalized
to [0,1] per user before blending.
"""
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from recommender.collaborative.svd_model import SVDRecommender


def minmax_normalize(scores):
    """Min-max normalize a {key: value} mapping to [0,1]. Returns a new dict."""
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


class CBFProfileScorer:
    """Content-based user-profile scorer built from the CBF v2 artifact.

    CBF_score(u, j) = sum_i rating_i * cos(X_i, X_j) / sum_i rating_i,
    over the user's train-rated movies i (rating-weighted mean similarity).
    """

    def __init__(self, artifact_path, survey_movie_ids):
        obj = joblib.load(artifact_path)
        self.weight = float(obj["weight"])
        X = sp.hstack([self.weight * obj["genre_onehot"], obj["overview_tfidf"]]).tocsr()
        self.Xn = normalize(X, norm="l2")
        self.ids = [int(i) for i in obj["ids"]]
        self.id_to_index = {i: k for k, i in enumerate(self.ids)}

        # Precompute pairwise cosine among the (15) survey movies only.
        self.survey_ids = [int(m) for m in survey_movie_ids]
        idx = [self.id_to_index[m] for m in self.survey_ids]
        self.survey_cos = (self.Xn[idx] @ self.Xn[idx].T).toarray()
        self.survey_pos = {m: p for p, m in enumerate(self.survey_ids)}

    def score(self, user_train_ratings, candidate_id):
        """Rating-weighted mean cosine similarity to the user's train-rated movies."""
        if int(candidate_id) not in self.survey_pos:
            return 0.0
        cpos = self.survey_pos[int(candidate_id)]
        num = den = 0.0
        for mid, r in user_train_ratings:
            p = self.survey_pos.get(int(mid))
            if p is None:
                continue
            num += float(r) * float(self.survey_cos[p, cpos])
            den += float(r)
        return num / den if den > 0 else 0.0


class HybridRecommender:
    def __init__(self, svd_path, cbf_artifact_path, survey_movie_ids, alpha=0.5):
        self.svd = SVDRecommender.load(svd_path)
        self.cbf = CBFProfileScorer(cbf_artifact_path, survey_movie_ids)
        self.alpha = float(alpha)

    def score_candidates(self, user_id, candidates, user_train_ratings):
        cf_raw = {m: self.svd.predict(int(user_id), int(m)) for m in candidates}
        cbf_raw = {m: self.cbf.score(user_train_ratings, m) for m in candidates}
        cf_n = minmax_normalize(cf_raw)
        cbf_n = minmax_normalize(cbf_raw)
        beta = 1.0 - self.alpha
        return {m: self.alpha * cf_n[m] + beta * cbf_n[m] for m in candidates}

    def recommend(self, user_id, candidates, user_train_ratings, top_n=10, exclude_rated=None):
        exclude = set(int(m) for m in (exclude_rated or []))
        cands = [int(m) for m in candidates if int(m) not in exclude]
        scored = self.score_candidates(user_id, cands, user_train_ratings)
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        return [(m, s) for m, s in ranked[:top_n]]
