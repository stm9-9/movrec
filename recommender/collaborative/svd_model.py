"""Collaborative Filtering via SVD (scikit-surprise).

Trained on ratings_train.csv only. Provides rating prediction and top-N
recommendation over the items the model has seen (the 15 survey movies).
Unknown users are handled by the caller via a popularity fallback.
"""
import joblib
import pandas as pd
from surprise import Dataset, Reader, SVD


class SVDRecommender:
    def __init__(self, n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02,
                 random_state=42):
        self.params = dict(
            n_factors=n_factors, n_epochs=n_epochs,
            lr_all=lr_all, reg_all=reg_all, random_state=random_state,
        )
        self.algo = SVD(**self.params)
        self.trainset = None
        self.trained_users = set()
        self.trained_items = set()
        self.global_mean = None

    def fit(self, train_df):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_df[["user_id", "movie_id", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()
        self.algo.fit(self.trainset)
        self.trained_users = {int(self.trainset.to_raw_uid(i)) for i in self.trainset.all_users()}
        self.trained_items = {int(self.trainset.to_raw_iid(i)) for i in self.trainset.all_items()}
        self.global_mean = float(self.trainset.global_mean)
        return self

    def predict(self, user_id, movie_id):
        return float(self.algo.predict(int(user_id), int(movie_id)).est)

    def recommend(self, user_id, candidate_movie_ids, top_n=10, exclude_rated=None):
        """Top-n by predicted rating over candidates, excluding already-rated.

        Returns None when the user is unknown (caller should use a fallback).
        """
        if int(user_id) not in self.trained_users:
            return None
        exclude = set(int(m) for m in (exclude_rated or []))
        cands = [int(m) for m in candidate_movie_ids if int(m) not in exclude]
        scored = [(m, self.predict(int(user_id), m)) for m in cands]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_n]

    def save(self, path):
        joblib.dump(
            {"params": self.params, "algo": self.algo, "trainset": self.trainset},
            path,
        )
        return path

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        inst = cls(**obj["params"])
        inst.algo = obj["algo"]
        inst.trainset = obj["trainset"]
        inst.trained_users = {int(inst.trainset.to_raw_uid(i)) for i in inst.trainset.all_users()}
        inst.trained_items = {int(inst.trainset.to_raw_iid(i)) for i in inst.trainset.all_items()}
        inst.global_mean = float(inst.trainset.global_mean)
        return inst
