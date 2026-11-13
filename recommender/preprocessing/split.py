"""Leakage-free per-user train/test split and popularity baseline.

The split is computed from raw ratings BEFORE any model fitting so that every
recommendation component only ever sees `train`. `test` is reserved for
evaluation.
"""
import numpy as np
import pandas as pd

from . import config


def split_ratings(ratings):
    """Per-user 80/20 holdout.

    Users with >= 2 ratings keep at least 1 rating in the test set; users with
    exactly 1 rating go entirely to train (and cannot be top-K evaluated).
    """
    train_parts, test_parts, info = [], [], []
    for uid, grp in ratings.groupby("user_id"):
        # deterministic shuffle per user
        grp = grp.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        n = len(grp)
        n_test = max(1, int(round(n * config.TEST_SIZE))) if n >= 2 else 0
        test_parts.append(grp.iloc[:n_test])
        train_parts.append(grp.iloc[n_test:])
        info.append({
            "user_id": int(uid),
            "n_ratings": n,
            "n_train": int(n - n_test),
            "n_test": int(n_test),
        })

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(
        columns=ratings.columns
    )
    split_info = pd.DataFrame(info)
    return train, test, split_info


def movie_popularity(train):
    """Mean rating and count per movie from TRAINING data only (cold-start fallback)."""
    pop = (
        train.groupby("movie_id")["rating"]
        .agg(train_mean_rating="mean", train_count="count")
        .reset_index()
    )
    return pop


def split_summary(train, test, split_info, ratings):
    """Aggregate split facts for reporting."""
    total = len(ratings)
    n_train = len(train)
    n_test = len(test)
    n_users = int(ratings["user_id"].nunique())
    n_users_with_test = int((split_info["n_test"] > 0).sum())
    n_users_excluded = int((split_info["n_test"] == 0).sum())
    return {
        "total_ratings": total,
        "n_train": n_train,
        "n_test": n_test,
        "train_frac": round(n_train / total, 4) if total else None,
        "n_users": n_users,
        "n_users_with_test": n_users_with_test,
        "n_users_excluded": n_users_excluded,
    }
