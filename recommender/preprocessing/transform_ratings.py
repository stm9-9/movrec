"""Transform the Google Forms ratings export into long form (user_id, movie_id, rating)."""
import pandas as pd

from . import config

# Rating columns in ratings.csv are indices 6..20 (0-based); demographics 0..5.
RATING_COL_START = 6
RATING_COL_END = 21


def _verify_column_order(df):
    """Guard against column-order drift by checking each survey title appears in its column."""
    movie_cols = list(df.columns[RATING_COL_START:RATING_COL_END])
    if len(movie_cols) != len(config.SURVEY_MOVIES):
        raise AssertionError(
            f"expected {len(config.SURVEY_MOVIES)} rating columns, found {len(movie_cols)}"
        )
    for col, (movie_id, title) in zip(movie_cols, config.SURVEY_MOVIES):
        if title.split(":")[0].strip().lower() not in col.lower():
            raise AssertionError(
                f"column '{col}' does not match expected survey movie '{title}' ({movie_id})"
            )
    return movie_cols


def transform_ratings(df):
    """Return (ratings_long_df, users_df, stats). Source df is not modified."""
    df = df.copy()
    movie_cols = _verify_column_order(df)
    col_to_movie = {c: mid for c, (mid, _) in zip(movie_cols, config.SURVEY_MOVIES)}

    alias_col = df.columns[1]
    df[alias_col] = df[alias_col].astype(str).str.strip()

    unique_aliases = sorted(df[alias_col].unique())
    alias_to_id = {a: i + 1 for i, a in enumerate(unique_aliases)}

    users_rows = []
    for a in unique_aliases:
        row = df[df[alias_col] == a].iloc[0]
        users_rows.append({
            "user_id": alias_to_id[a],
            "alias": a,
            "gender": row.iloc[2],
            "year": row.iloc[3],
            "age": row.iloc[4],
            "favorite_genres": row.iloc[5],
        })
    users = pd.DataFrame(users_rows)

    records = []
    n_empty = 0
    for _, r in df.iterrows():
        uid = alias_to_id[r[alias_col]]
        for col in movie_cols:
            raw = str(r[col]).strip()
            if raw == "":
                n_empty += 1
                continue
            try:
                val = float(raw)
            except ValueError:
                raise ValueError(f"non-numeric rating '{raw}' for user {uid}, column '{col}'")
            if not (config.RATING_MIN <= val <= config.RATING_MAX):
                raise ValueError(f"rating {val} out of range for user {uid}, column '{col}'")
            records.append({
                "user_id": uid,
                "movie_id": col_to_movie[col],
                "rating": val,
            })

    ratings = pd.DataFrame(records)

    stats = {
        "n_users": int(users["user_id"].nunique()),
        "n_unique_aliases": len(unique_aliases),
        "duplicate_aliases": int(df[alias_col].duplicated().sum()),
        "n_ratings": int(len(ratings)),
        "n_empty_cells": n_empty,
        "n_possible": int(df.shape[0] * len(movie_cols)),
        "rating_min": float(ratings["rating"].min()) if len(ratings) else None,
        "rating_max": float(ratings["rating"].max()) if len(ratings) else None,
    }
    return ratings, users, stats
