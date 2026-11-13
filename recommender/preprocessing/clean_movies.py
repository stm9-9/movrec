"""Clean the movie metadata: parse genres, listwise deletion, dedupe."""
import json

from . import config
from . import load_data


def _parse_genre_ids(value):
    """Convert a JSON-string genre list (or empty) into a Python list of ints."""
    value = (value or "").strip()
    if value in ("", "[]"):
        return []
    try:
        return [int(x) for x in json.loads(value)]
    except (ValueError, TypeError, json.JSONDecodeError):
        return []


def clean_movies(df):
    """Return (clean_df, stats). Source df is not modified."""
    df = df.copy()
    genres = {g["id"]: g["name"] for g in load_data.load_genres()}

    n_before = int(len(df))
    stats = {
        "n_before": n_before,
        "empty_title": int((df["title"].str.strip() == "").sum()),
        "empty_overview": int((df["overview"].str.strip() == "").sum()),
        "empty_genres": int((df["genre_ids"].str.strip() == "[]").sum()),
        "empty_poster": int((df["poster_path"].str.strip() == "").sum()),
        "empty_release_date": int((df["release_date"].str.strip() == "").sum()),
        "vote_average_zero": int((df["vote_average"] == 0.0).sum()),
        "vote_average_ten": int((df["vote_average"] == 10.0).sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()),
    }

    df["genre_ids_list"] = df["genre_ids"].apply(_parse_genre_ids)
    df["genres_text"] = df["genre_ids_list"].apply(
        lambda ids: " ".join(genres.get(i, "") for i in ids)
    )

    # Thesis 3.1.2.1: listwise deletion of rows missing overview/genre/poster.
    keep = (
        (df["overview"].str.strip() != "")
        & (df["genre_ids_list"].apply(len) > 0)
        & (df["poster_path"].str.strip() != "")
    )
    df = df[keep].reset_index(drop=True)

    # Thesis 3.1.2.3: dedupe by movie id (keep first).
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    stats["n_after"] = int(len(df))
    stats["removed"] = n_before - int(len(df))
    stats["duplicates_after"] = int(df["id"].duplicated().sum())

    # Drop the helper list column; keep genre_ids (JSON string) + genres_text.
    out_cols = [
        "id", "title", "overview", "genre_ids", "release_date",
        "vote_average", "popularity", "poster_path", "genres_text",
    ]
    return df[out_cols], stats
