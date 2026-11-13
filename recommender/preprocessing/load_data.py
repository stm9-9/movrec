"""Load immutable raw data files (never modify data/raw/)."""
import json

import pandas as pd

from . import config


def load_movies():
    """TMDb movie metadata. Empty strings are preserved (no NaN coercion)."""
    return pd.read_csv(config.RAW_MOVIES, keep_default_na=False)


def load_ratings():
    """Google Forms survey export (wide format, Thai headers)."""
    return pd.read_csv(config.RAW_RATINGS, keep_default_na=False)


def load_genres():
    """TMDB genre id -> Thai name mapping."""
    with open(config.RAW_GENRES, encoding="utf-8") as f:
        return json.load(f)
