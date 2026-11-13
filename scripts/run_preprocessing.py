"""Data pipeline: raw -> processed (clean movies, long ratings, train/test split).

Run from anywhere:
    python scripts/run_preprocessing.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.preprocessing import (  # noqa: E402
    clean_movies,
    config,
    load_data,
    report,
    split,
    transform_ratings,
)


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.DATA_REPORTS.mkdir(parents=True, exist_ok=True)

    # 1. Load raw (immutable)
    movies_raw = load_data.load_movies()
    ratings_raw = load_data.load_ratings()
    genres = load_data.load_genres()

    # 2. Clean movies
    movies_clean, movie_stats = clean_movies.clean_movies(movies_raw)

    # 3. Transform ratings -> long form + users
    ratings_long, users, ratings_stats = transform_ratings.transform_ratings(ratings_raw)

    # 4. Leakage-free per-user split (BEFORE any model fitting)
    train, test, split_info = split.split_ratings(ratings_long)
    popularity = split.movie_popularity(train)
    summary = split.split_summary(train, test, split_info, ratings_long)

    # 5. Write processed data
    movies_clean.to_csv(config.MOVIES_CLEAN, index=False, encoding="utf-8-sig")
    users.to_csv(config.USERS_CLEAN, index=False, encoding="utf-8-sig")
    ratings_long.to_csv(config.RATINGS_LONG, index=False, encoding="utf-8")
    train.to_csv(config.RATINGS_TRAIN, index=False, encoding="utf-8")
    test.to_csv(config.RATINGS_TEST, index=False, encoding="utf-8")
    popularity.to_csv(config.MOVIE_POPULARITY, index=False, encoding="utf-8")

    # 6. Reports
    p1 = report.data_overview_report(movies_raw, ratings_raw, genres, ratings_long)
    p2 = report.cleaning_report(movie_stats, ratings_stats)
    p3 = report.split_report(summary, split_info, popularity, ratings_long)

    # 7. Console summary
    print("=== DATA PIPELINE COMPLETE ===")
    print(f"movies: {movie_stats['n_before']} -> {movie_stats['n_after']} "
          f"(removed {movie_stats['removed']})")
    print(f"users: {ratings_stats['n_unique_aliases']}")
    print(f"ratings long: {ratings_stats['n_ratings']} "
          f"(dropped {ratings_stats['n_empty_cells']} empty cells)")
    print(f"split: train={summary['n_train']} test={summary['n_test']} "
          f"({summary['train_frac']*100:.1f}% train)")
    print(f"users with test data: {summary['n_users_with_test']} / {summary['n_users']} "
          f"(excluded {summary['n_users_excluded']})")
    print(f"reports: {p1.name}, {p2.name}, {p3.name}")


if __name__ == "__main__":
    main()
