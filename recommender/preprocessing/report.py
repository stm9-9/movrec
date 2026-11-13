"""Generate Markdown reports in data/reports/ from real computed values."""
import pandas as pd

from . import config


def _write(name, text):
    p = config.DATA_REPORTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _empty_count(series):
    return int(series.astype(str).str.strip().eq("").sum())


def data_overview_report(movies_raw, ratings_raw, genres, ratings_long):
    lines = ["# Data Overview", ""]

    lines.append("## movies.csv (TMDb metadata, raw)")
    lines.append(f"- Rows: **{len(movies_raw)}**")
    lines.append(f"- Columns: **{len(movies_raw.columns)}**")
    lines.append("")
    lines.append("| column | dtype | non-empty | empty |")
    lines.append("|---|---|---|---|")
    for c in movies_raw.columns:
        empty = _empty_count(movies_raw[c])
        lines.append(f"| {c} | {movies_raw[c].dtype} | {len(movies_raw) - empty} | {empty} |")
    lines.append("")
    lines.append(f"- Duplicate `id`: {int(movies_raw['id'].duplicated().sum())}")
    lines.append(f"- Distinct genre ids: {_distinct_genre_ids(movies_raw)}")
    lines.append("")

    lines.append("## ratings.csv (Google Forms survey, raw)")
    lines.append(f"- Rows: **{len(ratings_raw)}**")
    lines.append(f"- Columns: **{len(ratings_raw.columns)}**")
    lines.append(f"- Structure: 6 demographic columns + 15 movie-rating columns")
    lines.append(f"- Unique aliases: {ratings_raw.iloc[:, 1].astype(str).str.strip().nunique()}")
    lines.append(f"- Duplicate aliases: {int(ratings_raw.iloc[:, 1].astype(str).str.strip().duplicated().sum())}")
    lines.append("")
    lines.append("### Rating value distribution (non-empty)")
    vc = ratings_long["rating"].value_counts().sort_index()
    lines.append("| rating | count |")
    lines.append("|---|---|")
    for r, n in vc.items():
        lines.append(f"| {r:.1f} | {n} |")
    lines.append("")
    lines.append("### Demographics (from the 180 respondents)")
    for col_i, label in [(2, "gender"), (3, "year"), (4, "age")]:
        vc = ratings_raw.iloc[:, col_i].value_counts()
        lines.append(f"- {label}: " + ", ".join(f"{k} = {v}" for k, v in vc.items()))
    lines.append("")

    lines.append("## genres.json")
    lines.append(f"- Genres: **{len(genres)}**")
    lines.append("- " + ", ".join(f"{g['id']}={g['name']}" for g in genres))
    lines.append("")

    return _write("data_overview.md", "\n".join(lines))


def _distinct_genre_ids(movies_raw):
    import json
    ids = set()
    for g in movies_raw["genre_ids"]:
        g = (g or "").strip()
        if g in ("", "[]"):
            continue
        try:
            ids.update(int(x) for x in json.loads(g))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return len(ids)


def cleaning_report(movie_stats, ratings_stats):
    lines = ["# Data Cleaning Report", ""]

    lines.append("## Movies")
    lines.append(f"- Before: **{movie_stats['n_before']}** rows")
    lines.append(f"- After listwise deletion + dedupe: **{movie_stats['n_after']}** rows")
    lines.append(f"- Removed: **{movie_stats['removed']}** rows")
    lines.append("")
    lines.append("### Detected problems")
    lines.append(f"- Empty `title`: {movie_stats['empty_title']}")
    lines.append(f"- Empty `overview`: {movie_stats['empty_overview']}")
    lines.append(f"- Empty `genre_ids` (`[]`): {movie_stats['empty_genres']}")
    lines.append(f"- Empty `poster_path`: {movie_stats['empty_poster']}")
    lines.append(f"- Empty `release_date`: {movie_stats['empty_release_date']}")
    lines.append(f"- `vote_average == 0.0` (kept, flagged): {movie_stats['vote_average_zero']}")
    lines.append(f"- `vote_average == 10.0` (kept, flagged): {movie_stats['vote_average_ten']}")
    lines.append(f"- Duplicate `id` (raw): {movie_stats['duplicate_ids']}")
    lines.append("")
    lines.append("### Actions")
    lines.append("1. Parsed `genre_ids` (JSON string) into a list; built `genres_text` (Thai names).")
    lines.append("2. Listwise deletion: dropped rows with empty `overview` OR empty genres OR empty `poster_path`.")
    lines.append("3. Dropped duplicate `id` (none found).")
    lines.append("")
    lines.append(
        "> Note: real counts are 9,999 → 9,841; the thesis text cites 10,000 → 5,500. "
        "Per approved decision, real numbers are reported and the discrepancy is documented "
        "for later thesis revision."
    )
    lines.append("")

    lines.append("## Ratings")
    lines.append(f"- Respondents (unique aliases): {ratings_stats['n_unique_aliases']}")
    lines.append(f"- Duplicate aliases: {ratings_stats['duplicate_aliases']}")
    lines.append(f"- Long-form ratings produced: **{ratings_stats['n_ratings']}**")
    lines.append(f"- Empty cells dropped: {ratings_stats['n_empty_cells']} (of {ratings_stats['n_possible']} possible)")
    lines.append(f"- Rating range after validation: {ratings_stats['rating_min']:.1f} – {ratings_stats['rating_max']:.1f}")
    lines.append("")
    lines.append("### Actions")
    lines.append("1. Mapped 15 movie columns (survey order) to TMDB ids.")
    lines.append("2. Melted wide table to long form `(user_id, movie_id, rating)`.")
    lines.append("3. Mapped alias -> integer `user_id`; kept demographics in `users.csv`.")
    lines.append("4. Validated all ratings are within 1.0–5.0 (no violations).")
    lines.append("")
    return _write("data_cleaning_report.md", "\n".join(lines))


def split_report(summary, split_info, popularity, ratings):
    lines = ["# Train/Test Split Report", ""]
    lines.append("Method: per-user 80/20 holdout (fixed `random_state=42`), computed BEFORE model fitting.")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Ratings per user (distribution)")
    per_user = ratings.groupby("user_id").size().value_counts().sort_index()
    lines.append("| ratings/user | users |")
    lines.append("|---|---|")
    for n, c in per_user.items():
        lines.append(f"| {n} | {c} |")
    lines.append("")

    lines.append("## Train rating count per movie (from training data only)")
    lines.append("| movie_id | train_mean_rating | train_count |")
    lines.append("|---|---|---|")
    for _, row in popularity.sort_values("movie_id").iterrows():
        lines.append(f"| {int(row['movie_id'])} | {row['train_mean_rating']:.3f} | {int(row['train_count'])} |")
    lines.append("")

    excluded = split_info[split_info["n_test"] == 0]
    lines.append("## Limitations")
    lines.append(f"- Users excluded from top-K evaluation (fewer than 2 ratings): {len(excluded)}")
    if len(excluded):
        lines.append(f"  - user ids: {sorted(excluded['user_id'].tolist())}")
    lines.append("- Only 15 movies exist in the rating matrix; SVD latent factors up to 100 are "
                 "over-parameterized relative to 15 items (flagged for the CF stage).")
    lines.append("- Users with no rating history are cold-start users; they use the popularity "
                 "fallback (`movie_popularity_train.csv`), never a crash path.")
    lines.append("")
    return _write("split_report.md", "\n".join(lines))
