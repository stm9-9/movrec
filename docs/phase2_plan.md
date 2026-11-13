# Phase 2 — Implementation Plan

Status: PLAN (no implementation started). Awaiting approval before Phase 3.

## 0. Locked decisions (from TASK.md)
1. Real counts: 9,999 → 9,841 movies. Do not force 10,000 → 5,500. Document discrepancy for thesis revision.
2. CBF uses only existing features: `genre_ids` + `overview` (no cast/director).
3. CF/SVD: compare `(n_factors=50, n_epochs=30)` vs `(n_factors=100, n_epochs=20)`; report RMSE & MAE; recommend the better config.
4. No streaming-platform search.

## 1. Environment (verified)
Python 3.12 (bundled runtime). Installed & import-tested: `pandas`, `numpy`, `scipy`, `scikit-learn 1.9.1`, `scikit-surprise 1.1.5`, `matplotlib 3.11.2`, `flask 3.1.3`, `flask-cors 6.0.5`, `requests 2.34.2`.
Frontend: Node + pnpm (bundled) for a minimal React (Vite) app.

## 2. File layout (new work only; adapt existing dirs)
```
scripts/            run_preprocessing.py, run_evaluation.py, run_demo.py
recommender/
  preprocessing/    load_data.py, clean_movies.py, transform_ratings.py, report.py
  content_based/    tfidf_model.py
  collaborative/    svd_model.py, kmeans.py
  hybrid/           fusion.py
  evaluation/       metrics.py, topk_eval.py, split.py
backend/app/        main.py, config.py, database.py, api/, services/
frontend/           React (Vite): Home, Recommend, Search, MovieDetails
tests/              test_preprocessing.py, test_metrics.py, test_recommenders.py
data/processed/     cleaned CSVs
data/reports/       cleaning + summary reports
outputs/tables/     comparison tables
outputs/figures/    PNG figures
docs/chapter4/      evidence index + screenshot checklist
```

## 3. Data pipeline (reproducible, logged)
Script: `scripts/run_preprocessing.py` (reads only `data/raw/`, writes `data/processed/` + `data/reports/`).

3.1 **Movies** (`clean_movies.py`)
- Load `data/raw/movies.csv` (9,999 × 8).
- Parse `genre_ids` (JSON string) → int list; build `genres_text` (Thai names from `genres.json`).
- Cleaning (matches thesis 3.1.2): listwise delete rows with empty `overview`, `genre_ids == []`, or empty `poster_path` → **9,841 rows**; dedupe by `id` (already 0); keep `vote_average==0` rows but flag them (353) in the report for optional later filtering.
- Output `movies_clean.csv`; log before/after counts + per-column missing counts.

3.2 **Ratings** (`transform_ratings.py`)
- Load `data/raw/ratings.csv` (Google Forms, 180 × 21).
- Map 15 movie columns (cols 6–20) → TMDB ids using the fixed survey order (same as `program/testdata.py` `SURVEY_MOVIES`): 299534, 597, 138843, 109445, 168259, 296098, 157336, 18785, 671, 496243, 372058, 396535, 354912, 18886, 1083862.
- Melt to long form `(user_id, movie_id, rating)`; drop blank ratings (512) → **2,188 ratings**; map alias → integer `user_id`; validate ratings in [1,5] (already clean).
- Output `ratings_long.csv` and `users.csv` (alias, user_id, gender, year, age, favorite_genres).

3.3 **Reports** (`report.py`) → `data/reports/`
- `data_overview.md` (size, columns/dtypes, missing, duplicates) — Chapter 4 §data overview.
- `data_cleaning_report.md` (before/after, problems, actions) — Chapter 4 §data cleaning.

## 4. Content-Based Filtering (genre + overview)
Module `recommender/content_based/tfidf_model.py`.
- Build "metadata soup" per movie = `genres_text` + `overview` (Thai).
- `TfidfVectorizer(max_features=5000)` → sparse matrix (9,841 rows).
- Item similarity: fit `NearestNeighbors(metric='cosine')` on the matrix for "similar movies" (Movie Details page).
- User profile = mean of TF-IDF vectors of the user's **train**-rated movies (weighted by rating − 2.5 so disliked items lower weight). Recommend top-K by cosine(profile, movie) over unrated movies.
- Persist vectorizer + matrix (`data/processed/` or a model file) for backend reuse.

## 5. Collaborative Filtering (SVD) + K-Means
Module `recommender/collaborative/svd_model.py` + `kmeans.py`.
- Build long-form ratings → `surprise.Dataset` with `Reader(rating_scale=(1,5))`.
- **Config comparison:** for each of `(n_factors=50, n_epochs=30)` and `(n_factors=100, n_epochs=20)` (both `lr_all=0.005, reg_all=0.02, random_state=42`), `train_test_split(test_size=0.2, random_state=42)` → fit → RMSE & MAE. Report both; recommend the better one.
- Auxiliary K-Means: standardize the 180×15 rating matrix (`StandardScaler`, fill missing with user mean), K=2, `n_init=10, random_state=42` → store cluster label per user (reported, not fed into SVD).
- Persist best SVD model for backend.

## 6. Hybrid (Weighted)
Module `recommender/hybrid/fusion.py`.
- For a user: SVD predicted ratings over unrated movies → min-max to [0,1]; CBF cosine scores → already [0,1] (also min-max for safety).
- `score = 0.6 * SVD_norm + 0.4 * CBF_norm`; sort desc; exclude train-rated movies; return top-10.

## 7. Evaluation protocol (no leakage)
Module `recommender/evaluation/`.
- **Rating prediction (CF only):** 80/20 surprise split → RMSE & MAE for both SVD configs (decision #3).
- **Top-K quality (all methods, primary comparison):** per-user holdout — for each user with ≥2 ratings, hold out 20% (≥1) as test, rest as train (`random_state=42`). Each method ranks unrated-in-train movies; `relevant = rating >= 3.0`. Compute `Precision@10, Recall@10, F1@10`, average over users.
- Methods compared: **Popularity baseline** (highest mean survey rating, non-personalized), **CBF**, **CF (best SVD config)**, **Hybrid**.
- Output: `outputs/tables/evaluation_comparison.csv` (+ `.md`), `outputs/tables/svd_config_comparison.csv`.

## 8. Figures (matplotlib)
- Rating distribution histogram; genre frequency bar chart; SVD learning-curve (RMSE/MAE per epoch via a minimal numpy SGD-MF mirroring SVD, clearly labeled) — optional; top-K metric comparison bar chart. → `outputs/figures/`.

## 9. Web application (minimal, non-slop)
Backend (Flask + SQLite): tables USERS, RATINGS, MOVIES, GENRES, MOVIE_GENRES_MAP (skip CLUSTERS unless needed). Endpoints per thesis Table 3-2: `GET /api/movies`, `GET /api/movies/<id>`, `GET /api/recommend/<user_id>`, `GET /api/search?q=`, `POST /api/ratings`. Load models once at startup.
Frontend (React/Vite, plain CSS, dark theme): Home (movie grid), Recommend (top-10 per user), Search, Movie Details (with "similar movies" + rating). Real data only; no fake stats; responsive; minimal animation.

## 10. Chapter 4 evidence
- Tables: data overview, cleaning before/after, SVD config comparison, top-K comparison. → `outputs/tables/`.
- Figures: as §8 → `outputs/figures/`.
- Sample recommendations: top-10 for 2–3 example users (`scripts/run_demo.py`).
- `docs/chapter4/evidence_index.md` mapping each Chapter 4 section → supporting file; `docs/chapter4/screenshot_checklist.md`.
- `README.md`: update with run instructions.

## 11. Reproducibility
- All random operations use fixed `random_state=42`.
- Raw data never modified; processed data + reports regenerated by scripts.
- Exact commands recorded in README and `docs/chapter4/`.

## 12. Risks / notes
- `scikit-surprise 1.1.5` imported successfully with numpy 2.x (verified).
- CBF is Thai-language TF-IDF; fine for Thai UI, but tokenization is whitespace/char-level — acceptable for a bachelor thesis.
- The 9,999 → 9,841 counts differ from thesis "5,500"; documented for revision, not forced.
