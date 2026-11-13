# Phase 1 — Analysis Findings (Movie Recommendation Thesis)

Scope: analysis only. No code/data changes made (except extracting the thesis PDF text).

## 1. Thesis requirements relevant to Chapter 4

Source: `thesis/chapter 1-3.pdf` (54 pages, Thai). Full extracted text: `outputs/thesis_ch1-3_extracted.txt`.

- **Goal (1.2):** build a movie recommender using data mining; **compare 3 models** — Content-Based Filtering (CBF), Collaborative Filtering (CF), Hybrid.
- **Scope (1.3):** data from TMDb API; similarity uses Genres + director + actors + overview + rating.
  - CBF = Cosine Similarity; CF = SVD (Matrix Factorization); Hybrid = Weighted Hybrid.
- **Tools (1.4):** Python, SQLite, React.
- **Data (3.1):**
  - Primary: survey ratings from 180 CS students (RMUTK), 15 movies, scale 1.0–5.0 → utility matrix 180×15 (max 2,700 ratings).
  - Secondary: TMDb metadata, target **10,000 movies** (500 pages × 20), fields `id, title, overview, genre_ids, release_date, vote_average, popularity, poster_path`.
  - Cleaning: listwise deletion of missing overview/genre/poster; noise filtering (ratings >5, duplicate aliases); dedup by movie id.
  - Integration: join by movie id → stated **5,500 movies**.
- **Models (3.3):**
  - CBF: metadata soup (genres + director + overview) → TF-IDF (`max_features=5000`) → cosine similarity matrix.
  - CF: 180×15 matrix → StandardScaler → K-Means (K=2, n_init=10, random_state=42) → SVD (surprise) with `rating_scale=(1,5)`.
  - Hybrid: Min-Max scale to [0,1]; **weight SVD=0.6, Cosine=0.4**; sort desc; top-10 excluding already-rated.
- **Evaluation (3.5):** train/test 80/20 (fixed random_state); **RMSE, MAE**; SVD learning-curve (loss per epoch); classification threshold **≥3.0 = like**, confusion matrix + **Precision/Recall/F1**; user acceptance testing (Render free tier).
- **Backend/UI (3.2/3.4):** Flask + Flask-CORS + SQLite + React (4 pages: Home, Recommend, Search, Movie Details). DB tables: USERS, RATINGS, MOVIES, GENRES, MOVIE_GENRES_MAP, CLUSTERS. API: `GET /api/movies`, `GET /api/movies/<id>`, `GET /api/recommend/<user_id>`, `GET /api/search?q=`, `POST /api/ratings`.

## 2. Available data (measured)

`data/raw/` (immutable) contains 3 files. `program/` holds identical copies of two plus the fetch script.

| File | Shape | Notes |
|---|---|---|
| `movies.csv` | 9,999 × 8 | TMDb metadata |
| `ratings.csv` | 180 × 21 | Google Forms survey export |
| `genres.json` | 19 rows | Thai genre id→name |

`movies.csv` columns: `id(int), title, overview, genre_ids(str JSON list), release_date, vote_average(float), popularity(float), poster_path`.

- Duplicate `id`: **0**.
- Empty overview: 79; empty genre_ids (`[]`): 65; empty poster_path: 61; empty title: 0; empty release_date: 49.
- Rows missing overview OR genre OR poster (union): **158** → after listwise deletion **9,841 movies remain**.
- `vote_average` range 0.0–10.0; 353 rows = 0.0, 19 rows = 10.0.
- 19 distinct genre ids, all present in `genres.json` (Thai names: บู๊/ผจญ/แอนนิเมชั่น/ตลก/…).
- All **15 survey movie ids are present** in `movies.csv`.

`ratings.csv` (Google Forms):
- Col 0 timestamp; col 1 alias; col 2 gender; col 3 year; col 4 age; col 5 favourite-genres (free text); cols 6–20 = 15 movie ratings.
- 180 unique aliases, **0 duplicates**.
- Non-empty rating cells: **2,188 / 2,700 (81.0%)**; 512 missing.
- Rating range 1.0–5.0 (no out-of-range values). Distribution: 5.0=238, 4.0=849, 3.0=858, 2.0=229, 1.0=14.
- Demographics: ชาย 91 / หญิง 89; year 1–4 = 45 each; age 21–22=91, 18–20=48, 23–24=41.

`program/testdata.py` is the TMDb fetch script (`fetch_tmdb_movies.py`); it hardcodes a TMDB API key (line 21) and defines `SURVEY_MOVIES` (the 15 ids → titles).

## 3. Current application architecture

- **No application code exists yet.** `recommender/*`, `scripts/`, `tests/`, `frontend/`, `backend/app/{api,database,services}` are empty. Only `program/testdata.py` (fetch script) exists.
- `backend/requirements.txt.txt` and `backend/app/config..txt` / `main..txt` are 0-byte placeholders.
- `README.md.txt` is empty.
- The thesis prescribes Flask + SQLite + React (not yet implemented).

## 4. Existing recommendation methods

None implemented. Only data-fetching logic (`program/testdata.py`) is present.

## 5. Problems / conflicts

1. **Dataset size mismatch.** Thesis (2.9, 3.1.3, Table 3-1) states 10,000 raw → **5,500** after cleaning. Actual: **9,999** raw → **~9,841** after listwise deletion. The "5,500" number is not reproducible from the fetched 10k. Chapter 4 tables should use the real 9,999 → 9,841.
2. **CBF features missing.** Scope 1.3.2 and design 3.3.1.1 require **director and actors**, but `movies.csv` has no cast/crew fields. CBF can only use `genre_ids` + `overview` unless director/cast is fetched separately.
3. **SVD hyperparameter conflict.** 3.3.2.4 says `n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02`; 3.3.2.6 later says `n_factors=100, n_epochs=20`. Need to choose one (recommend the 3.3.2.4 values).
4. **"Find streaming platform" benefit (1.6.2)** has no watch-provider data and no API endpoint. Likely out of data scope.
5. **Genre names are abbreviated Thai** (e.g. Action→"บู๊", Adventure→"ผจญ", Animation→"แอนนิเมชั่น") because `language=th-TH` was used. Fine functionally, but a fuller Thai label mapping would improve UI quality.
6. **Hardcoded TMDB API key** in `program/testdata.py` (script itself warns against committing it).

## 6. Recommended implementation plan (Phases 2–4)

1. **Data pipeline** (reproducible, logged): transform `ratings.csv` (Google Forms) → long form `(user_id, movie_id, rating)`; map aliases→users and 15 title columns→survey ids; clean `movies.csv` (listwise deletion; decide on `vote_average==0` rows; dedup). Output to `data/processed/` + summary to `data/reports/`.
2. **CBF**: genre_ids + overview → TF-IDF (`max_features=5000`) → cosine similarity; profile-based top-N per user. (Director/actors unavailable — flag or extend fetch.)
3. **CF**: 180×15 utility matrix → K-Means (K=2) cluster label + SVD (`rating_scale=(1,5)`, pick one hyperparameter set). Predict held-out ratings.
4. **Hybrid**: min-max → 0.6·SVD + 0.4·Cosine → top-10 excluding rated.
5. **Evaluation**: 80/20 split; RMSE + MAE; learning curve; classification (threshold 3.0) → confusion matrix + Precision/Recall/F1; compare CBF vs CF vs Hybrid.
6. **Backend + UI**: Flask + SQLite + React (4 pages), endpoints per Table 3-2.
7. **Evidence**: tables/figures → `outputs/`; Chapter 4 write-up + screenshot checklist → `docs/chapter4/`.

### Decisions needed before Phase 2
- (a) Accept real counts (9,999 → 9,841) instead of thesis "5,500"?
- (b) CBF without director/actors (genre+overview only), or add a fetch for cast/crew?
- (c) Which SVD hyperparameters to use?
