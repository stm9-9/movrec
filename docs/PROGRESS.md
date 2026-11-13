# Progress

## Stage 1 — Data pipeline (COMPLETE)

Implemented `recommender/preprocessing/` (config, load_data, clean_movies, transform_ratings, split, report) and `scripts/run_preprocessing.py`.

### Real results (run on 2026-11-13)
- Movies: **9,999 → 9,841** (removed 158 rows via listwise deletion of empty overview/genre/poster; 0 duplicate ids).
- Users: **180** (unique aliases, 0 duplicates).
- Ratings long-form: **2,188** (512 empty cells dropped from 2,700 possible; all values within 1.0–5.0).
- Train/test (per-user 80/20, `random_state=42`): **train=1,749, test=439** (79.9% train). All 180 users have ≥8 ratings, so none excluded from top-K evaluation.
- Rating distribution: 5.0=238, 4.0=849, 3.0=858, 2.0=229, 1.0=14.

### Outputs
- `data/processed/`: `movies_clean.csv`, `users.csv`, `ratings_long.csv`, `ratings_train.csv`, `ratings_test.csv`, `movie_popularity_train.csv`.
- `data/reports/`: `data_overview.md`, `data_cleaning_report.md`, `split_report.md`.

### Guardrails satisfied
- Raw data (`data/raw/`) untouched.
- Leakage: split computed before any model fitting; only `ratings_train.csv` feeds models; `ratings_test.csv` reserved for evaluation.
- Cold-start fallback: `movie_popularity_train.csv` (train-only mean rating per movie) for users with no history.
- Small data: every user has ≥8 ratings (verified); 15-item matrix flagged for the SVD stage (n_factors up to 100 vs 15 items).

### K-Means determination
K-Means **is required** by the thesis (Ch.2 §2.4 theory + Ch.3 §3.3.2 CF pipeline: K=2 user clustering). It will be retained in the CF stage with a documented purpose (cluster users into K=2, store label) and reported as an auxiliary step — not injected into SVD training. No additional clustering beyond the thesis design.

## Stage 2 — Content-Based Filtering (COMPLETE)

Implemented `recommender/content_based/tfidf_model.py` (`ContentBasedModel`) and `scripts/run_cbf.py`.

### Real results (run on 2026-11-13)
- Movies used: **9,841** (0 empty-feature rows dropped).
- Feature text: `genres_text` (Thai) + `overview`.
- TF-IDF: `max_features=5000` → matrix **9,841 × 5,000** (sparse).
- Pairwise cosine matrix dimension: **9,841 × 9,841** (retrieved lazily via kNN `metric='cosine', k=11`; source excluded).
- Tested 6 source movies (Avengers: Endgame, Titanic, Parasite, Your Name, Interstellar, The Conjuring); top-10 similar each. Recommendations are genre-coherent (e.g. The Conjuring → horror: Annabelle, The Conjuring 2; Interstellar → sci-fi). Similarity range ~0.15–0.30 (expected for a 9,841-movie corpus).

### Outputs / evidence
- `outputs/tables/cbf_similar_movies.csv` (all example rows).
- `outputs/figures/cbf_top10_examples.png` (Thai-capable font verified).
- `data/processed/cbf_model.joblib` (reusable model artifact).
- `data/reports/cbf_report.md` (implementation + test report).
- `outputs/screenshots/chapter4/02_content_based/` (model_summary.txt, execution_output.txt, cbf_examples.md, figure copy).
- `docs/chapter4/02_content_based_screenshot_checklist.md`.

### Limitations (documented in report)
- Thai tokenized by whitespace (`TfidfVectorizer` default); genre signal dominates. Char n-grams/Thai segmenter not added to avoid extra dependencies.
- Full dense cosine matrix not materialized; kNN used (equivalent top-k).

## Stage 2.1 — CBF quality improvement (COMPLETE)

Implemented `recommender/content_based/cbf_v2.py` and `scripts/run_cbf_v2_experiment.py`.
CBF v1 (`tfidf_model.py`) preserved unchanged as the baseline.

### Methods compared (same 6 source movies)
v1 (combined word TF-IDF), genre-only (one-hot), overview-only (char 3-5 n-gram TF-IDF), and
v2 w∈{1,2,4,8,16} (`[w·genre_onehot | overview_char_tfidf]`).

### Real results
| config | genre_jaccard | content_sim | f1_balance |
|---|---|---|---|
| v1 (combined word TF-IDF) | 0.4105 | 0.2367 | 0.3002 |
| genre-only | 0.9750 | 0.1044 | 0.1887 |
| overview-only | 0.2849 | 0.3575 | 0.3171 |
| **v2 w=1** | **0.9181** | **0.2729** | **0.4208** |
| v2 w=2/4/8/16 | 0.9750 | 0.2112 | 0.3472 |

- **Selected: v2 w=1** — the only config improving on v1 in BOTH metrics (genre Jaccard 0.41→0.92,
  content sim 0.24→0.27; F1 0.30→0.42). Duplicate/source checks all 0.
- Genre-only peaks genre Jaccard but content sim collapses; higher w collapses to genre-only.
- Overview char n-grams (dependency-free) beat whitespace tokenization for Thai overviews.

### Outputs / evidence
- `outputs/tables/cbf_config_comparison.csv`, `cbf_v2_experiment_details.csv`, `cbf_v2_top10.csv`.
- `outputs/figures/cbf_config_comparison.png`.
- `data/processed/cbf_model_v2.joblib` (selected model artifact).
- `data/reports/cbf_v2_report.md`.
- `outputs/screenshots/chapter4/02_content_based/cbf_v2_examples.md` + `cbf_config_comparison.png`.
- `docs/chapter4/02_content_based_screenshot_checklist.md` (updated).

## Stage 3 — Collaborative Filtering (SVD) (COMPLETE)

Implemented `recommender/collaborative/svd_model.py` (`SVDRecommender`), `kmeans.py`, and
`scripts/run_cf.py`. Trained on `ratings_train.csv` only; evaluated on `ratings_test.csv`.

### Real results (run on 2026-11-13)
| config | RMSE | MAE | P@1 | P@3 | P@5 | P@10 | R@10 | F1@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| A: n_factors=50, n_epochs=30 | 0.8479 | 0.6927 | 0.4944 | 0.4667 | 0.4267 | 0.2206 | 0.9833 | 0.3558 | 0.6903 |
| **B: n_factors=100, n_epochs=20** | **0.8346** | **0.6826** | 0.5167 | 0.4611 | 0.4256 | 0.2206 | 0.9833 | 0.3558 | 0.6990 |
| popularity (reference) | — | — | 0.5389 | 0.4593 | 0.4267 | 0.2206 | 0.9833 | 0.3558 | 0.7125 |

- **Selected: B (100,20)** — lowest RMSE (0.8346) and MAE (0.6826), and best P@1/MRR among SVD configs.
- Top-10 ranking is capped by the small candidate set (3–9 items/user, mean 5.3), so P@10/R@10/F1@10 are
  near-identical across models; small-K (P@1/@3, MRR) is the meaningful ranking signal.
- Honest finding: the non-personalized popularity baseline is competitive at K=1 (0.5389 / MRR 0.7125)
  — SVD personalization gives only marginal ranking gains on this 15-item dataset. Documented.
- K-Means K=2 (auxiliary, NOT used by SVD): cluster sizes 98 / 82.

### Outputs / evidence
- `outputs/tables/cf_config_comparison.csv`, `cf_topn_examples.csv`, `user_clusters_k2.csv`.
- `outputs/figures/cf_svd_comparison.png`.
- `data/processed/svd_model.joblib` (selected model).
- `data/reports/cf_report.md`.
- `outputs/screenshots/chapter4/03_collaborative/cf_examples.md`.
- `docs/chapter4/03_collaborative_screenshot_checklist.md`.

## Stage 3.1 — CF evaluation audit (COMPLETE)

Read-only audit via `scripts/audit_cf_evaluation.py` → `data/reports/cf_evaluation_audit.md`.

### Verified (with real numbers)
- **No leakage:** train∩test = **0** overlapping (user, movie) pairs; SVD fit on train only.
- **Popularity is train-only:** recomputed mean-rating diff = 4.44e-16.
- **Candidate set** = 15 survey movies minus train-rated → **3–9 items/user (mean 5.28)**; 180/180 users have ≤10 candidates.
- **Metrics correct:** Precision/Recall/F1@K and MRR formulas match standard definitions.
- **Fair comparison:** SVD and popularity use identical per-user candidate sets (only the scoring fn differs).
- **Threshold ≥3:** 88.9% of ratings are positive → Recall@10 inflated toward 1.

### Issues found + corrections
1. **Correction #1 (APPLIED — reporting/labeling only):** `data/reports/cf_report.md` and
   `docs/chapter4/03_collaborative_screenshot_checklist.md` now label **P@10/R@10/F1@10 as
   "reference-only"** and state **P@1/@3/@5 + MRR** (ranking) and **RMSE/MAE** (prediction) as the
   primary metrics. No numbers changed, no models rerun.
2. **Correction #2 (DEFERRED to Chapter 4 limitations):** leave-one-out ranking over test items is
   not implemented now; documented as a limitation.
3. RMSE/MAE kept as the primary CF prediction metric.

Stage 3 results preserved unchanged as the original baseline.

## Stage 4 — Hybrid recommendation (COMPLETE)

Implemented `recommender/hybrid/hybrid_model.py` (`HybridRecommender`, `CBFProfileScorer`) and
`scripts/run_hybrid.py`. Combined SVD B + CBF v2 over the same 15-movie candidate set as Stage 3;
min-max normalized each source per user, blend α·CF + β·CBF.

### Real results (run on 2026-11-13)
| method | RMSE | MAE | P@1 | P@3 | P@5 | MRR | div@3 | div@5 |
|---|---|---|---|---|---|---|---|---|
| Popularity | — | — | 0.5389 | 0.4593 | 0.4267 | 0.7125 | 0.8241 | 0.8614 |
| CF (SVD B) | 0.8346 | 0.6826 | 0.5167 | 0.4611 | 0.4256 | 0.6990 | 0.8349 | 0.8624 |
| CBF v2 | — | — | 0.5111 | 0.4481 | 0.4189 | 0.7042 | 0.8708 | 0.8651 |
| **Hybrid A (0.3/0.7)** | — | — | **0.5444** | 0.4759 | 0.4178 | **0.7245** | 0.8630 | 0.8638 |
| Hybrid B (0.5/0.5) | — | — | 0.5333 | **0.4852** | 0.4189 | 0.7184 | 0.8548 | 0.8638 |
| Hybrid C (0.7/0.3) | — | — | 0.5278 | 0.4741 | 0.4244 | 0.7102 | 0.8453 | 0.8632 |

- **Selected: Hybrid A (α=0.3 CF / β=0.7 CBF)** — highest MRR (0.7245) and P@1 (0.5444), marginally
  higher than the best single model (Popularity: MRR 0.7125 / P@1 0.5389). The gap is small (~1 user
  on P@1) and was not significance-tested (see the report's statistical caution).
- Hybrid B best on P@3 (0.4852).
- Diversity@3: CBF most diverse (0.8708), popularity least (0.8241); hybrids in between.
- **Data-driven note:** results favor a CBF-heavy blend (0.3/0.7), contradicting the thesis's assumed
  0.6 CF / 0.4 CBF weighting. Documented honestly in the report.
- P@10/R@10/F1@10 (0.2206/0.9833/0.3558) and diversity@10 (0.8685) are reference-only (identical
  across methods — top-10 == whole candidate set).

### Outputs / evidence
- `outputs/tables/hybrid_config_comparison.csv`, `hybrid_topn_examples.csv`.
- `outputs/figures/hybrid_comparison.png`.
- `data/reports/hybrid_report.md`.
- `outputs/screenshots/chapter4/04_hybrid/` (hybrid_examples.md + figure).
- `docs/chapter4/04_hybrid_screenshot_checklist.md`.

## Stage 5.1 — Model serving layer (COMPLETE)

Implemented `webapp/inference.py` and `webapp/test_inference.py`.

- API: `recommend_cf` (item-item CF from SVD B item latent factors — works for NEW users),
  `recommend_cbf` (genre cosine from CBF v2 over the full 9,841-movie catalog), `recommend_popular`
  (train mean rating, 15 survey movies), `recommend_hybrid` (min-max normalize + blend α·CF + (1−α)·CBF,
  with the specified fallbacks). Helpers: `known_movies()` (15), `all_movies()` (9,841), `id_to_title()`,
  `genre_labels()`.
- Smoke test (synthetic user rated 5 movies + genres [บู๊, นิยายวิทยาศาสตร์]): CF → Conjuring/F&F7/Harry
  Potter/Resident Evil/Frozen; CBF → The Matrix/Spider-Man/Hunger Games/Venom/Terminator (genre match);
  Hybrid (α=0.3) blends both; no-signal fallback → popularity. All correct.
- Output: `docs/chapter4/05_webapp/inference_smoke_test.txt`.
- Correction (Stage 5.1 review): `recommend_cf` relabeled as a **CF approximation** (item-item cosine
  on SVD B item factors — NOT Surprise `predict()`); `recommend_cbf` fixed to reconstruct and query the
  FULL CBF v2 feature matrix (genre + overview); added a deterministic TMDB-popularity tie-break for
  equal scores.

## Stage 5.2 — Streamlit UI (COMPLETE)

Implemented `webapp/app.py` (Streamlit UI) + evidence docs. Added dependency: `streamlit 1.67.0`.

- 3-step flow: หน้าแรก → ขั้นที่ 1 เลือกประเภท (multiselect, 19 Thai genres) → ขั้นที่ 2 ให้คะแนน
  (10 popular movies, 1–5 stars) → ขั้นที่ 3 ผลแนะนำ (top-10 with poster/title/genres + reason
  "เพราะคุณชอบ …").
- Sidebar: method toggle (Hybrid A / CF only / CBF only / Popularity) + α/β weights shown.
- "บันทึกผลเป็น CSV" download button; edge cases handled (no signal → popularity + message; CF without
  ratings / CBF without genres → warning + popularity fallback; empty catalog → friendly error).
- Verified: `py_compile` OK; `streamlit run webapp/app.py` starts cleanly (localhost:8511).
- Design note added to `webapp/inference.py` docstring + smoke-test header (CF approximation / CBF
  genre-only query / hybrid approximates Stage 4).
- Evidence: `docs/chapter4/05_webapp/ui_walkthrough.md`, `outputs/screenshots/chapter4/05_webapp/README.md`
  (6-shot checklist). Screenshots to be captured by the user (no fabricated images).

## Stage 5.2b — UI redesign (COMPLETE)

Redesigned `webapp/app.py` ("HBO Max × Linear" dark minimal) + `.streamlit/config.toml` theme.
No logic changed in `webapp/inference.py`; no Stage 1–4 files touched.

- Theme: bg `#0a0e1a`, card `#121826`, border `#1f2937`, accent `#4f8cff`, hover `#6ba3ff`.
- Fonts: Inter (UI) + Playfair Display (hero only), via Google Fonts.
- Single-scroll main area: hero → genre pills (`st.pills`) → rating grid (5-col cards + stars) →
  results grid (5-col cards with poster, score badge, genres, reason).
- Sidebar "แผงควบคุมโมเดล": method radio, α slider (β=1−α) when Hybrid A, live Top-5 preview,
  CSV download at bottom.
- CSV columns: session, method, alpha, beta, movie_id, title, score, rank.
- Edge cases preserved. Verified: `py_compile` OK; `streamlit run webapp/app.py` starts cleanly
  (localhost:8512).
- Evidence: `ui_walkthrough.md` + `README.md` updated to 7 screenshots (no fabricated images).

## Stage 5.3 — Testing & UAT preparation (COMPLETE)

- Functional tests (`webapp/test_functional.py` → `functional_test_log.md`): all PASS.
  Cold start → popularity; genre-only → CBF; rating-only → CF; both → hybrid; CSV columns (8) + 10 rows OK.
- Consistency check (`webapp/test_consistency.py` → `consistency_check.md`): all PASS.
  Popularity matches `movie_popularity_train.csv` exactly (max |diff| = 0); SVD item factors shape
  (15, 100); CBF webapp scores match an independent `Xn@q` recomputation exactly (max |diff| = 0 over
  9,841 movies); CF/Hybrid documented as an approximation (not compared to Stage 4 Surprise predict /
  Hybrid A).
- UAT materials: `uat_survey.md` (6 Likert 1–5 + 1 open-ended, Thai); `uat_results.csv` empty template
  (header only — no fabricated results).

## Stage 5.4 — Deployment & evidence preparation (COMPLETE — awaiting owner deploy)

- `requirements.txt` created with pinned versions verified via `importlib.metadata` (not guessed):
  streamlit 1.67.0, pandas 3.0.1, numpy 2.3.5, scikit-learn 1.9.1, scikit-surprise 1.1.5,
  joblib 1.6.0, scipy 1.18.1.
- `webapp/README.md` created (run locally + Streamlit Cloud deploy steps + required files +
  model-size/loading note).
- `docs/chapter4/05_webapp/deployment.md` created (checklist, live-URL placeholder, known limitations,
  UAT pending note).
- Verified: no hard-coded absolute paths in `webapp/`, `recommender/`, or `scripts/` — all use
  project-root-relative `Path(__file__)` paths; models/data referenced with paths that exist on
  Streamlit Cloud.
- No auto-deploy (owner deploys manually). `uat_results.csv` left empty (header only).

## Next
- Owner: deploy to Streamlit Community Cloud + collect UAT results. No further automated work
  pending owner instruction.
