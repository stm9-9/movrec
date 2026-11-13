# TASK — Stage 5: Evaluation & Web Integration (Streamlit)

## Context
- Stages 1–4 complete. Stage 4 hybrid selected: Hybrid A (α=0.3 CF / β=0.7 CBF), 
  with honest statistical caution documented in hybrid_report.md.
- Models available:
  - CF: data/processed/svd_model.joblib (SVD Config B)
  - CBF: data/processed/cbf_model_v2.joblib (v2, w=1)
  - Popularity fallback: data/processed/movie_popularity_train.csv
- Movie catalog: data/processed/movies_clean.csv (9,841 movies)
- Ratings: data/processed/ratings_train.csv, ratings_test.csv

## Decisions (locked)
- Architecture: Streamlit (single app, no separate backend/frontend)
- Cold start: (a) user selects preferred genres on entry → CBF recommendation 
  immediately; (b) fallback to popularity ranking if no signal
- User ratings: session-based (st.session_state) with optional export to 
  CSV/JSON for evidence; no real database
- Deploy: Streamlit Cloud (primary) + recorded video demo as backup
- UAT: functional test + short satisfaction survey with friends/advisor

## Constraints (global)
- Do NOT modify Stage 1–4 models, data, or reports.
- Do NOT fabricate numbers, screenshots, or survey results.
- Keep existing project structure; add new files under `webapp/` and `docs/chapter4/05_webapp/`.
- All user-facing text in Thai (with English fallback where needed).
- Follow PROJECT_RULES.md.

---

## Stage 5.1 — Model Serving Layer (no UI yet)

### Objective
Wrap the three scoring functions into a single reusable inference module 
so the Streamlit app never loads raw model logic itself.

### Tasks
1. Create `webapp/inference.py` with functions:
   - `recommend_cf(user_rated: dict, top_k: int)` — returns ranked list using SVD B
   - `recommend_cbf(preferred_genres: list, top_k: int)` — returns ranked list using CBF v2
   - `recommend_popular(top_k: int)` — returns popularity ranking from train data
   - `recommend_hybrid(user_rated: dict, preferred_genres: list, alpha: float, top_k: int)` 
     — min-max normalize per user, blend α·CF + (1−α)·CBF; if user_rated is empty 
     → use CBF only; if preferred_genres is empty → use CF only; if both empty 
     → popularity fallback
2. Add a `known_movies()` helper returning movies that have been rated in train 
   (for CF to work) and a `all_movies()` helper for CBF/popularity.
3. Write `webapp/test_inference.py` (or a `__main__` block) that runs a small 
   sanity check: e.g., for a synthetic user who rated 5 movies, print top-5 
   from each method. Save output to `docs/chapter4/05_webapp/inference_smoke_test.txt`.

### Deliverables
- `webapp/__init__.py`
- `webapp/inference.py`
- `docs/chapter4/05_webapp/inference_smoke_test.txt`

### Gate
STOP after 5.1. Report the smoke-test output. Wait for approval before 5.2.

---

## Stage 5.2 — Streamlit UI (recommendation flow only)

### Objective
Build the interactive UI using `webapp/inference.py` — no deployment yet.

### Tasks
1. Create `webapp/app.py` with sections:
   - **หน้าแรก:** welcome + explainer + "เริ่มต้น" button
   - **ขั้นที่ 1 — เลือกประเภทที่ชอบ:** multiselect of genres 
     (from `genre_ids` in movies_clean.csv, mapped to readable Thai labels)
   - **ขั้นที่ 2 — ให้คะแนนภาพยนตร์:** show 5–10 popular movies with 1–5 star 
     rating widgets (stored in `st.session_state`)
   - **ขั้นที่ 3 — ผลแนะนำ:** display top-10 hybrid recommendations with 
     poster, title, genres, and a short reason ("เพราะคุณชอบ [genre]")
   - **แถบด้านข้าง:** toggle method (Hybrid A / CF only / CBF only / Popularity) 
     for demo/comparison; show α/β weights used
2. Add a "บันทึกผลเป็น CSV" button exporting the recommendation list (for 
   evidence collection).
3. Handle edge cases:
   - no genre selected, no rating given → popularity fallback with a message
   - empty catalog → friendly error (should not happen)
4. All user-visible strings in Thai.

### Deliverables
- `webapp/app.py`
- `docs/chapter4/05_webapp/ui_walkthrough.md` (step-by-step screenshots + captions)
- `outputs/screenshots/chapter4/05_webapp/` (at least: home, genre select, 
  rating step, result page, sidebar toggle, CSV export)

### Gate
STOP after 5.2. Report the UI walkthrough and list of screenshots captured. 
Wait for approval before 5.3.

---

## Stage 5.3 — Testing & UAT Preparation

### Objective
Verify correctness end-to-end and prepare UAT materials.

### Tasks
1. **Functional tests:** run `webapp/app.py` locally, test:
   - cold start (no ratings, no genres) → popularity
   - genre-only → CBF
   - rating-only → CF
   - both → Hybrid A
   - export CSV works
2. **Consistency check:** ensure recommendation results from `webapp/inference.py` 
   match the offline results from Stage 4 for a sample user (same candidate set, 
   same scores). Save diff report to `docs/chapter4/05_webapp/consistency_check.md`.
3. **UAT form:** create `docs/chapter4/05_webapp/uat_survey.md` containing:
   - Short intro (what the app does)
   - 5–7 Likert-scale questions (1–5) on: ease of use, relevance of 
     recommendations, clarity of explanations, overall satisfaction
   - 1 open-ended feedback question
   - Note: distribute as Google Form or printed; collect ≥5 respondents
4. **UAT results template:** create `docs/chapter4/05_webapp/uat_results.csv` 
   with empty rows ready to fill (do NOT fabricate results).

### Deliverables
- `docs/chapter4/05_webapp/consistency_check.md`
- `docs/chapter4/05_webapp/uat_survey.md`
- `docs/chapter4/05_webapp/uat_results.csv` (empty template)
- `docs/chapter4/05_webapp/functional_test_log.md`

### Gate
STOP after 5.3. Report functional test results and consistency check outcome. 
Wait for approval before 5.4.

---

## Stage 5.4 — Deployment & Evidence Collection

### Objective
Deploy to Streamlit Cloud and collect final evidence.

### Tasks
1. Create `requirements.txt` at project root with pinned versions 
   (streamlit, pandas, numpy, scikit-learn, scikit-surprise, joblib, etc.).
2. Create `webapp/README.md` with deploy instructions for Streamlit Cloud.
3. Deploy `webapp/app.py` to Streamlit Community Cloud (free tier).
   - Record the public URL.
   - If deploy fails for any reason, document the failure and provide a 
     video walkthrough as backup evidence.
4. Collect final screenshots for Chapter 4:
   - deployed app home page
   - deployed app result page
   - UAT survey form (screenshot)
   - UAT results summary (after respondents fill it in)
5. Update `docs/PROGRESS.md` with Stage 5 complete + URL + note on 
   UAT respondent count.

### Deliverables
- `requirements.txt`
- `webapp/README.md`
- `docs/chapter4/05_webapp/deployment.md` (URL, deploy steps, screenshots)
- `outputs/screenshots/chapter4/05_webapp/deployed_*.png`
- Updated `docs/PROGRESS.md`

### Constraints
- Do NOT fabricate the deployed URL — if not deployed, say so and document why.
- Do NOT fabricate UAT results — leave the CSV empty if no respondents yet.
- Do NOT modify any Stage 1–4 files.

### Gate
Final gate: after 5.4, STOP and report. This is the end of the planned 
stages — no further work without explicit owner instruction.

---

## Global rules
- Work strictly sub-stage by sub-stage. Do not skip ahead.
- Stop and wait for approval after every sub-stage.
- Report real numbers and real URLs only. If something cannot be done, 
  say so explicitly.
- Keep docs/PROGRESS.md updated after each sub-stage.
- User-visible text in Thai; code comments and logs in English.