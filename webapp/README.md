# Webapp — Movie Recommender (Streamlit)

Single-page Streamlit app for the movie recommendation system (thesis project).
It wraps the trained models behind `webapp/inference.py` and provides a Thai UI:
hero → genre selection → movie rating → recommendations, with a persistent
"Model Control Panel" sidebar (method toggle, α/β weights, live top-5 preview,
CSV export).

## Run locally

From the repository root:

```
pip install -r requirements.txt
streamlit run webapp/app.py
```

Then open the printed URL (default `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub. The repo must include `webapp/`, `recommender/`,
   and the required `data/` files (see below).
2. In [Streamlit Community Cloud](https://share.streamlit.io): **New app** → pick the
   repository and branch.
3. Set **Main file path** = `webapp/app.py`.
4. Streamlit installs dependencies from `requirements.txt` automatically.
5. Open the deployed URL.

## Required files in the repository

- `webapp/app.py`, `webapp/inference.py`, `webapp/__init__.py`
- `recommender/` (at least `collaborative/svd_model.py`, `hybrid/hybrid_model.py`,
  and the package `__init__.py` files)
- `data/raw/genres.json`
- `data/processed/movies_clean.csv`
- `data/processed/svd_model.joblib`
- `data/processed/cbf_model_v2.joblib`
- `data/processed/movie_popularity_train.csv`

All paths are project-root-relative (`Path(__file__).resolve().parents[1]`), so the
app works from any checkout location including Streamlit Cloud.

## Model loading & size

- `data/processed/cbf_model_v2.joblib` ≈ **66 MB** (largest artifact).
- `data/processed/svd_model.joblib` ≈ 0.2 MB; `movies_clean.csv` ≈ 9.5 MB.
- Models are loaded lazily and cached **once per process** via module-level
  `functools.lru_cache` in `webapp/inference.py`. This is equivalent to
  `@st.cache_resource` for these module-level singletons, so the heavy files are
  read from disk only on first use.

## Notes

- Python 3.12; dependencies pinned in `requirements.txt` to match the local env.
- Posters are loaded live from TMDB (`image.tmdb.org`); an internet connection is
  needed for poster images (the app still works without them).
