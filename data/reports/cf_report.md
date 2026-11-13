# Collaborative Filtering (SVD) — Implementation & Evaluation Report

## Method
- Library: `scikit-surprise` `SVD` (matrix factorization).
- Trained on `ratings_train.csv` only (1,749 ratings); evaluated on `ratings_test.csv` (439 ratings).
- `Reader(rating_scale=(1,5))`; `lr_all=0.005`, `reg_all=0.02`, `random_state=42`.
- Candidate universe for ranking = the 15 survey movies (the only items with rating history).
- Relevance threshold for ranking = **rating >= 3.0**.
- K-Means K=2 is computed as an auxiliary user clustering (NOT used by SVD).

## Results

> **Reference-only columns:** `Precision@10`, `Recall@10`, `F1@10` are reported for completeness
> but NOT used for model selection — the per-user candidate set is ≤10 movies (3–9, mean 5.28),
> so top-10 equals the entire candidate set and these three are ranking-order-independent.

| config | n_factors | n_epochs | RMSE | MAE | Precision@1 | Recall@1 | F1@1 | Precision@3 | Recall@3 | F1@3 | Precision@5 | Recall@5 | F1@5 | Precision@10 | Recall@10 | F1@10 | MRR | fit_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: n_factors=50, n_epochs=30 | 50.0 | 30.0 | 0.8479 | 0.6927 | 0.4944 | 0.2074 | 0.287 | 0.4667 | 0.6102 | 0.5174 | 0.4267 | 0.9472 | 0.5773 | 0.2206 | 0.9833 | 0.3558 | 0.6903 | 0.019 |
| B: n_factors=100, n_epochs=20 | 100.0 | 20.0 | 0.8346 | 0.6826 | 0.5167 | 0.2204 | 0.3028 | 0.4611 | 0.6037 | 0.5109 | 0.4256 | 0.9444 | 0.5757 | 0.2206 | 0.9833 | 0.3558 | 0.699 | 0.016 |
| popularity (reference) |  |  |  |  | 0.5389 | 0.2296 | 0.3157 | 0.4593 | 0.6 | 0.5078 | 0.4267 | 0.9472 | 0.5773 | 0.2206 | 0.9833 | 0.3558 | 0.7125 |  |

## Interpretation
- **Primary prediction metrics: RMSE / MAE** — meaningful, used for selection. B has the lowest error.
- **Primary ranking metrics: Precision@1 / @3 / @5 and MRR** — measure actual ranking ability at small K and differentiate models.
- **Reference-only: Precision@10 / Recall@10 / F1@10** — per-user candidates = 15 minus train-rated = **3–9 items (mean 5.28)**, so top-10 equals the entire candidate set; these metrics are ranking-order-independent and identical across models, and therefore are NOT used for selection.

## Selection
- Chosen: **B: n_factors=100, n_epochs=20** (RMSE=0.8346, MAE=0.6826).
- Selection rule: lowest RMSE, tie-broken by MAE.

## Limitations
- Only 15 items have rating history, so SVD can only score those 15 movies; it cannot score the other 9,826 catalog movies (addressed by CBF/hybrid).
- `n_factors` up to 100 is over-parameterized relative to 15 items; reported as-is for the required comparison.
- Unknown users / empty history use the popularity fallback (`movie_popularity_train.csv`).

## Outputs
- `outputs/tables/cf_config_comparison.csv`
- `outputs/tables/cf_topn_examples.csv`, `outputs/tables/user_clusters_k2.csv`
- `outputs/figures/cf_svd_comparison.png`
- `data/processed/svd_model.joblib` (selected model)