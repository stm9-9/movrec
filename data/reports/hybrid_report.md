# Hybrid Recommendation — Implementation & Evaluation Report

## Method
- Models: CF = SVD B (`n_factors=100, n_epochs=20`) from `svd_model.joblib`; CBF v2 (`w=1`) from `cbf_model_v2.joblib`.
- Candidate set: same as Stage 3 — the 15 survey movies minus the user's train-rated movies.
- CF score = SVD predicted rating; CBF score = rating-weighted mean cosine similarity of the candidate to the user's train-rated movies.
- Each score source is **min-max normalized to [0,1] per user**; blend = α·CF_norm + β·CBF_norm.
- Weights: Hybrid A (0.3/0.7), B (0.5/0.5), C (0.7/0.3).
- Relevance threshold = rating ≥ 3.0.
- Diversity@K = 1 − mean pairwise genre Jaccard similarity within the top-K list.

## Results

| method | RMSE | MAE | Precision@1 | Precision@3 | Precision@5 | MRR | Precision@10 | Recall@10 | F1@10 | diversity@3 | diversity@5 | diversity@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Popularity |  |  | 0.5389 | 0.4593 | 0.4267 | 0.7125 | 0.2206 | 0.9833 | 0.3558 | 0.8241 | 0.8614 | 0.8685 |
| CF (SVD B) | 0.8346 | 0.6826 | 0.5167 | 0.4611 | 0.4256 | 0.699 | 0.2206 | 0.9833 | 0.3558 | 0.8349 | 0.8624 | 0.8685 |
| CBF v2 |  |  | 0.5111 | 0.4481 | 0.4189 | 0.7042 | 0.2206 | 0.9833 | 0.3558 | 0.8708 | 0.8651 | 0.8685 |
| Hybrid A (0.3/0.7) |  |  | 0.5444 | 0.4759 | 0.4178 | 0.7245 | 0.2206 | 0.9833 | 0.3558 | 0.863 | 0.8638 | 0.8685 |
| Hybrid B (0.5/0.5) |  |  | 0.5333 | 0.4852 | 0.4189 | 0.7184 | 0.2206 | 0.9833 | 0.3558 | 0.8548 | 0.8638 | 0.8685 |
| Hybrid C (0.7/0.3) |  |  | 0.5278 | 0.4741 | 0.4244 | 0.7102 | 0.2206 | 0.9833 | 0.3558 | 0.8453 | 0.8632 | 0.8685 |

## Interpretation
- **Primary prediction metrics (RMSE/MAE)** apply only to CF (SVD B), the only rating predictor.
- **Primary ranking metrics: Precision@1/@3/@5 and MRR.**
- **Reference-only: Precision@10/Recall@10/F1@10** (top-10 == entire candidate set).
- **Diversity**: diversity@3/@5 are the meaningful signals (top lists differ across methods); diversity@10 is ranking-order-independent and identical across methods for the same reason.

## Selection
- Best hybrid: **Hybrid A (0.3/0.7)** (MRR=0.7245, P@1=0.5444).
- Best single model: **Popularity** (MRR=0.7125, P@1=0.5389).
- Selection rule: highest MRR among the three hybrid weights, tie-broken by Precision@1.
- **A vs B justification:** P@1 and MRR are the primary ranking metrics (per the Stage 3.1
  protocol); P@3 is secondary. Hybrid B leads only on P@3 (0.4852 vs 0.4759), which is not the
  primary selection criterion, so Hybrid A is chosen.
- **Scope of the win:** the improvement is limited to top-1 ranking (P@1) and MRR. Hybrid A is
  **lower** than Popularity on P@5 (0.4178 vs 0.4267), so it does NOT win on all ranking metrics.

> Data-driven note: the thesis design proposed a 0.6 CF / 0.4 CBF weighting, but the empirical results favor the CBF-heavy **Hybrid A (0.3 CF / 0.7 CBF)**. The selection is based on actual results, not on the thesis's assumed weights.

## Statistical caution
- The P@1 gap between Hybrid A (0.5444) and Popularity (0.5389) equals ~1 user out of 180
  (0.0055 × 180 ≈ 1).
- The MRR gap is 0.012.
- No significance test (e.g., paired t-test, bootstrap) was performed.
- Therefore Hybrid A's advantage should be described as "marginally higher on P@1 and MRR, within a
  margin that may not be statistically reliable" — NOT as "beating every single model."

## Outputs
- `outputs/tables/hybrid_config_comparison.csv`
- `outputs/tables/hybrid_topn_examples.csv`
- `outputs/figures/hybrid_comparison.png`
- `outputs/screenshots/chapter4/04_hybrid/`

## Limitations
- Hybrid is evaluated on the same 15-movie candidate set as CF, so it cannot yet recommend from the full 9,841-movie catalog (CBF can; CF cannot). Web integration will use CBF for the full catalog and CF/hybrid for the 15 known items.
- CF and CBF are min-max normalized per user; with ≤9 candidates the normalization is coarse.