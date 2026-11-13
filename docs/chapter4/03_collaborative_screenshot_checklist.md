# Chapter 4 — Collaborative Filtering (SVD) Screenshot Checklist

Stage 3 evidence. All items are real outputs produced by `python scripts/run_cf.py`.

Evidence directory: `outputs/screenshots/chapter4/03_collaborative/`

## Items to screenshot

- [ ] **1. Configuration comparison table**
      File: `outputs/tables/cf_config_comparison.csv` (also rendered in `data/reports/cf_report.md`)
      Shows RMSE/MAE, Precision/Recall/F1 at K=1/3/5/10, and MRR for config A (50,30),
      config B (100,20), and the popularity reference.

- [ ] **2. Evaluation figure**
      File: `outputs/figures/cf_svd_comparison.png`
      Four panels: rating error (RMSE/MAE), top-10 ranking, Precision@K at small K, and
      K-Means (K=2) cluster sizes.

- [ ] **3. Example recommendations**
      File: `outputs/screenshots/chapter4/03_collaborative/cf_examples.md`
      Top-5 predicted movies for users 1–5.

- [ ] **4. K-Means cluster assignment**
      File: `outputs/tables/user_clusters_k2.csv`
      Auxiliary K=2 user cluster labels (98 / 82 users).

- [ ] **5. Evaluation report**
      File: `data/reports/cf_report.md`
      Method, results, interpretation, selection, and limitations.

## Metric labeling (for captioning)
- **Primary prediction metrics:** RMSE, MAE.
- **Primary ranking metrics:** Precision@1, @3, @5, and MRR.
- **Reference-only (NOT used for selection):** Precision@10, Recall@10, F1@10 — because top-10
  equals the entire per-user candidate set (3–9 items, mean 5.28), these are ranking-order-independent.

## Key results (for captioning)
- Selected: **B (n_factors=100, n_epochs=20)** — RMSE=0.8346, MAE=0.6826.
- Config A (50,30): RMSE=0.8479, MAE=0.6927.
- Ranking (primary, small K): Precision@1 B=0.5167 vs A=0.4944; MRR B=0.6990 vs A=0.6903.
- Popularity reference is competitive at K=1 (0.5389 / MRR 0.7125) — an honest small-data finding.
