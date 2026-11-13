# Chapter 4 — Hybrid Recommendation Screenshot Checklist

Stage 4 evidence. All items are real outputs produced by `python scripts/run_hybrid.py`.

Evidence directory: `outputs/screenshots/chapter4/04_hybrid/`

## Items to screenshot

- [ ] **1. Method comparison table**
      File: `outputs/tables/hybrid_config_comparison.csv` (also in `data/reports/hybrid_report.md`)
      Compares Popularity, CF (SVD B), CBF v2, and Hybrid A/B/C on RMSE/MAE, Precision@1/3/5,
      MRR, Precision@10/Recall@10/F1@10, and diversity@3/@5/@10.

- [ ] **2. Comparison figure**
      File: `outputs/figures/hybrid_comparison.png`
      Three panels: Precision@1/3/5, MRR, and diversity@3/@5 across all six methods.

- [ ] **3. Example recommendations (best hybrid)**
      File: `outputs/screenshots/chapter4/04_hybrid/hybrid_examples.md`
      Top-5 hybrid recommendations for users 1–5 (Hybrid A).

- [ ] **4. Evaluation report**
      File: `data/reports/hybrid_report.md`
      Method, results, interpretation, selection, and limitations.

## Key results (for captioning)
- **Best hybrid: Hybrid A (α=0.3 CF / β=0.7 CBF)** — MRR=0.7245, P@1=0.5444.
- Best single model: Popularity (MRR=0.7125, P@1=0.5389).
- Hybrid A is **marginally higher** on P@1 (+0.0055 ≈ 1 user of 180) and MRR (+0.012). No
  significance test was performed, so report this as "marginally higher, within a margin that may
  not be statistically reliable" — not as "beating every single model."
- The win is limited to P@1 and MRR: Hybrid A is lower than Popularity on P@5 (0.4178 vs 0.4267).
- A vs B justification: P@1/MRR are the primary ranking metrics (Stage 3.1 protocol); Hybrid B
  leads only on P@3 (0.4852 vs 0.4759), a secondary metric, so Hybrid A is selected.
- Data-driven note: results favor a CBF-heavy blend (0.3/0.7), not the thesis's assumed 0.6/0.4.
- Diversity@3: CBF most diverse (0.8708), popularity least (0.8241); hybrids in between (A=0.8630).

## Metric labeling
- Primary prediction: RMSE/MAE (CF only, the sole rating predictor).
- Primary ranking: Precision@1/@3/@5 and MRR.
- Reference-only: Precision@10/Recall@10/F1@10 (top-10 == entire candidate set) and diversity@10
  (ranking-order-independent); diversity@3/@5 are the meaningful diversity signals.
