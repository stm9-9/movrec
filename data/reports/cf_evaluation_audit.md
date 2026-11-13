# CF Evaluation Audit (Stage 3.1)

Read-only audit. No raw data, processed data, or model artifacts were modified.

## 1. Train/test separation (leakage)
- train ratings: 1749; test ratings: 439.
- overlapping (user, movie) pairs between train and test: **0** (must be 0).
- users in train: 180; in test: 180; test users not seen in train: 0.
- movies in train: 15; in test: 14; test movies not seen in train: 0.
- SVD is fit on `ratings_train.csv` only (`Dataset.load_from_df(train_df)` + `build_full_trainset()`); test is only used in `predict()` / ground truth.

## 2. Popularity baseline is train-only
- `movie_popularity_train.csv` recomputed from train: max |diff| = 4.44e-16 (must be ~0).

## 3. Candidate-set construction (why it is small)
- `all_movie_ids` = union of train+test movie ids = **15 survey movies**.
- For each user, candidates = all_movie_ids minus the movies they rated in TRAIN.
- Because each user rated 8–15 movies and ~80% of those are in train, only `15 − |train_rated|` items remain as candidates.
- Candidate count per user: min=3, max=9, mean=5.28, median=5.
- Distribution: {3: 8, 4: 23, 5: 97, 6: 27, 7: 14, 8: 9, 9: 2}
- **180/180 users have <= 10 candidates**, so top-10 includes the entire candidate set for every user.

## 4. Is Precision@10 / Recall@10 degenerate?
- Mean relevant test items per user (rating >= 3): 2.206.
- Users with 0 relevant test items: 3.
- Since top-10 == all candidates for every user, TP@10 == |relevant|, so:
  - Precision@10 ≈ mean(|relevant|)/10 = 0.2206 (matches the reported 0.2206).
  - Recall@10 ≈ share of users with >=1 relevant = 0.9833 (matches the reported 0.9833).
- Conclusion: **@10 does not depend on ranking order** for any model, so it cannot compare SVD vs popularity. It is reported but should NOT be used for model selection.

## 5. Do SVD and popularity use the same candidate sets?
- Yes. Both call the same `ranking_metrics()` with the same `train_df`, `test_df`, and `all_movie_ids`; `excl` is derived identically from train. Only the `score_fn` differs (SVD predicts per-user; popularity ignores the user).
- Candidate sets are therefore identical per user — a fair comparison.

## 6. Relevance threshold (rating >= 3)
- Rating distribution: {1.0: 14, 2.0: 229, 3.0: 858, 4.0: 849, 5.0: 238}
- Share of ratings >= 3: 0.889 (positive-class imbalance).
- With threshold 3, most test items are 'relevant', so Recall@10 is inflated toward 1 and the threshold choice mainly affects Precision.

## 7. Metric formulas (verified against definitions)
- Precision@K = |topK ∩ relevant| / K  — correct.
- Recall@K = |topK ∩ relevant| / |relevant| (0 when no relevant) — correct.
- F1@K = 2PR/(P+R) — correct.
- MRR = mean over users of 1/rank(first relevant item) — correct (users with no relevant item skipped).

## 8. Identified issues and recommended corrections
1. **Issue:** Precision/Recall/F1@10 are degenerate (top-10 == all ~3–9 candidates), so they measure nothing about ranking and are identical for SVD and popularity.
   **Correction (proposed, not yet applied):** report ranking at K ≤ 5 (e.g. @1/@3/@5) as the primary ranking metric, and label @10 as 'capped/for-reference only'.
2. **Issue:** never-rated movies are included as candidates but have no ground-truth label, so they can never count as hits (they act as unlabeled distractors).
   **Correction (optional):** add a leave-one-out evaluation over test items only (rank the 2–3 held-out items; report Precision@1 / MRR / NDCG).
3. **Recommendation:** keep **RMSE/MAE as the primary CF metric** (meaningful, standard for SVD, and differentiates the two configs). Use ranking metrics as supplementary.

> Current Stage 3 results are preserved unchanged as the original baseline. No correction has been applied yet (awaiting approval).
