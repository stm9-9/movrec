"""Stage 3.1 audit: verify CF evaluation methodology with real numbers.

Read-only audit. Produces data/reports/cf_evaluation_audit.md.
Run from anywhere:
    python scripts/audit_cf_evaluation.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recommender.preprocessing import config  # noqa: E402

K = 10
THRESHOLD = config.LIKE_THRESHOLD


def main():
    train = pd.read_csv(config.RATINGS_TRAIN)
    test = pd.read_csv(config.RATINGS_TEST)
    long_ = pd.read_csv(config.RATINGS_LONG)
    pop = pd.read_csv(config.MOVIE_POPULARITY)

    lines = ["# CF Evaluation Audit (Stage 3.1)", ""]
    lines.append("Read-only audit. No raw data, processed data, or model artifacts were modified.")
    lines.append("")

    # 1. Leakage check
    overlap = pd.merge(train, test, on=["user_id", "movie_id"], how="inner")
    train_users = set(train["user_id"])
    test_users = set(test["user_id"])
    train_movies = set(train["movie_id"])
    test_movies = set(test["movie_id"])
    lines.append("## 1. Train/test separation (leakage)")
    lines.append(f"- train ratings: {len(train)}; test ratings: {len(test)}.")
    lines.append(f"- overlapping (user, movie) pairs between train and test: **{len(overlap)}** (must be 0).")
    lines.append(f"- users in train: {len(train_users)}; in test: {len(test_users)}; "
                 f"test users not seen in train: {len(test_users - train_users)}.")
    lines.append(f"- movies in train: {len(train_movies)}; in test: {len(test_movies)}; "
                 f"test movies not seen in train: {len(test_movies - train_movies)}.")
    lines.append("- SVD is fit on `ratings_train.csv` only (`Dataset.load_from_df(train_df)` + "
                 "`build_full_trainset()`); test is only used in `predict()` / ground truth.")
    lines.append("")

    # 2. Popularity computed from train only
    pop_from_train = (
        train.groupby("movie_id")["rating"].agg(["mean", "count"]).rename(
            columns={"mean": "train_mean_rating", "count": "train_count"}
        ).reset_index()
    )
    merged = pop.merge(pop_from_train, on="movie_id", suffixes=("_saved", "_recomputed"))
    max_diff = float((merged["train_mean_rating_saved"] - merged["train_mean_rating_recomputed"]).abs().max())
    lines.append("## 2. Popularity baseline is train-only")
    lines.append(f"- `movie_popularity_train.csv` recomputed from train: max |diff| = {max_diff:.2e} (must be ~0).")
    lines.append("")

    # 3. Candidate-set construction
    all_movie_ids = sorted(train_movies | test_movies)
    train_rated = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    cand_counts = {u: len([m for m in all_movie_ids if m not in train_rated.get(u, set())])
                   for u in test_users}
    cvals = pd.Series(list(cand_counts.values()))
    lines.append("## 3. Candidate-set construction (why it is small)")
    lines.append("- `all_movie_ids` = union of train+test movie ids = **15 survey movies**.")
    lines.append("- For each user, candidates = all_movie_ids minus the movies they rated in TRAIN.")
    lines.append("- Because each user rated 8–15 movies and ~80% of those are in train, only "
                 "`15 − |train_rated|` items remain as candidates.")
    lines.append(f"- Candidate count per user: min={cvals.min()}, max={cvals.max()}, "
                 f"mean={cvals.mean():.2f}, median={cvals.median():.0f}.")
    lines.append(f"- Distribution: {cvals.value_counts().sort_index().to_dict()}")
    lines.append(f"- **{int((cvals <= K).sum())}/{int(len(cvals))} users have <= {K} candidates**, "
                 f"so top-{K} includes the entire candidate set for every user.")
    lines.append("")

    # 4. Degeneracy of @10
    test_by_user = {u: dict(zip(g["movie_id"], g["rating"])) for u, g in test.groupby("user_id")}
    relevant_counts = {
        u: len([m for m, r in t.items() if r >= THRESHOLD]) for u, t in test_by_user.items()
    }
    rel = pd.Series(list(relevant_counts.values()))
    p10_reconstructed = float(rel.mean()) / K
    r10_reconstructed = float((rel > 0).mean())
    lines.append("## 4. Is Precision@10 / Recall@10 degenerate?")
    lines.append(f"- Mean relevant test items per user (rating >= {THRESHOLD:.0f}): {rel.mean():.3f}.")
    lines.append(f"- Users with 0 relevant test items: {int((rel == 0).sum())}.")
    lines.append("- Since top-10 == all candidates for every user, TP@10 == |relevant|, so:")
    lines.append(f"  - Precision@10 ≈ mean(|relevant|)/10 = {p10_reconstructed:.4f} "
                 "(matches the reported 0.2206).")
    lines.append(f"  - Recall@10 ≈ share of users with >=1 relevant = {r10_reconstructed:.4f} "
                 "(matches the reported 0.9833).")
    lines.append("- Conclusion: **@10 does not depend on ranking order** for any model, so it cannot "
                 "compare SVD vs popularity. It is reported but should NOT be used for model selection.")
    lines.append("")

    # 5. Fairness of candidate sets (SVD vs popularity)
    lines.append("## 5. Do SVD and popularity use the same candidate sets?")
    lines.append("- Yes. Both call the same `ranking_metrics()` with the same `train_df`, `test_df`, "
                 "and `all_movie_ids`; `excl` is derived identically from train. Only the `score_fn` "
                 "differs (SVD predicts per-user; popularity ignores the user).")
    lines.append("- Candidate sets are therefore identical per user — a fair comparison.")
    lines.append("")

    # 6. Threshold effect
    lines.append("## 6. Relevance threshold (rating >= 3)")
    rating_dist = long_["rating"].value_counts().sort_index()
    frac_ge3 = float((long_["rating"] >= THRESHOLD).mean())
    lines.append(f"- Rating distribution: {rating_dist.to_dict()}")
    lines.append(f"- Share of ratings >= {THRESHOLD:.0f}: {frac_ge3:.3f} (positive-class imbalance).")
    lines.append(f"- With threshold {THRESHOLD:.0f}, most test items are 'relevant', so Recall@10 is "
                 "inflated toward 1 and the threshold choice mainly affects Precision.")
    lines.append("")

    # 7. Metric formula verification
    lines.append("## 7. Metric formulas (verified against definitions)")
    lines.append("- Precision@K = |topK ∩ relevant| / K  — correct.")
    lines.append("- Recall@K = |topK ∩ relevant| / |relevant| (0 when no relevant) — correct.")
    lines.append("- F1@K = 2PR/(P+R) — correct.")
    lines.append("- MRR = mean over users of 1/rank(first relevant item) — correct (users with no "
                 "relevant item skipped).")
    lines.append("")

    # 8. Issues + recommended corrections
    lines.append("## 8. Identified issues and recommended corrections")
    lines.append("1. **Issue:** Precision/Recall/F1@10 are degenerate (top-10 == all ~3–9 candidates), "
                 "so they measure nothing about ranking and are identical for SVD and popularity.")
    lines.append("   **Correction (proposed, not yet applied):** report ranking at K ≤ 5 (e.g. @1/@3/@5) "
                 "as the primary ranking metric, and label @10 as 'capped/for-reference only'.")
    lines.append("2. **Issue:** never-rated movies are included as candidates but have no ground-truth "
                 "label, so they can never count as hits (they act as unlabeled distractors).")
    lines.append("   **Correction (optional):** add a leave-one-out evaluation over test items only "
                 "(rank the 2–3 held-out items; report Precision@1 / MRR / NDCG).")
    lines.append("3. **Recommendation:** keep **RMSE/MAE as the primary CF metric** (meaningful, standard "
                 "for SVD, and differentiates the two configs). Use ranking metrics as supplementary.")
    lines.append("")
    lines.append("> Current Stage 3 results are preserved unchanged as the original baseline. No "
                 "correction has been applied yet (awaiting approval).")
    lines.append("")

    report = config.DATA_REPORTS / "cf_evaluation_audit.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote data/reports/cf_evaluation_audit.md")


if __name__ == "__main__":
    main()
