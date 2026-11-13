"""Stage 3: Collaborative Filtering (SVD) — config comparison, evaluation, examples.

Run from anywhere:
    python scripts/run_cf.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recommender.collaborative import kmeans, svd_model  # noqa: E402
from recommender.preprocessing import config  # noqa: E402

plt.rcParams["font.family"] = ["Tahoma", "Microsoft Sans Serif", "Leelawadee UI"]
plt.rcParams["axes.unicode_minus"] = False

OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
EVID = ROOT / "outputs" / "screenshots" / "chapter4" / "03_collaborative"

CONFIGS = [
    ("A: n_factors=50, n_epochs=30", dict(n_factors=50, n_epochs=30)),
    ("B: n_factors=100, n_epochs=20", dict(n_factors=100, n_epochs=20)),
]
KS = [1, 3, 5, 10]
THRESHOLD = config.LIKE_THRESHOLD


def regression_metrics(model, test_df):
    trues, preds = [], []
    for _, r in test_df.iterrows():
        trues.append(float(r["rating"]))
        preds.append(model.predict(int(r["user_id"]), int(r["movie_id"])))
    t, p = np.array(trues), np.array(preds)
    return float(np.sqrt(np.mean((t - p) ** 2))), float(np.mean(np.abs(t - p)))


def ranking_metrics(score_fn, test_df, train_df, all_movie_ids, ks=KS):
    """Per-user top-k + MRR over candidates = (15 movies - train-rated)."""
    train_rated = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_by_user = {u: dict(zip(g["movie_id"], g["rating"])) for u, g in test_df.groupby("user_id")}

    p_at = {k: [] for k in ks}
    r_at = {k: [] for k in ks}
    f_at = {k: [] for k in ks}
    rr_list = []

    for u in test_df["user_id"].unique():
        excl = set(int(m) for m in train_rated.get(u, set()))
        cands = [m for m in all_movie_ids if int(m) not in excl]
        ranked = sorted(((m, score_fn(int(u), int(m))) for m in cands), key=lambda x: -x[1])
        ranked_ids = [m for m, _ in ranked]

        tset = test_by_user.get(u, {})
        relevant = {int(m) for m, r in tset.items() if r >= THRESHOLD}

        for k in ks:
            top = set(ranked_ids[:k])
            tp = len(top & relevant)
            prec = tp / k
            rec = tp / len(relevant) if relevant else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            p_at[k].append(prec)
            r_at[k].append(rec)
            f_at[k].append(f1)

        if relevant:
            for rank, m in enumerate(ranked_ids, 1):
                if m in relevant:
                    rr_list.append(1.0 / rank)
                    break

    out = {}
    for k in ks:
        out[f"Precision@{k}"] = float(np.mean(p_at[k]))
        out[f"Recall@{k}"] = float(np.mean(r_at[k]))
        out[f"F1@{k}"] = float(np.mean(f_at[k]))
    out["MRR"] = float(np.mean(rr_list)) if rr_list else 0.0
    return out


def main():
    for d in (OUT_FIG, OUT_TAB, EVID):
        d.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(config.RATINGS_TRAIN)
    test = pd.read_csv(config.RATINGS_TEST)
    movies = pd.read_csv(config.MOVIES_CLEAN, keep_default_na=False)
    users = pd.read_csv(config.USERS_CLEAN, keep_default_na=False)
    pop = pd.read_csv(config.MOVIE_POPULARITY)

    id_to_title = dict(zip(movies["id"], movies["title"]))
    id_to_alias = dict(zip(users["user_id"], users["alias"]))
    all_movie_ids = sorted(set(train["movie_id"]) | set(test["movie_id"]))
    global_mean = float(train["rating"].mean())

    rows, models, rank_detail = [], {}, {}
    for label, params in CONFIGS:
        t0 = time.perf_counter()
        m = svd_model.SVDRecommender(**params).fit(train)
        fit_s = time.perf_counter() - t0
        rmse, mae = regression_metrics(m, test)
        rk = ranking_metrics(m.predict, test, train, all_movie_ids)
        rows.append({
            "config": label, "n_factors": params["n_factors"], "n_epochs": params["n_epochs"],
            "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            **{k: round(v, 4) for k, v in rk.items()},
            "fit_s": round(fit_s, 3),
        })
        models[label] = m
        rank_detail[label] = rk

    # popularity reference (ranking only)
    pop_map = dict(zip(pop["movie_id"], pop["train_mean_rating"]))
    rk_pop = ranking_metrics(lambda u, m: float(pop_map.get(m, global_mean)),
                             test, train, all_movie_ids)
    rows.append({"config": "popularity (reference)", "n_factors": None, "n_epochs": None,
                 "RMSE": None, "MAE": None, **{k: round(v, 4) for k, v in rk_pop.items()},
                 "fit_s": None})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "cf_config_comparison.csv", index=False, encoding="utf-8-sig")

    svd_rows = df[df["config"].str.startswith(("A:", "B:"))].copy()
    best = svd_rows.sort_values(["RMSE", "MAE"]).iloc[0]
    best_label = best["config"]

    artifact = config.DATA_PROCESSED / "svd_model.joblib"
    models[best_label].save(artifact)

    # ---- example recommendations (selected model) ----
    example_users = sorted(test["user_id"].unique())[:5]
    ex_lines = ["# Collaborative Filtering (SVD) — example recommendations", ""]
    ex_rows = []
    for u in example_users:
        excl = set(int(m) for m in train[train["user_id"] == u]["movie_id"])
        recs = models[best_label].recommend(u, all_movie_ids, top_n=5, exclude_rated=excl)
        ex_lines.append(f"## user {u} ({id_to_alias.get(u, '?')})")
        ex_lines.append("| rank | movie | predicted rating |")
        ex_lines.append("|---|---|---|")
        for rank, (mid, score) in enumerate(recs, 1):
            ex_lines.append(f"| {rank} | {id_to_title.get(mid, mid)} | {score:.3f} |")
            ex_rows.append({"user_id": u, "alias": id_to_alias.get(u, ""), "rank": rank,
                            "movie_id": mid, "title": id_to_title.get(mid, mid),
                            "predicted_rating": round(score, 3)})
        ex_lines.append("")
    pd.DataFrame(ex_rows).to_csv(OUT_TAB / "cf_topn_examples.csv", index=False, encoding="utf-8-sig")
    (EVID / "cf_examples.md").write_text("\n".join(ex_lines), encoding="utf-8")

    # ---- K-Means K=2 (auxiliary) ----
    ratings_long = pd.read_csv(config.RATINGS_LONG)
    cluster_res, km, pivot = kmeans.cluster_users(ratings_long, n_clusters=2)
    cluster_sizes = cluster_res["cluster"].value_counts().sort_index().to_dict()
    cluster_res.to_csv(OUT_TAB / "user_clusters_k2.csv", index=False, encoding="utf-8-sig")

    # ---- figures ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    labels_short = ["A (50,30)", "B (100,20)"]
    x = np.arange(2)

    ax = axes[0, 0]
    ax.bar(x - 0.2, svd_rows["RMSE"], 0.4, label="RMSE", color="#3b6ea5")
    ax.bar(x + 0.2, svd_rows["MAE"], 0.4, label="MAE", color="#c0504d")
    ax.set_xticks(x); ax.set_xticklabels(labels_short)
    ax.set_title("Rating-prediction error (lower is better)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    w = 0.25
    ax.bar(x - w, svd_rows["Precision@10"], w, label="Precision@10", color="#3b6ea5")
    ax.bar(x, svd_rows["Recall@10"], w, label="Recall@10", color="#6f9b4f")
    ax.bar(x + w, svd_rows["F1@10"], w, label="F1@10", color="#c0504d")
    ax.set_xticks(x); ax.set_xticklabels(labels_short)
    ax.set_title("Top-10 ranking (threshold ≥3.0; capped by ~3–7 candidates)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    for lbl in ["A: n_factors=50, n_epochs=30", "B: n_factors=100, n_epochs=20"]:
        d = rank_detail[lbl]
        ax.plot(KS, [d[f"Precision@{k}"] for k in KS], marker="o", label=lbl.split(":")[0])
    ax.set_xlabel("K"); ax.set_ylabel("Precision@K"); ax.set_title("Precision@K (ranking ability at small K)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar([str(k) for k in sorted(cluster_sizes)], [cluster_sizes[k] for k in sorted(cluster_sizes)], color="#8a7fb8")
    ax.set_title("K-Means (K=2) user cluster sizes (auxiliary)"); ax.set_xlabel("cluster"); ax.set_ylabel("users")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Collaborative Filtering (SVD) — configuration comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG / "cf_svd_comparison.png", dpi=130)
    plt.close(fig)

    print(df.to_string(index=False))
    print("\nSELECTED:", best_label, f"(RMSE={best['RMSE']}, MAE={best['MAE']})")
    print("K-Means cluster sizes:", cluster_sizes)

    _write_report(df, best_label, best, cluster_sizes)
    print("wrote outputs/tables/cf_config_comparison.csv, cf_topn_examples.csv, user_clusters_k2.csv")
    print("wrote outputs/figures/cf_svd_comparison.png")
    print("wrote data/reports/cf_report.md, data/processed/svd_model.joblib")


def _write_report(df, best_label, best, cluster_sizes):
    lines = [
        "# Collaborative Filtering (SVD) — Implementation & Evaluation Report",
        "",
        "## Method",
        "- Library: `scikit-surprise` `SVD` (matrix factorization).",
        "- Trained on `ratings_train.csv` only (1,749 ratings); evaluated on `ratings_test.csv` (439 ratings).",
        "- `Reader(rating_scale=(1,5))`; `lr_all=0.005`, `reg_all=0.02`, `random_state=42`.",
        "- Candidate universe for ranking = the 15 survey movies (the only items with rating history).",
        "- Relevance threshold for ranking = **rating >= 3.0**.",
        "- K-Means K=2 is computed as an auxiliary user clustering (NOT used by SVD).",
        "",
        "## Results",
        "",
        _df_to_md(df),
        "",
        "## Interpretation",
        "- **Regression (RMSE/MAE)**: meaningful, used for selection. B has the lowest error.",
        "- **Top-10 ranking is capped**: per-user candidates = 15 minus train-rated ≈ 3–7 items, so "
        "`Precision@10`/`Recall@10`/`F1@10` are nearly identical across models (top-10 includes all "
        "candidates). They are reported as required but are NOT used for selection.",
        "- **Precision@1/@3 and MRR** measure actual ranking ability at small K and differentiate models.",
        "",
        "## Selection",
        f"- Chosen: **{best_label}** (RMSE={best['RMSE']}, MAE={best['MAE']}).",
        "- Selection rule: lowest RMSE, tie-broken by MAE.",
        "",
        "## Limitations",
        "- Only 15 items have rating history, so SVD can only score those 15 movies; it cannot score "
        "the other 9,826 catalog movies (addressed by CBF/hybrid).",
        "- `n_factors` up to 100 is over-parameterized relative to 15 items; reported as-is for the "
        "required comparison.",
        "- Unknown users / empty history use the popularity fallback (`movie_popularity_train.csv`).",
        "",
        "## Outputs",
        "- `outputs/tables/cf_config_comparison.csv`",
        "- `outputs/tables/cf_topn_examples.csv`, `outputs/tables/user_clusters_k2.csv`",
        "- `outputs/figures/cf_svd_comparison.png`",
        "- `data/processed/svd_model.joblib` (selected model)",
    ]
    (config.DATA_REPORTS / "cf_report.md").write_text("\n".join(lines), encoding="utf-8")


def _df_to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
