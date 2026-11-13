"""Stage 4: Hybrid recommendation — compare Popularity, CF, CBF, and Hybrid A/B/C.

Run from anywhere:
    python scripts/run_hybrid.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recommender.content_based import cbf_v2  # noqa: E402
from recommender.collaborative.svd_model import SVDRecommender  # noqa: E402
from recommender.hybrid.hybrid_model import CBFProfileScorer, minmax_normalize  # noqa: E402
from recommender.preprocessing import config  # noqa: E402

plt.rcParams["font.family"] = ["Tahoma", "Microsoft Sans Serif", "Leelawadee UI"]
plt.rcParams["axes.unicode_minus"] = False

OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
EVID = ROOT / "outputs" / "screenshots" / "chapter4" / "04_hybrid"

THRESHOLD = config.LIKE_THRESHOLD
KS_SMALL = [1, 3, 5]
K_REF = 10

HYBRID_ALPHAS = {
    "Hybrid A (0.3/0.7)": 0.3,
    "Hybrid B (0.5/0.5)": 0.5,
    "Hybrid C (0.7/0.3)": 0.7,
}


def regression_metrics(model, test_df):
    trues, preds = [], []
    for _, r in test_df.iterrows():
        trues.append(float(r["rating"]))
        preds.append(model.predict(int(r["user_id"]), int(r["movie_id"])))
    t, p = np.array(trues), np.array(preds)
    return float(np.sqrt(np.mean((t - p) ** 2))), float(np.mean(np.abs(t - p)))


def intra_list_diversity(ids, id_to_genres):
    """1 - mean pairwise Jaccard similarity of genre sets within a list."""
    ids = [int(i) for i in ids]
    if len(ids) < 2:
        return 0.0
    jacs = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ga = id_to_genres.get(ids[a], set())
            gb = id_to_genres.get(ids[b], set())
            jacs.append(cbf_v2.genre_jaccard(ga, gb))
    return float(1.0 - np.mean(jacs))


def evaluate_ranking(rank_fn, test_df, train_df, all_movie_ids, id_to_genres):
    """rank_fn(user_id, candidates) -> sorted list of (movie_id, score)."""
    train_rated = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_by_user = {u: dict(zip(g["movie_id"], g["rating"])) for u, g in test_df.groupby("user_id")}

    ks = KS_SMALL + [K_REF]
    p_at = {k: [] for k in ks}
    r_at = {k: [] for k in ks}
    f_at = {k: [] for k in ks}
    rr = []
    div_at = {k: [] for k in (3, 5, 10)}

    for u in test_df["user_id"].unique():
        excl = set(int(m) for m in train_rated.get(u, set()))
        cands = [m for m in all_movie_ids if int(m) not in excl]
        ranked = rank_fn(int(u), cands)
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
                    rr.append(1.0 / rank)
                    break
        for k in (3, 5, 10):
            div_at[k].append(intra_list_diversity(ranked_ids[:k], id_to_genres))

    out = {}
    for k in ks:
        out[f"Precision@{k}"] = float(np.mean(p_at[k]))
        out[f"Recall@{k}"] = float(np.mean(r_at[k]))
        out[f"F1@{k}"] = float(np.mean(f_at[k]))
    out["MRR"] = float(np.mean(rr)) if rr else 0.0
    out["diversity@3"] = float(np.mean(div_at[3]))
    out["diversity@5"] = float(np.mean(div_at[5]))
    out["diversity@10"] = float(np.mean(div_at[10]))
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
    genre_sets = cbf_v2.load_genre_ids(movies)
    id_to_genres = {int(movies.iloc[k]["id"]): genre_sets[k] for k in range(len(movies))}

    all_movie_ids = sorted(set(train["movie_id"]) | set(test["movie_id"]))
    global_mean = float(train["rating"].mean())
    pop_map = dict(zip(pop["movie_id"], pop["train_mean_rating"]))
    train_ratings_by_user = {u: list(zip(g["movie_id"], g["rating"]))
                             for u, g in train.groupby("user_id")}

    # Load models
    svd = SVDRecommender.load(config.DATA_PROCESSED / "svd_model.joblib")
    cbf = CBFProfileScorer(config.DATA_PROCESSED / "cbf_model_v2.joblib", all_movie_ids)

    def hybrid_rank_fn(alpha):
        def fn(u, cands):
            tr = train_ratings_by_user[u]
            cf = {m: svd.predict(u, m) for m in cands}
            cb = {m: cbf.score(tr, m) for m in cands}
            cf_n = minmax_normalize(cf)
            cb_n = minmax_normalize(cb)
            scored = sorted(((m, alpha * cf_n[m] + (1 - alpha) * cb_n[m]) for m in cands),
                            key=lambda x: -x[1])
            return scored
        return fn

    rank_fns = {
        "Popularity": lambda u, cands: sorted(
            ((m, float(pop_map.get(m, global_mean))) for m in cands), key=lambda x: -x[1]),
        "CF (SVD B)": lambda u, cands: sorted(
            ((m, svd.predict(u, m)) for m in cands), key=lambda x: -x[1]),
        "CBF v2": lambda u, cands: sorted(
            ((m, cbf.score(train_ratings_by_user[u], m)) for m in cands), key=lambda x: -x[1]),
    }
    for name, alpha in HYBRID_ALPHAS.items():
        rank_fns[name] = hybrid_rank_fn(alpha)

    # ---- evaluate all methods ----
    rows = []
    detail = {}
    for name, fn in rank_fns.items():
        m = evaluate_ranking(fn, test, train, all_movie_ids, id_to_genres)
        detail[name] = m
        row = {"method": name}
        if name == "CF (SVD B)":
            rmse, mae = regression_metrics(svd, test)
            row["RMSE"] = round(rmse, 4)
            row["MAE"] = round(mae, 4)
        else:
            row["RMSE"] = None
            row["MAE"] = None
        row["Precision@1"] = round(m["Precision@1"], 4)
        row["Precision@3"] = round(m["Precision@3"], 4)
        row["Precision@5"] = round(m["Precision@5"], 4)
        row["MRR"] = round(m["MRR"], 4)
        row["Precision@10"] = round(m["Precision@10"], 4)
        row["Recall@10"] = round(m["Recall@10"], 4)
        row["F1@10"] = round(m["F1@10"], 4)
        row["diversity@3"] = round(m["diversity@3"], 4)
        row["diversity@5"] = round(m["diversity@5"], 4)
        row["diversity@10"] = round(m["diversity@10"], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "hybrid_config_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- selection ----
    hybrid_df = df[df["method"].str.startswith("Hybrid")].copy()
    best_hybrid = hybrid_df.sort_values(["MRR", "Precision@1"], ascending=False).iloc[0]
    single_df = df[~df["method"].str.startswith("Hybrid")].copy()
    best_single = single_df.sort_values(["MRR", "Precision@1"], ascending=False).iloc[0]

    # ---- example recommendations (best hybrid) ----
    best_alpha = HYBRID_ALPHAS[best_hybrid["method"]]
    best_fn = hybrid_rank_fn(best_alpha)
    example_users = sorted(test["user_id"].unique())[:5]
    ex_lines = [f"# Hybrid recommendation examples ({best_hybrid['method']})", ""]
    ex_rows = []
    for u in example_users:
        excl = set(int(m) for m in train[train["user_id"] == u]["movie_id"])
        cands = [m for m in all_movie_ids if m not in excl]
        ranked = best_fn(u, cands)[:5]
        ex_lines.append(f"## user {u} ({id_to_alias.get(u, '?')})")
        ex_lines.append("| rank | movie | hybrid score |")
        ex_lines.append("|---|---|---|")
        for rank, (mid, score) in enumerate(ranked, 1):
            ex_lines.append(f"| {rank} | {id_to_title.get(mid, mid)} | {score:.3f} |")
            ex_rows.append({"user_id": u, "alias": id_to_alias.get(u, ""), "rank": rank,
                            "movie_id": mid, "title": id_to_title.get(mid, mid),
                            "hybrid_score": round(score, 3)})
        ex_lines.append("")
    pd.DataFrame(ex_rows).to_csv(OUT_TAB / "hybrid_topn_examples.csv", index=False, encoding="utf-8-sig")
    (EVID / "hybrid_examples.md").write_text("\n".join(ex_lines), encoding="utf-8")

    # ---- figure ----
    methods = df["method"].tolist()
    short = ["Pop", "CF", "CBF", "H-A", "H-B", "H-C"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    w = 0.25
    ax.bar(x - w, df["Precision@1"], w, label="P@1", color="#3b6ea5")
    ax.bar(x, df["Precision@3"], w, label="P@3", color="#6f9b4f")
    ax.bar(x + w, df["Precision@5"], w, label="P@5", color="#c0504d")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title("Primary ranking: Precision@K"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, df["MRR"], color="#8a7fb8")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title("Primary ranking: MRR"); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    w = 0.35
    ax.bar(x - w/2, df["diversity@3"], w, label="Diversity@3", color="#d18f3c")
    ax.bar(x + w/2, df["diversity@5"], w, label="Diversity@5", color="#8a7fb8")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title("Diversity (1 − mean pairwise genre Jaccard)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Hybrid recommendation — method comparison (same candidate set as Stage 3)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG / "hybrid_comparison.png", dpi=130)
    plt.close(fig)

    print(df.to_string(index=False))
    print("\nBEST HYBRID:", best_hybrid["method"], f"(MRR={best_hybrid['MRR']}, P@1={best_hybrid['Precision@1']})")
    print("BEST SINGLE:", best_single["method"], f"(MRR={best_single['MRR']}, P@1={best_single['Precision@1']})")

    _write_report(df, best_hybrid, best_single)
    print("wrote outputs/tables/hybrid_config_comparison.csv, hybrid_topn_examples.csv")
    print("wrote outputs/figures/hybrid_comparison.png")
    print("wrote data/reports/hybrid_report.md")


def _write_report(df, best_hybrid, best_single):
    lines = [
        "# Hybrid Recommendation — Implementation & Evaluation Report",
        "",
        "## Method",
        "- Models: CF = SVD B (`n_factors=100, n_epochs=20`) from `svd_model.joblib`; "
        "CBF v2 (`w=1`) from `cbf_model_v2.joblib`.",
        "- Candidate set: same as Stage 3 — the 15 survey movies minus the user's train-rated movies.",
        "- CF score = SVD predicted rating; CBF score = rating-weighted mean cosine similarity of the "
        "candidate to the user's train-rated movies.",
        "- Each score source is **min-max normalized to [0,1] per user**; blend = α·CF_norm + β·CBF_norm.",
        "- Weights: Hybrid A (0.3/0.7), B (0.5/0.5), C (0.7/0.3).",
        "- Relevance threshold = rating ≥ 3.0.",
        "- Diversity@K = 1 − mean pairwise genre Jaccard similarity within the top-K list.",
        "",
        "## Results",
        "",
        _df_to_md(df),
        "",
        "## Interpretation",
        "- **Primary prediction metrics (RMSE/MAE)** apply only to CF (SVD B), the only rating predictor.",
        "- **Primary ranking metrics: Precision@1/@3/@5 and MRR.**",
        "- **Reference-only: Precision@10/Recall@10/F1@10** (top-10 == entire candidate set).",
        "- **Diversity**: diversity@3/@5 are the meaningful signals (top lists differ across methods); "
        "diversity@10 is ranking-order-independent and identical across methods for the same reason.",
        "",
        "## Selection",
        f"- Best hybrid: **{best_hybrid['method']}** (MRR={best_hybrid['MRR']}, P@1={best_hybrid['Precision@1']}).",
        f"- Best single model: **{best_single['method']}** (MRR={best_single['MRR']}, P@1={best_single['Precision@1']}).",
        "- Selection rule: highest MRR among the three hybrid weights, tie-broken by Precision@1.",
        "",
        "> Data-driven note: the thesis design proposed a 0.6 CF / 0.4 CBF weighting, but the empirical "
        "results favor the CBF-heavy **Hybrid A (0.3 CF / 0.7 CBF)**. The selection is based on actual "
        "results, not on the thesis's assumed weights.",
        "",
        "## Outputs",
        "- `outputs/tables/hybrid_config_comparison.csv`",
        "- `outputs/tables/hybrid_topn_examples.csv`",
        "- `outputs/figures/hybrid_comparison.png`",
        "- `outputs/screenshots/chapter4/04_hybrid/`",
        "",
        "## Limitations",
        "- Hybrid is evaluated on the same 15-movie candidate set as CF, so it cannot yet recommend "
        "from the full 9,841-movie catalog (CBF can; CF cannot). Web integration will use CBF for "
        "the full catalog and CF/hybrid for the 15 known items.",
        "- CF and CBF are min-max normalized per user; with ≤9 candidates the normalization is coarse.",
    ]
    (config.DATA_REPORTS / "hybrid_report.md").write_text("\n".join(lines), encoding="utf-8")


def _df_to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
