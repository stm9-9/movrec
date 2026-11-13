"""Stage 2.1 experiment: compare CBF v1 vs improved feature representations.

Run from anywhere:
    python scripts/run_cbf_v2_experiment.py
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
import scipy.sparse as sp  # noqa: E402

from recommender.content_based import cbf_v2  # noqa: E402
from recommender.content_based.tfidf_model import ContentBasedModel  # noqa: E402
from recommender.preprocessing import config, load_data  # noqa: E402

plt.rcParams["font.family"] = ["Tahoma", "Microsoft Sans Serif", "Leelawadee UI"]
plt.rcParams["axes.unicode_minus"] = False

OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
EVID = ROOT / "outputs" / "screenshots" / "chapter4" / "02_content_based"

SOURCE_MOVIES = [
    (299534, "Avengers: Endgame"),
    (597, "Titanic"),
    (496243, "Parasite"),
    (372058, "Your Name"),
    (157336, "Interstellar"),
    (138843, "The Conjuring"),
]
WEIGHTS = [1, 2, 4, 8, 16]


def main():
    for d in (OUT_FIG, OUT_TAB, EVID):
        d.mkdir(parents=True, exist_ok=True)

    movies = pd.read_csv(config.MOVIES_CLEAN, keep_default_na=False)
    genre_ids_sorted = sorted(g["id"] for g in load_data.load_genres())

    # Shared content reference: overview character n-gram TF-IDF.
    overview_vec, overview_tfidf = cbf_v2.build_overview_tfidf(movies)
    id_to_index = {int(i): k for k, i in enumerate(movies["id"])}
    genre_sets = cbf_v2.load_genre_ids(movies)
    id_to_genres = {int(movies.iloc[k]["id"]): genre_sets[k] for k in range(len(movies))}

    # Feature building blocks.
    genre_onehot = cbf_v2.build_genre_onehot(movies, genre_ids_sorted)

    def top10_from_matrix(feature_matrix):
        sim = cbf_v2.TopKSimilarity(feature_matrix, movies["id"].tolist())
        return sim

    # v1 model
    t0 = time.perf_counter()
    v1_model = ContentBasedModel(movies, max_features=5000)
    v1_build = time.perf_counter() - t0

    configs = []

    # (label, feature_matrix or "v1", build_time, query_time, recs_by_source)
    def run(label, matrix_or_v1, build_time):
        q0 = time.perf_counter()
        recs = {}
        for mid, lbl in SOURCE_MOVIES:
            if matrix_or_v1 == "v1":
                r = v1_model.similar_movies(mid, top_n=10)
                recs[mid] = [x["movie_id"] for x in r]
            else:
                sim = matrix_or_v1
                recs[mid] = [x["movie_id"] for x in sim.similar(mid, top_n=10)]
        qtime = time.perf_counter() - q0
        return recs, qtime

    def evaluate(recs):
        """Return aggregate metrics over all (source, rec) pairs."""
        jacs, cs, dup_total, src_present = [], [], 0, 0
        for mid, _ in SOURCE_MOVIES:
            rids = recs[mid]
            if len(rids) != len(set(rids)):
                dup_total += 1
            if mid in rids:
                src_present += 1
            for rid in rids:
                jacs.append(cbf_v2.genre_jaccard(id_to_genres[mid], id_to_genres[rid]))
                cs.append(cbf_v2.content_cosine(overview_tfidf, id_to_index, mid, rid))
        return {
            "genre_jaccard": float(np.mean(jacs)),
            "content_sim": float(np.mean(cs)),
            "f1_balance": float(2 * np.mean(jacs) * np.mean(cs) / (np.mean(jacs) + np.mean(cs))),
            "lists_with_dup": dup_total,
            "lists_with_source": src_present,
        }

    rows = []
    details = []

    # v1
    recs, q = run("v1", "v1", v1_build)
    rows.append({"config": "v1 (combined word TF-IDF)", "build_s": round(v1_build, 3), "query_s": round(q, 4), **evaluate(recs)})
    _append_details(details, "v1", recs, movies, id_to_index)

    # genre-only
    t0 = time.perf_counter()
    g = top10_from_matrix(genre_onehot)
    bt = time.perf_counter() - t0
    recs, q = run("genre-only", g, bt)
    rows.append({"config": "genre-only (one-hot)", "build_s": round(bt, 3), "query_s": round(q, 4), **evaluate(recs)})
    _append_details(details, "genre-only", recs, movies, id_to_index)

    # overview-only
    t0 = time.perf_counter()
    o = top10_from_matrix(overview_tfidf)
    bt = time.perf_counter() - t0
    recs, q = run("overview-only", o, bt)
    rows.append({"config": "overview-only (char 3-5 TF-IDF)", "build_s": round(bt, 3), "query_s": round(q, 4), **evaluate(recs)})
    _append_details(details, "overview-only", recs, movies, id_to_index)

    # v2 sweep
    v2_recs_by_w = {}
    for w in WEIGHTS:
        t0 = time.perf_counter()
        feat = sp.hstack([w * genre_onehot, overview_tfidf]).tocsr()
        sim = top10_from_matrix(feat)
        bt = time.perf_counter() - t0
        recs, q = run(f"v2 w={w}", sim, bt)
        v2_recs_by_w[w] = recs
        rows.append({"config": f"v2 w={w} (genre*{w} + overview char)", "build_s": round(bt, 3), "query_s": round(q, 4), **evaluate(recs)})
        _append_details(details, f"v2 w={w}", recs, movies, id_to_index)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "cbf_config_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(OUT_TAB / "cbf_v2_experiment_details.csv", index=False, encoding="utf-8-sig")

    # ---- selection ----
    v1_row = df[df["config"].str.startswith("v1")].iloc[0]
    sweep = df[df["config"].str.startswith("v2 w")]
    eps = 0.02
    candidates = sweep[
        (sweep["genre_jaccard"] >= v1_row["genre_jaccard"] - eps)
        & (sweep["content_sim"] >= v1_row["content_sim"] - eps)
        & (
            (sweep["genre_jaccard"] > v1_row["genre_jaccard"] + 0.01)
            | (sweep["content_sim"] > v1_row["content_sim"] + 0.01)
        )
    ]
    if len(candidates):
        chosen = candidates.loc[candidates["f1_balance"].idxmax()]
        rule = "max F1 among configs improving on v1 (no >0.02 regression)"
    else:
        chosen = sweep.loc[sweep["f1_balance"].idxmax()]
        rule = "max F1 among v2 sweep (no config improved v1 on both axes)"

    chosen_w = int(chosen["config"].split("w=")[1].split(" ")[0])

    # save selected v2 model artifact
    import joblib
    best_feat = sp.hstack([chosen_w * genre_onehot, overview_tfidf]).tocsr()
    artifact = config.DATA_PROCESSED / "cbf_model_v2.joblib"
    joblib.dump(
        {
            "genre_onehot": genre_onehot,
            "overview_vec": overview_vec,
            "overview_tfidf": overview_tfidf,
            "weight": chosen_w,
            "ids": movies["id"].tolist(),
            "nn": cbf_v2.TopKSimilarity(best_feat, movies["id"].tolist()).nn,
        },
        artifact,
    )

    # selected config top-10 table
    sel_rows = []
    for mid, lbl in SOURCE_MOVIES:
        for rank, rid in enumerate(v2_recs_by_w[chosen_w][mid], 1):
            sel_rows.append({
                "source_movie_id": mid, "source_label": lbl, "rank": rank,
                "movie_id": rid,
                "title": movies.iloc[id_to_index[rid]]["title"],
                "genres_text": movies.iloc[id_to_index[rid]]["genres_text"],
                "similarity": _similarity_from_ids(best_feat, id_to_index, mid, rid),
            })
    pd.DataFrame(sel_rows).to_csv(OUT_TAB / "cbf_v2_top10.csv", index=False, encoding="utf-8-sig")

    # figure: grouped bar (genre_jaccard vs content_sim) across configs
    labels = df["config"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    wbar = 0.38
    ax.bar(x - wbar / 2, df["genre_jaccard"], wbar, label="Genre Jaccard (relevance)", color="#3b6ea5")
    ax.bar(x + wbar / 2, df["content_sim"], wbar, label="Overview content similarity", color="#c0504d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score (0–1)")
    ax.set_title("CBF configurations — genre relevance vs overview content similarity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "cbf_config_comparison.png", dpi=130)
    plt.close(fig)

    # ---- report ----
    print(df.to_string(index=False))
    print("\nSELECTED:", chosen["config"], "|", rule)
    print(f"v1  : genre_jaccard={v1_row['genre_jaccard']:.4f} content_sim={v1_row['content_sim']:.4f}")
    print(f"best: genre_jaccard={chosen['genre_jaccard']:.4f} content_sim={chosen['content_sim']:.4f}")

    _write_report(df, v1_row, chosen, rule, chosen_w)
    print("wrote outputs/tables/cbf_config_comparison.csv, cbf_v2_top10.csv, cbf_v2_experiment_details.csv")
    print("wrote outputs/figures/cbf_config_comparison.png")
    print("wrote data/reports/cbf_v2_report.md, data/processed/cbf_model_v2.joblib")


def _append_details(details, cfg, recs, movies, id_to_index):
    for mid, lbl in SOURCE_MOVIES:
        for rank, rid in enumerate(recs[mid], 1):
            details.append({
                "config": cfg, "source_movie_id": mid, "source_label": lbl,
                "rank": rank, "movie_id": rid,
                "title": movies.iloc[id_to_index[rid]]["title"],
            })


def _similarity_from_ids(feature_matrix, id_to_index, src_id, rec_id):
    i, j = id_to_index[int(src_id)], id_to_index[int(rec_id)]
    from sklearn.metrics.pairwise import cosine_similarity
    return round(float(cosine_similarity(feature_matrix[i], feature_matrix[j])[0, 0]), 6)


def _df_to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_report(df, v1_row, chosen, rule, chosen_w):
    lines = [
        "# CBF v2 — Improvement Report",
        "",
        "## Methods tested",
        "1. **CBF v1 (baseline)** — combined `genres_text + overview`, whitespace TF-IDF, max_features=5000.",
        "2. **genre-only** — L2-normalized one-hot genre vector (19 genres), cosine.",
        "3. **overview-only** — character n-gram TF-IDF `analyzer='char', ngram_range=(3,5)`, max_features=5000, cosine.",
        "4. **v2 w** — separate features concatenated: `[w * genre_onehot | overview_char_tfidf]`, cosine; w ∈ {1,2,4,8,16}.",
        "",
        "## Results (real, computed on the same 6 source movies)",
        "",
        _df_to_md(df),
        "",
        "Metrics: `genre_jaccard` = mean Jaccard overlap of genre sets (source vs top-10); "
        "`content_sim` = mean cosine of overview char n-gram TF-IDF (source vs top-10); "
        "`f1_balance` = harmonic mean of the two; `lists_with_dup` / `lists_with_source` = count of "
        "recommendation lists (out of 6) containing a duplicate or the source itself (must be 0).",
        "",
        "## Selection",
        f"- Chosen: **{chosen['config']}** (w={chosen_w}).",
        f"- Rule: {rule}.",
        f"- v1  : genre_jaccard={v1_row['genre_jaccard']:.4f}, content_sim={v1_row['content_sim']:.4f}.",
        f"- v2  : genre_jaccard={chosen['genre_jaccard']:.4f}, content_sim={chosen['content_sim']:.4f}.",
        "",
        "> These are proxy quality signals, not a learned relevance label. Final recommendation "
        "relevance is validated against held-out ratings in the evaluation stage.",
        "",
        "## Limitations / remaining weaknesses",
        "- Genre one-hot + char n-grams are dependency-free; Thai word segmentation (e.g. pythainlp) "
        "was considered but not added to avoid an extra dependency.",
        "- Overview content similarity (cosine) is dominated by shared common char n-grams, so the "
        "absolute values are informative only in relative terms across configurations.",
        "- Full dense cosine matrix not materialized (kNN top-k is equivalent).",
        "",
        "## Outputs",
        "- `outputs/tables/cbf_config_comparison.csv` (aggregate comparison).",
        "- `outputs/tables/cbf_v2_experiment_details.csv` (all configs, all examples).",
        "- `outputs/tables/cbf_v2_top10.csv` (selected config examples).",
        "- `outputs/figures/cbf_config_comparison.png`.",
        "- `data/processed/cbf_model_v2.joblib` (selected model artifact).",
    ]
    (config.DATA_REPORTS / "cbf_v2_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
