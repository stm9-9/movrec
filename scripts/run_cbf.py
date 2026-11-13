"""Stage 2 demo + evidence: fit CBF, test similar-movie recommendations, save outputs.

Run from anywhere:
    python scripts/run_cbf.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from recommender.content_based.tfidf_model import ContentBasedModel  # noqa: E402
from recommender.preprocessing import config  # noqa: E402

# Thai-capable fonts (matplotlib default has no Thai glyphs).
plt.rcParams["font.family"] = ["Tahoma", "Microsoft Sans Serif", "Leelawadee UI"]
plt.rcParams["axes.unicode_minus"] = False

OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
EVID = ROOT / "outputs" / "screenshots" / "chapter4" / "02_content_based"
MODEL_ARTIFACT = config.DATA_PROCESSED / "cbf_model.joblib"

# At least 5 well-known source movies (survey movies, guaranteed real data).
SOURCE_MOVIES = [
    (299534, "Avengers: Endgame"),
    (597, "Titanic"),
    (496243, "Parasite"),
    (372058, "Your Name"),
    (157336, "Interstellar"),
    (138843, "The Conjuring"),
]

TOP_N = 10


def make_figure(model, examples, path):
    """Bar chart of top-10 cosine similarity for two example movies."""
    two = examples[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ex in zip(axes, two):
        titles = [r["title"][:24] for r in ex["results"]]
        scores = [r["similarity"] for r in ex["results"]]
        ax.barh(range(len(titles)), scores, color="#3b6ea5")
        ax.set_yticks(range(len(titles)))
        ax.set_yticklabels(titles, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Cosine similarity")
        ax.set_title(f"Top-10 similar to: {ex['label']} (id {ex['movie_id']})", fontsize=10)
        ax.set_xlim(0, 1)
    fig.suptitle("Content-Based Filtering — Cosine Similarity (TF-IDF: genres + overview)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    for d in (OUT_FIG, OUT_TAB, EVID):
        d.mkdir(parents=True, exist_ok=True)

    movies = pd.read_csv(config.MOVIES_CLEAN, keep_default_na=False)
    model = ContentBasedModel(movies, max_features=5000)

    rows, cols, pn, pn2 = model.matrix_dimensions()

    log = []
    log.append("=== CBF MODEL SUMMARY ===")
    log.append(f"Input movies: {model.n_movies} (dropped {model.n_dropped_empty} empty-feature rows)")
    log.append(f"TF-IDF max_features: {model.max_features}")
    log.append(f"TF-IDF matrix: {rows} x {cols} (sparse)")
    log.append(f"Pairwise cosine matrix dimension: {pn} x {pn2} (computed lazily via kNN)")
    log.append(f"NearestNeighbors: metric='cosine', k=11 (top-10 after excluding self)")
    log.append("")

    examples = []
    all_rows = []
    for movie_id, label in SOURCE_MOVIES:
        try:
            results = model.similar_movies(movie_id, top_n=TOP_N)
        except KeyError as e:
            log.append(f"[skip] {e}")
            continue
        examples.append({"movie_id": movie_id, "label": label, "results": results})
        log.append(f"--- {label} (id {movie_id}) ---")
        for rank, r in enumerate(results, 1):
            log.append(f"  {rank:>2}. {r['title'][:50]:<50}  sim={r['similarity']:.4f}  [{r['genres_text']}]")
            all_rows.append({
                "source_movie_id": movie_id,
                "source_label": label,
                "rank": rank,
                "movie_id": r["movie_id"],
                "title": r["title"],
                "genres_text": r["genres_text"],
                "similarity": r["similarity"],
            })
        log.append("")

    # Save results table
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(OUT_TAB / "cbf_similar_movies.csv", index=False, encoding="utf-8-sig")

    # Persist model for later reuse
    model.save(MODEL_ARTIFACT)

    # Figure
    fig_path = OUT_FIG / "cbf_top10_examples.png"
    make_figure(model, examples, fig_path)

    # Evidence files under screenshots/chapter4/02_content_based/
    model_summary = "\n".join(log[:6]) + "\n"
    (EVID / "model_summary.txt").write_text(model_summary, encoding="utf-8")

    ex_lines = ["# CBF recommendation examples (real results)\n"]
    for ex in examples:
        ex_lines.append(f"\n## {ex['label']} (id {ex['movie_id']})\n")
        ex_lines.append("| rank | movie | similarity | genres |")
        ex_lines.append("|---|---|---|---|")
        for rank, r in enumerate(ex["results"], 1):
            ex_lines.append(f"| {rank} | {r['title']} | {r['similarity']:.4f} | {r['genres_text']} |")
    (EVID / "cbf_examples.md").write_text("\n".join(ex_lines), encoding="utf-8")

    full_log = "\n".join(log)
    (EVID / "execution_output.txt").write_text(full_log + "\n", encoding="utf-8")

    # Copy the figure into the evidence dir for screenshot convenience
    import shutil
    shutil.copyfile(fig_path, EVID / fig_path.name)

    # Implementation + test report
    report_lines = [
        "# Content-Based Filtering — Implementation & Test Report",
        "",
        "## Implementation",
        "- Module: `recommender/content_based/tfidf_model.py` (`ContentBasedModel`).",
        "- Feature text: `genres_text` (Thai genre names) + `overview` (Thai/English).",
        "- Vectorization: `TfidfVectorizer(max_features=5000)`.",
        "- Similarity: cosine via `NearestNeighbors(metric='cosine', k=11)`; source movie excluded.",
        "",
        "## Model summary (actual)",
        f"- Movies used: **{model.n_movies}**",
        f"- Empty-feature rows dropped: **{model.n_dropped_empty}**",
        f"- TF-IDF matrix: **{rows} x {cols}**",
        f"- Pairwise cosine matrix dimension: **{pn} x {pn2}** (lazy, top-k only)",
        "",
        "## Test results (top-10 similar per source movie)",
    ]
    for ex in examples:
        report_lines.append(f"\n### {ex['label']} (id {ex['movie_id']})")
        report_lines.append("| rank | movie | similarity | genres |")
        report_lines.append("|---|---|---|---|")
        for rank, r in enumerate(ex["results"], 1):
            report_lines.append(f"| {rank} | {r['title']} | {r['similarity']:.4f} | {r['genres_text']} |")
    report_lines += [
        "",
        "## Outputs",
        f"- Results table: `outputs/tables/cbf_similar_movies.csv`",
        f"- Figure: `outputs/figures/cbf_top10_examples.png`",
        f"- Model artifact: `data/processed/cbf_model.joblib`",
        f"- Evidence: `outputs/screenshots/chapter4/02_content_based/`",
        "",
        "## Limitations",
        "- Thai text is tokenized by whitespace (standard `TfidfVectorizer`); genre names are "
        "single space-separated tokens, so the genre signal dominates. A Thai word-segmenter or "
        "char n-grams could improve overview matching but was not added to keep dependencies minimal.",
        "- The full dense 9841x9841 cosine matrix is not materialized; item-item similarity is "
        "retrieved on demand via a brute-force kNN index (equivalent result, far less memory).",
    ]
    report_path = config.DATA_REPORTS / "cbf_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Console output
    print(full_log)
    print(f"results table: outputs/tables/cbf_similar_movies.csv")
    print(f"figure:        {fig_path.relative_to(ROOT)}")
    print(f"model:         data/processed/cbf_model.joblib")
    print(f"report:        data/reports/cbf_report.md")
    print(f"evidence:      outputs/screenshots/chapter4/02_content_based/")


if __name__ == "__main__":
    main()
