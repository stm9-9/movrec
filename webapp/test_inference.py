"""Smoke test for webapp/inference.py (Stage 5.1).

Run from anywhere:
    python webapp/test_inference.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import inference  # noqa: E402

OUT = ROOT / "docs" / "chapter4" / "05_webapp" / "inference_smoke_test.txt"


def main():
    # Synthetic new user: rated 5 of the 15 survey movies; likes action + sci-fi.
    user_rated = {299534: 5.0, 157336: 5.0, 496243: 4.0, 372058: 4.0, 597: 3.0}
    preferred_genres = [28, 878]  # บู๊ (action), นิยายวิทยาศาสตร์ (sci-fi)

    gl = inference.genre_labels()
    lines = [
        "=== Inference smoke test (Stage 5.1) ===",
        "DESIGN NOTE:",
        "- CF: item-item cosine approximation on SVD B item factors (new users have no fitted p_u).",
        "- CBF: full CBF v2 feature matrix, genre-only query at cold start (overview block = 0).",
        "- Hybrid therefore approximates, but does not exactly reproduce, Stage 4's Hybrid A.",
        "",
        f"known_movies() count (SVD can score): {len(inference.known_movies())}",
        f"all_movies() count (full catalog): {len(inference.all_movies())}",
        "",
        f"Synthetic user ratings: {user_rated}",
        f"Preferred genres: {preferred_genres} "
        f"({[gl[g] for g in preferred_genres]})",
        "",
    ]

    def show(title, recs):
        lines.append(f"--- {title} (top {len(recs)}) ---")
        for rank, (mid, score) in enumerate(recs, 1):
            lines.append(f"  {rank}. {inference.id_to_title(mid)} (id {mid}) score={score:.4f}")
        lines.append("")

    show("recommend_cf (item-item via SVD B)", inference.recommend_cf(user_rated, top_k=5))
    show("recommend_cbf (genre cosine, CBF v2)", inference.recommend_cbf(preferred_genres, top_k=5))
    show("recommend_popular (train mean rating)", inference.recommend_popular(top_k=5))
    show("recommend_hybrid (alpha=0.3)", inference.recommend_hybrid(user_rated, preferred_genres, alpha=0.3, top_k=5))
    show("recommend_hybrid fallback (no signal)", inference.recommend_hybrid({}, [], top_k=3))

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
