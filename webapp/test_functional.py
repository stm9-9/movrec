"""Functional tests for the webapp (Stage 5.3). Writes functional_test_log.md."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import inference  # noqa: E402

OUT = ROOT / "docs" / "chapter4" / "05_webapp" / "functional_test_log.md"

USER_RATED = {299534: 5.0, 157336: 5.0, 496243: 4.0, 372058: 4.0, 597: 3.0}
GENRES = [28, 878]  # บู๊, นิยายวิทยาศาสตร์
CSV_COLS = ["session", "method", "alpha", "beta", "movie_id", "title", "score", "rank"]


def _short(mid):
    t = inference.id_to_title(mid)
    return t if len(t) <= 26 else t[:26] + "…"


def _fmt(recs):
    return ", ".join(f"{_short(m)}={s:.3f}" for m, s in recs)


def check(lines, name, cond, detail):
    lines.append(f"- [{'PASS' if cond else 'FAIL'}] {name} — {detail}")


def main():
    lines = [
        "# Functional Test Log — Stage 5.3",
        "",
        f"Run: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    # 1. Cold start
    r_cold = inference.recommend_hybrid({}, [], alpha=0.3, top_k=10)
    r_pop = inference.recommend_popular(top_k=10)
    lines.append("## 1. Cold start (no ratings, no genres)")
    check(lines, "falls back to popularity", r_cold == r_pop,
          f"top-1 = {_short(r_cold[0][0])} ({r_cold[0][1]:.3f})")
    lines.append(f"  top-5: {_fmt(r_cold[:5])}")
    lines.append("")

    # 2. Genre-only
    r_genre = inference.recommend_hybrid({}, GENRES, alpha=0.3, top_k=10)
    r_cbf = inference.recommend_cbf(GENRES, top_k=10)
    lines.append("## 2. Genre-only (no ratings)")
    check(lines, "equals CBF", r_genre == r_cbf,
          f"top-1 = {_short(r_genre[0][0])} ({r_genre[0][1]:.3f})")
    lines.append(f"  top-5: {_fmt(r_genre[:5])}")
    lines.append("")

    # 3. Rating-only
    r_rating = inference.recommend_hybrid(USER_RATED, [], alpha=0.3, top_k=10)
    r_cf = inference.recommend_cf(USER_RATED, top_k=10)
    lines.append("## 3. Rating-only (no genres)")
    check(lines, "equals CF", r_rating == r_cf,
          f"top-1 = {_short(r_rating[0][0])} ({r_rating[0][1]:.3f})")
    lines.append(f"  top-5: {_fmt(r_rating[:5])}")
    lines.append("")

    # 4. Both
    r_both = inference.recommend_hybrid(USER_RATED, GENRES, alpha=0.3, top_k=10)
    lines.append("## 4. Both (hybrid, α=0.3)")
    lines.append(f"  top-10: {_fmt(r_both[:10])}")
    lines.append("")

    # 5. CSV export (mirrors webapp/app.py _to_csv)
    rows = [{
        "session": "FUNC-TEST",
        "method": "Hybrid A",
        "alpha": 0.3,
        "beta": 0.7,
        "movie_id": mid,
        "title": inference.id_to_title(mid),
        "score": round(float(s), 4),
        "rank": rank,
    } for rank, (mid, s) in enumerate(r_both, 1)]
    df = pd.DataFrame(rows)
    lines.append("## 5. CSV export")
    check(lines, "columns match spec", list(df.columns) == CSV_COLS, str(list(df.columns)))
    check(lines, "10 rows", len(df) == 10, f"{len(df)} rows")
    lines.append("")

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
