"""Consistency checks (Stage 5.3). Writes consistency_check.md."""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import inference  # noqa: E402

OUT = ROOT / "docs" / "chapter4" / "05_webapp" / "consistency_check.md"
POP = ROOT / "data" / "processed" / "movie_popularity_train.csv"
SVD = ROOT / "data" / "processed" / "svd_model.joblib"
CBF = ROOT / "data" / "processed" / "cbf_model_v2.joblib"
GENRES_JSON = ROOT / "data" / "raw" / "genres.json"


def check(lines, name, cond, detail):
    lines.append(f"- [{'PASS' if cond else 'FAIL'}] {name} — {detail}")


def main():
    lines = [
        "# Consistency Check — Stage 5.3",
        "",
        f"Run: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    # 1. Popularity must match movie_popularity_train.csv exactly.
    pop = pd.read_csv(POP).sort_values(["train_mean_rating", "train_count"], ascending=False)
    expected = [(int(r.movie_id), float(r.train_mean_rating)) for r in pop.itertuples()]
    actual = inference.recommend_popular(top_k=len(pop))
    max_diff = max(abs(a - b) for (_, a), (_, b) in zip(actual, expected)) if actual else 0.0
    lines.append("## 1. Popularity (must match movie_popularity_train.csv exactly)")
    check(lines, "ids + order match", [m for m, _ in actual] == [m for m, _ in expected],
          f"{len(actual)} items")
    check(lines, "scores match", max_diff < 1e-12, f"max |diff| = {max_diff:.2e}")
    lines.append("")

    # 2. SVD item factors: load + shape.
    svd = joblib.load(SVD)
    qi = np.asarray(svd["algo"].qi, dtype=float)
    lines.append("## 2. SVD item factors (load + shape)")
    check(lines, "qi shape == (15, 100)", qi.shape == (15, 100), f"shape = {qi.shape}")
    lines.append("")

    # 3. CBF: independent recomputation vs webapp function.
    cbf = joblib.load(CBF)
    w = float(cbf["weight"])
    X = sp.hstack([w * cbf["genre_onehot"], cbf["overview_tfidf"]]).tocsr()
    Xn = normalize(X, norm="l2")
    ids = [int(i) for i in cbf["ids"]]
    id2idx = {i: k for k, i in enumerate(ids)}

    genres = json.load(open(GENRES_JSON, encoding="utf-8"))
    gsorted = sorted(g["id"] for g in genres)
    col = {g: i for i, g in enumerate(gsorted)}
    qg = np.zeros(len(gsorted)); qg[col[28]] = 1.0; qg[col[878]] = 1.0
    qg /= np.linalg.norm(qg)
    q = np.zeros(Xn.shape[1]); q[:len(gsorted)] = qg
    indep = np.asarray(Xn @ q).ravel()

    web_scores = inference._cbf_raw_scores([28, 878])
    diffs = [abs(web_scores.get(mid, 0.0) - float(indep[id2idx[mid]])) for mid in ids]
    md = max(diffs)
    lines.append("## 3. CBF — independent recomputation vs webapp function")
    check(lines, "scores match over full catalog", md < 1e-9,
          f"max |diff| = {md:.2e} over {len(ids)} movies")
    lines.append("")

    # 4. CF and Hybrid: documented as approximation (not compared to Stage 4).
    lines.append("## 4. CF and Hybrid — documented as approximation")
    lines.append("- `recommend_cf` = item-item cosine on SVD B item factors (new users have no fitted p_u).")
    lines.append("- Therefore it is NOT compared directly to Stage 4's Surprise `predict()` results.")
    lines.append("- `recommend_hybrid` blends that CF approximation with genre-only CBF; it approximates, "
                 "but does not exactly reproduce, Stage 4's Hybrid A.")
    lines.append("")

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
