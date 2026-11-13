"""Streamlit UI for the movie recommender (Stage 5.2b — redesigned).

Run from the project root:
    streamlit run webapp/app.py
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import inference  # noqa: E402

MOVIES_CLEAN = ROOT / "data" / "processed" / "movies_clean.csv"
POSTER_BASE = "https://image.tmdb.org/t/p/w200"
METHODS = ["Hybrid A", "CF only", "CBF only", "Popularity"]

st.set_page_config(page_title="ระบบแนะนำภาพยนตร์", page_icon="🎬", layout="wide")

# --------------------------------------------------------------------------
# Custom CSS — "HBO Max × Linear" dark minimal
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif;
    color: #e5e7eb;
}

/* Hero */
.hero {
    background: linear-gradient(180deg, #0a0e1a 0%, #121826 100%);
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 3rem 2.25rem 2.5rem 2.25rem;
    margin-bottom: 2.25rem;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3.4rem;
    font-weight: 700;
    color: #e5e7eb;
    line-height: 1.1;
    letter-spacing: 0.01em;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    color: #9ca3af;
    font-size: 1.05rem;
    max-width: 46rem;
    margin-bottom: 1.5rem;
}
.btn-primary {
    display: inline-block;
    background: #4f8cff;
    color: #0a0e1a;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    border-radius: 8px;
    text-decoration: none;
    font-size: 0.95rem;
}
.btn-primary:hover { background: #6ba3ff; color: #0a0e1a; }

/* Section label */
.section-label {
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 2rem 0 0.75rem 0;
}

/* Card containers (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #121826;
    border: 1px solid #1f2937 !important;
    border-radius: 12px;
    padding: 12px;
    transition: border-color 0.15s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #4f8cff !important; }

/* Movie card bits */
.poster {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border-radius: 8px;
    display: block;
}
.card-title {
    font-weight: 600;
    color: #e5e7eb;
    font-size: 0.9rem;
    line-height: 1.25;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.35em;
    margin-top: 0.4rem;
}
.card-meta { color: #9ca3af; font-size: 0.76rem; line-height: 1.3; margin-top: 0.25rem; }
.card-reason { color: #9ca3af; font-style: italic; font-size: 0.78rem; margin-top: 0.2rem; }
.score-badge {
    display: inline-block;
    background: #4f8cff;
    color: #0a0e1a;
    font-weight: 700;
    border-radius: 8px;
    padding: 2px 8px;
    font-size: 0.76rem;
}

/* Primary buttons (Streamlit) */
.stButton > button {
    background: #4f8cff;
    color: #0a0e1a;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover { background: #6ba3ff; color: #0a0e1a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0e1a;
    border-right: 1px solid #1f2937;
}

/* Hide default header/footer chrome */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_movie_map():
    m = pd.read_csv(MOVIES_CLEAN, keep_default_na=False)
    return {
        int(r.id): {"title": r.title, "poster": r.poster_path, "genres": r.genres_text}
        for r in m.itertuples()
    }


@st.cache_data
def genre_options():
    """Sorted list of (genre_id, Thai name)."""
    gl = inference.genre_labels()
    return sorted(((gid, name) for gid, name in gl.items()), key=lambda x: x[1])


def _genre_name(genre_id):
    return inference.genre_labels().get(int(genre_id), "?")


def _current_ratings(popular_ids):
    """Read ratings from the star-widget session keys (0 = skip)."""
    return {
        int(mid): int(st.session_state.get(f"rating_{mid}", 0))
        for mid in popular_ids
        if int(st.session_state.get(f"rating_{mid}", 0)) > 0
    }


def _reason(genres, ratings):
    if genres:
        return f"เพราะคุณชอบ {_genre_name(genres[0])}"
    if ratings:
        return "คล้ายกับภาพยนตร์ที่คุณให้คะแนน"
    return "ภาพยนตร์ยอดนิยม"


def compute_recs(method, alpha, genres, ratings):
    """Return (recs, reason, message)."""
    if not genres and not ratings:
        return inference.recommend_popular(top_k=10), _reason(genres, ratings), \
            "ยังไม่เลือกประเภทหรือให้คะแนน — แสดงภาพยนตร์ยอดนิยม"

    if method == "Hybrid A":
        return inference.recommend_hybrid(ratings, genres, alpha=alpha, top_k=10), \
            _reason(genres, ratings), ""
    if method == "CF only":
        if not ratings:
            return inference.recommend_popular(top_k=10), "ภาพยนตร์ยอดนิยม", \
                "การกรองร่วมต้องใช้คะแนน — แสดงภาพยนตร์ยอดนิยมแทน"
        return inference.recommend_cf(ratings, top_k=10), _reason(genres, ratings), ""
    if method == "CBF only":
        if not genres:
            return inference.recommend_popular(top_k=10), "ภาพยนตร์ยอดนิยม", \
                "การกรองตามเนื้อหาต้องเลือกประเภท — แสดงภาพยนตร์ยอดนิยมแทน"
        return inference.recommend_cbf(genres, top_k=10), _reason(genres, ratings), ""
    # Popularity
    return inference.recommend_popular(top_k=10), "ภาพยนตร์ยอดนิยม", ""


def _to_csv(recs, movie_map, method, alpha, beta):
    session = st.session_state.get("session_id", "")
    rows = []
    for rank, (mid, score) in enumerate(recs, 1):
        info = movie_map.get(mid, {})
        rows.append({
            "session": session,
            "method": method,
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "movie_id": mid,
            "title": info.get("title", ""),
            "score": round(float(score), 4),
            "rank": rank,
        })
    return pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">ค้นพบหนังที่ใช่</div>
            <div class="hero-subtitle">ระบบแนะนำภาพยนตร์ที่ผสานการกรองตามเนื้อหา
            การกรองร่วม และแบบผสม — เลือกประเภทที่ชอบ ให้คะแนน แล้วรับคำแนะนำเฉพาะตัว</div>
            <a class="btn-primary" href="#genre">เริ่มต้น</a>
        </div>
        <a name="genre"></a>
        """,
        unsafe_allow_html=True,
    )


def render_genre_section():
    st.markdown('<div class="section-label">ขั้นที่ 1 — เลือกประเภทที่ชอบ</div>',
                unsafe_allow_html=True)
    opts = genre_options()
    ids = [gid for gid, _ in opts]
    names = {gid: name for gid, name in opts}
    if hasattr(st, "pills"):
        st.pills(
            "เลือกได้หลายประเภท",
            options=ids,
            selection_mode="multi",
            format_func=lambda gid: names[gid],
            key="genre_ids",
            label_visibility="collapsed",
        )
    else:
        st.multiselect(
            "เลือกได้หลายประเภท",
            options=ids,
            format_func=lambda gid: names[gid],
            key="genre_ids",
            label_visibility="collapsed",
        )


def _card_poster(poster_path):
    if not poster_path:
        return ""
    return f'<img class="poster" src="{POSTER_BASE}{poster_path}" />'


def render_rating_section(popular, movie_map):
    st.markdown('<div class="section-label">ขั้นที่ 2 — ให้คะแนนภาพยนตร์</div>',
                unsafe_allow_html=True)
    st.caption("ให้คะแนนภาพยนตร์ยอดนิยม 1–5 ดาว (ข้ามได้ถ้าไม่เคยดู)")

    for i in range(0, len(popular), 5):
        cols = st.columns(5)
        for j in range(5):
            idx = i + j
            if idx >= len(popular):
                break
            mid, _ = popular[idx]
            info = movie_map.get(mid, {})
            with cols[j]:
                with st.container(border=True):
                    if info.get("poster"):
                        st.markdown(_card_poster(info["poster"]), unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="card-title">{info.get("title", "?")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.select_slider(
                        "คะแนน",
                        options=[0, 1, 2, 3, 4, 5],
                        value=st.session_state.get(f"rating_{mid}", 0),
                        key=f"rating_{mid}",
                        format_func=lambda x: "ข้าม" if x == 0 else "★" * x,
                        label_visibility="collapsed",
                    )


def render_results(recs, reason, method, alpha, beta, movie_map, message):
    st.markdown('<div class="section-label">ขั้นที่ 3 — ผลแนะนำสำหรับคุณ</div>',
                unsafe_allow_html=True)
    chip = f'<span class="score-badge">{method}</span>'
    if method == "Hybrid A":
        chip = f'<span class="score-badge">Hybrid A · α={alpha:.2f} β={beta:.2f}</span>'
    st.markdown(chip, unsafe_allow_html=True)

    if message:
        st.info(message)

    if not recs:
        st.error("ไม่พบภาพยนตร์ในคลังข้อมูล (คลังว่าง)")
        return

    for i in range(0, len(recs), 5):
        cols = st.columns(5)
        for j in range(5):
            idx = i + j
            if idx >= len(recs):
                break
            rank, (mid, score) = idx + 1, recs[idx]
            info = movie_map.get(mid, {})
            with cols[j]:
                with st.container(border=True):
                    if info.get("poster"):
                        st.markdown(_card_poster(info["poster"]), unsafe_allow_html=True)
                    st.markdown(
                        f'<span class="score-badge">#{rank} · {float(score):.2f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="card-title">{info.get("title", "?")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="card-meta">{info.get("genres", "")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="card-reason">{reason}</div>',
                        unsafe_allow_html=True,
                    )

    if st.button("เริ่มใหม่"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


def main():
    st.session_state.setdefault("genre_ids", [])
    st.session_state.setdefault("alpha", 0.3)
    st.session_state.setdefault("session_id", datetime.now().strftime("%Y%m%d-%H%M%S"))

    movie_map = load_movie_map()
    popular = inference.recommend_popular(top_k=10)
    popular_ids = [m for m, _ in popular]

    genre_ids = list(st.session_state.get("genre_ids", []))
    ratings = _current_ratings(popular_ids)
    alpha = float(st.session_state.get("alpha", 0.3))

    # ---- Sidebar: Model Control Panel (always visible) ----
    with st.sidebar:
        st.markdown("## แผงควบคุมโมเดล")
        method = st.radio("วิธีแนะนำ", METHODS, index=0)
        if method == "Hybrid A":
            alpha = st.slider("α (น้ำหนัก CF)", 0.0, 1.0, alpha, 0.05, key="alpha")
            st.caption(f"α = {alpha:.2f} · β = {1 - alpha:.2f}")
        else:
            st.caption("น้ำหนักใช้เฉพาะ Hybrid A")
        beta = 1.0 - alpha

        recs, reason, _ = compute_recs(method, alpha, genre_ids, ratings)

        st.markdown("### ตัวอย่างผล (Top 5)")
        if not recs:
            st.caption("ยังไม่มีผลลัพธ์")
        for rank, (mid, score) in enumerate(recs[:5], 1):
            t = movie_map.get(mid, {}).get("title", "?")
            st.markdown(f"**{rank}.** {t} · `{score:.2f}`")

        st.markdown("---")
        csv = _to_csv(recs, movie_map, method, alpha, beta)
        st.download_button("บันทึกผลเป็น CSV", data=csv,
                           file_name="recommendations.csv", mime="text/csv")

    # ---- Main area (single scroll) ----
    render_hero()
    render_genre_section()
    render_rating_section(popular, movie_map)

    # Re-read state after widgets rendered (values are identical; kept for clarity).
    genre_ids = list(st.session_state.get("genre_ids", []))
    ratings = _current_ratings(popular_ids)
    recs, reason, message = compute_recs(method, alpha, genre_ids, ratings)
    render_results(recs, reason, method, alpha, beta, movie_map, message)


if __name__ == "__main__":
    main()
