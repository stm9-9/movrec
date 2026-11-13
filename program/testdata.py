"""
fetch_tmdb_movies.py (ฉบับปรับปรุง)
ดึงข้อมูลภาพยนตร์ 10,000 เรื่องจาก TMDb API ตามข้อกำหนดบทที่ 3

วิธีรัน:   python fetch_tmdb_movies.py
- ถ้าโปรแกรมหยุดกลางทาง (เน็ตหลุด ฯลฯ) ให้รันซ้ำ จะต่อจากหน้าที่ค้างไว้อัตโนมัติ
- ห้ามอัปโหลดไฟล์นี้ขึ้น GitHub ทั้งที่มี API Key อยู่ในไฟล์
  (แนะนำให้ตั้งเป็นตัวแปรสภาพแวดล้อม TMDB_API_KEY แทน)
"""
import json
import os
import time

import pandas as pd
import requests

# ==========================================
# 1. ตั้งค่า
# ==========================================
# อ่านจากตัวแปรสภาพแวดล้อมก่อน ถ้าไม่มีให้ใช้ค่าที่วางตรงนี้
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "4737b01b170e8f7db17bee70ce41ce30")
BASE_URL = "https://api.themoviedb.org/3"

RAW_CSV = "movies_raw.csv"
PARTIAL_CSV = "movies_raw_partial.csv"
PROGRESS_FILE = "fetch_progress.json"

SAVE_EVERY_PAGES = 25      # บันทึกความคืบหน้าทุกกี่หน้า
RATE_LIMIT_EVERY = 40      # ตามบท 3: พักทุก 40 คำขอ
RATE_LIMIT_SLEEP = 10      # พัก 10 วินาที
REQUEST_TIMEOUT = 15       # วินาที กันโปรแกรมค้างเมื่อเน็ตช้า
MAX_RETRIES = 4

# ภาพยนตร์จากแบบสอบถาม (ชื่อไว้ตรวจสอบว่า ID ตรงเรื่องจริง)
SURVEY_MOVIES = {
    299534: "Avengers: Endgame",
    597: "Titanic",
    138843: "The Conjuring",
    109445: "Frozen",
    168259: "Fast & Furious 7",
    296098: "Me Before You",
    157336: "Interstellar",
    18785: "The Hangover",
    671: "Harry Potter and the Philosopher's Stone",
    496243: "Parasite",
    372058: "Your Name",
    396535: "Train to Busan",
    354912: "Coco",
    18886: "Shutter",
    1083862: "Hunger",
}

SESSION = requests.Session()


# ==========================================
# 2. ฟังก์ชันเรียก API (มี retry และ timeout)
# ==========================================
def tmdb_get(path, params=None):
    """เรียก TMDb แล้วคืน JSON (หรือ None ถ้าล้มเหลว) ลองซ้ำเมื่อโดน rate limit/เน็ตหลุด"""
    p = {"api_key": TMDB_API_KEY}
    if params:
        p.update(params)
    url = f"{BASE_URL}{path}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = SESSION.get(url, params=p, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"[!] เครือข่ายขัดข้อง ({e.__class__.__name__}) ลองใหม่ใน {wait} วินาที")
            time.sleep(wait)
            continue

        if res.status_code == 200:
            return res.json()
        if res.status_code == 401:
            raise SystemExit("[X] API Key ไม่ถูกต้อง (401) กรุณาตรวจสอบ Key")
        if res.status_code == 429:  # ถี่เกินไป
            wait = int(res.headers.get("Retry-After", 2 ** attempt))
            print(f"[!] โดนจำกัดอัตราเรียก (429) รอ {wait} วินาที")
            time.sleep(wait)
            continue
        if res.status_code >= 500:  # เซิร์ฟเวอร์ TMDb มีปัญหา
            time.sleep(2 ** attempt)
            continue

        print(f"[!] {path} ได้ status {res.status_code}")
        return None

    print(f"[!] เรียก {path} ไม่สำเร็จหลังลอง {MAX_RETRIES} ครั้ง")
    return None


def fetch_genres():
    """ดึงตารางประเภทภาพยนตร์ทั้งหมดและบันทึกเป็น genres_tmdb.json"""
    data = tmdb_get("/genre/movie/list", {"language": "th-TH"})
    genres = (data or {}).get("genres", [])
    if genres:
        with open("genres_tmdb.json", "w", encoding="utf-8") as f:
            json.dump(genres, f, ensure_ascii=False, indent=2)
        print(f"[+] บันทึก Genres สำเร็จ ({len(genres)} หมวดหมู่)")
    else:
        print("[!] ไม่สามารถดึง Genres ได้")
    return genres


def fetch_movie_detail(movie_id):
    """ดึงรายละเอียดรายเรื่อง ภาษาไทยก่อน ถ้าเรื่องย่อว่างค่อยใช้ภาษาอังกฤษ"""
    data = tmdb_get(f"/movie/{movie_id}", {"language": "th-TH"}) or {}
    if not data.get("overview"):
        en = tmdb_get(f"/movie/{movie_id}", {"language": "en-US"}) or {}
        data["overview"] = en.get("overview") or ""
        if not data.get("title"):
            data["title"] = en.get("title") or ""
    return data


def fetch_overview_en(movie_id):
    """ใช้เมื่อรู้อยู่แล้วว่าเรื่องย่อภาษาไทยว่าง ไปดึงภาษาอังกฤษทันที (ประหยัดคำขอ 1 ครั้ง)"""
    return tmdb_get(f"/movie/{movie_id}", {"language": "en-US"}) or {}


# ==========================================
# 3. บันทึก/โหลดความคืบหน้า
# ==========================================
def save_progress(movies_list, last_page):
    pd.DataFrame(movies_list).to_csv(PARTIAL_CSV, index=False, encoding="utf-8-sig")
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_page": last_page}, f)


def load_progress():
    if os.path.exists(PROGRESS_FILE) and os.path.exists(PARTIAL_CSV):
        with open(PROGRESS_FILE) as f:
            last_page = json.load(f)["last_page"]
        df = pd.read_csv(PARTIAL_CSV, encoding="utf-8-sig", keep_default_na=False)
        print(f"[*] พบความคืบหน้าเดิม ต่อจากหน้า {last_page + 1} (มีข้อมูลแล้ว {len(df)} เรื่อง)")
        return last_page, df.to_dict("records")
    return 0, []


# ==========================================
# 4. ดึงภาพยนตร์ดิบ
# ==========================================
def fetch_raw_movies(target_count=10000):
    pages_to_fetch = target_count // 20  # 20 เรื่อง/หน้า
    last_page, movies_list = load_progress()
    seen_ids = {int(m["id"]) for m in movies_list}
    request_counter = 0

    print(f"[*] เริ่มดึงข้อมูลภาพยนตร์ {target_count} เรื่อง ({pages_to_fetch} หน้า)...")

    for page in range(last_page + 1, pages_to_fetch + 1):
        data = tmdb_get("/discover/movie", {
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "page": page,
            "language": "th-TH",
        })
        request_counter += 1

        if data is None:
            print(f"[!] ข้ามหน้า {page} (ดึงไม่สำเร็จ)")
        else:
            for m in data.get("results", []):
                mid = m.get("id")
                if mid is None or mid in seen_ids:
                    continue
                seen_ids.add(mid)

                title = (m.get("title") or "").strip()
                overview = (m.get("overview") or "").strip()

                # เรื่องย่อไทยว่าง -> ลองภาษาอังกฤษ
                if not overview:
                    en = fetch_overview_en(mid)
                    request_counter += 1
                    overview = (en.get("overview") or "").strip()
                    if not title:
                        title = (en.get("title") or "").strip()

                movies_list.append({
                    "id": mid,
                    "title": title,
                    "overview": overview,
                    "genre_ids": json.dumps(m.get("genre_ids", [])),
                    "release_date": m.get("release_date") or "",
                    "vote_average": m.get("vote_average", 0.0),
                    "popularity": m.get("popularity", 0.0),
                    "poster_path": m.get("poster_path") or "",
                })

        # หน่วงเวลาตามบท 3
        if request_counter >= RATE_LIMIT_EVERY:
            print(f"[*] ครบ {RATE_LIMIT_EVERY} คำขอ พัก {RATE_LIMIT_SLEEP} วินาที ตาม Rate Limit")
            time.sleep(RATE_LIMIT_SLEEP)
            request_counter = 0
        else:
            time.sleep(0.2)

        if page % SAVE_EVERY_PAGES == 0:
            save_progress(movies_list, page)
            print(f"-> ถึงหน้า {page}/{pages_to_fetch} (สะสม {len(movies_list)} เรื่อง) บันทึกแล้ว")

    # ภาพยนตร์จากแบบสอบถามให้ครบ
    print("[*] ตรวจสอบภาพยนตร์จากแบบสอบถาม...")
    for smid, expected in SURVEY_MOVIES.items():
        if smid in seen_ids:
            continue
        detail = fetch_movie_detail(smid)
        if detail.get("id"):
            seen_ids.add(smid)
            movies_list.append({
                "id": smid,
                "title": detail.get("title") or "",
                "overview": detail.get("overview") or "",
                "genre_ids": json.dumps([g["id"] for g in detail.get("genres", [])]),
                "release_date": detail.get("release_date") or "",
                "vote_average": detail.get("vote_average", 0.0),
                "popularity": detail.get("popularity", 0.0),
                "poster_path": detail.get("poster_path") or "",
            })
        else:
            print(f"[!] ไม่พบ ID {smid} ({expected}) ตรวจสอบ ID อีกครั้ง")

    df_raw = pd.DataFrame(movies_list)
    df_raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    # ลบไฟล์ความคืบหน้าเมื่อเสร็จ
    for f in (PARTIAL_CSV, PROGRESS_FILE):
        if os.path.exists(f):
            os.remove(f)

    print(f"\n[V] บันทึก {RAW_CSV} สำเร็จ จำนวน {len(df_raw)} แถว")
    report_quality(df_raw)
    verify_survey_ids(df_raw)


def report_quality(df):
    """สรุปค่าว่างที่พบ เก็บตัวเลขนี้ไว้ใช้เขียนตารางที่ 3-1 ในรายงาน"""
    empty_overview = (df["overview"].astype(str).str.strip() == "").sum()
    empty_genres = (df["genre_ids"].astype(str) == "[]").sum()
    empty_poster = (df["poster_path"].astype(str).str.strip() == "").sum()
    print("---- สรุปคุณภาพข้อมูลดิบ (เก็บไว้เขียนบท 4) ----")
    print(f"แถวทั้งหมด        : {len(df)}")
    print(f"เรื่องย่อว่าง      : {empty_overview}")
    print(f"ไม่มีประเภท       : {empty_genres}")
    print(f"ไม่มีโปสเตอร์      : {empty_poster}")
    print(f"id ซ้ำ            : {df['id'].duplicated().sum()}")


def verify_survey_ids(df):
    """พิมพ์ชื่อที่ TMDb ตอบกลับสำหรับแต่ละ ID ให้ตรวจด้วยตาว่าตรงเรื่องที่ต้องการ"""
    print("---- ตรวจ ID ภาพยนตร์ในแบบสอบถาม ----")
    for smid, expected in SURVEY_MOVIES.items():
        row = df[df["id"] == smid]
        got = row.iloc[0]["title"] if len(row) else "(ไม่พบ)"
        print(f"{smid:>8} | ต้องการ: {expected:<42} | TMDb: {got}")


if __name__ == "__main__":
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        print("[X] กรุณาใส่ TMDB_API_KEY ของคุณก่อนรัน")
    else:
        fetch_genres()
        fetch_raw_movies(target_count=10000)