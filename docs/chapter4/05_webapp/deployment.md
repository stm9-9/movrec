# การติดตั้งใช้งาน (Deployment) — Stage 5.4

## Checklist การ deploy (เจ้าของดำเนินการ)

1. อัปโหลดโปรเจกต์ขึ้น **GitHub** (ต้องรวม `webapp/`, `recommender/`, `data/raw/`, `data/processed/`)
2. เข้า [Streamlit Community Cloud](https://share.streamlit.io) → **New app** → เลือก repository
3. ตั้ง **Main file path** = `webapp/app.py`
4. Streamlit ติดตั้งไลบรารีจาก `requirements.txt` โดยอัตโนมัติ
5. เปิดและบันทึก URL ที่ได้

## Live URL (เจ้าของกรอกหลัง deploy)

- [ ] URL: `https://movrecthesis.streamlit.app/` — **ยังไม่ deploy** (รอเจ้าของดำเนินการ)

## ข้อจำกัดที่ทราบ (Known limitations)

- **CF (การกรองร่วม):** ใช้ item-item cosine บน SVD item factors เป็นการประมาณสำหรับผู้ใช้ใหม่
  (ผู้ใช้ใหม่ไม่มี latent factor p_u) — ไม่ใช่ Surprise `predict()` เต็มรูปแบบ
- **CBF (การกรองตามเนื้อหา):** ที่ cold start ใช้เฉพาะ genre (overview block = 0) เพราะยังไม่มี
  ข้อมูลเนื้อหาอื่นให้แมตช์
- **Hybrid:** รวม CF approximation + genre-only CBF → ประมาณผล แต่ไม่ซ้ำกับ Stage 4 Hybrid A แบบเป๊ะ
- **Normalization:** min-max ต่อผู้ใช้ บน candidate set ขนาดเล็ก (3–9 เรื่อง) จึงหยาบ
- **ขนาดโมเดล:** `cbf_model_v2.joblib` ≈ 66 MB — โหลดครั้งเดียวแล้ว cache (lru_cache)

## ผล UAT

- ผลการทดสอบการยอมรับของผู้ใช้ (UAT) อยู่ระหว่างรอเก็บข้อมูลจากเจ้าของ
- ไฟล์ `uat_results.csv` ยังเป็น**เทมเพลตว่าง** (มีเฉพาะ header) — ห้ามกรอกผลปลอม
