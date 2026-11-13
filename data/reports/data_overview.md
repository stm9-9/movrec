# Data Overview

## movies.csv (TMDb metadata, raw)
- Rows: **9999**
- Columns: **8**

| column | dtype | non-empty | empty |
|---|---|---|---|
| id | int64 | 9999 | 0 |
| title | object | 9999 | 0 |
| overview | object | 9920 | 79 |
| genre_ids | object | 9999 | 0 |
| release_date | object | 9950 | 49 |
| vote_average | float64 | 9999 | 0 |
| popularity | float64 | 9999 | 0 |
| poster_path | object | 9938 | 61 |

- Duplicate `id`: 0
- Distinct genre ids: 19

## ratings.csv (Google Forms survey, raw)
- Rows: **180**
- Columns: **21**
- Structure: 6 demographic columns + 15 movie-rating columns
- Unique aliases: 180
- Duplicate aliases: 0

### Rating value distribution (non-empty)
| rating | count |
|---|---|
| 1.0 | 14 |
| 2.0 | 229 |
| 3.0 | 858 |
| 4.0 | 849 |
| 5.0 | 238 |

### Demographics (from the 180 respondents)
- gender: ชาย = 91, หญิง = 89
- year: ปี 4 = 45, ปี 2 = 45, ปี 3 = 45, ปี 1 = 45
- age: 21–22 ปี = 91, 18–20 ปี = 48, 23–24 ปี = 41

## genres.json
- Genres: **19**
- 28=บู๊, 12=ผจญ, 16=แอนนิเมชั่น, 35=ตลก, 80=อาชญากรรม, 99=สารคดี, 18=หนังชีวิต, 10751=ครอบครัว, 14=จินตนาการ, 36=ประวัติศาสตร์, 27=สยองขวัญ, 10402=ดนตรี, 9648=ลึกลับ, 10749=หนังรักโรแมนติก, 878=นิยายวิทยาศาสตร์, 10770=ภาพยนตร์โทรทัศน์, 53=ระทึกขวัญ, 10752=สงคราม, 37=หนังคาวบอยตะวันตก
