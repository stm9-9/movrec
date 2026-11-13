# Data Cleaning Report

## Movies
- Before: **9999** rows
- After listwise deletion + dedupe: **9841** rows
- Removed: **158** rows

### Detected problems
- Empty `title`: 0
- Empty `overview`: 79
- Empty `genre_ids` (`[]`): 65
- Empty `poster_path`: 61
- Empty `release_date`: 49
- `vote_average == 0.0` (kept, flagged): 353
- `vote_average == 10.0` (kept, flagged): 19
- Duplicate `id` (raw): 0

### Actions
1. Parsed `genre_ids` (JSON string) into a list; built `genres_text` (Thai names).
2. Listwise deletion: dropped rows with empty `overview` OR empty genres OR empty `poster_path`.
3. Dropped duplicate `id` (none found).

> Note: real counts are 9,999 → 9,841; the thesis text cites 10,000 → 5,500. Per approved decision, real numbers are reported and the discrepancy is documented for later thesis revision.

## Ratings
- Respondents (unique aliases): 180
- Duplicate aliases: 0
- Long-form ratings produced: **2188**
- Empty cells dropped: 512 (of 2700 possible)
- Rating range after validation: 1.0 – 5.0

### Actions
1. Mapped 15 movie columns (survey order) to TMDB ids.
2. Melted wide table to long form `(user_id, movie_id, rating)`.
3. Mapped alias -> integer `user_id`; kept demographics in `users.csv`.
4. Validated all ratings are within 1.0–5.0 (no violations).
