# Consistency Check — Stage 5.3

Run: 2026-11-13T08:40:20

## 1. Popularity (must match movie_popularity_train.csv exactly)
- [PASS] ids + order match — 15 items
- [PASS] scores match — max |diff| = 0.00e+00

## 2. SVD item factors (load + shape)
- [PASS] qi shape == (15, 100) — shape = (15, 100)

## 3. CBF — independent recomputation vs webapp function
- [PASS] scores match over full catalog — max |diff| = 0.00e+00 over 9841 movies

## 4. CF and Hybrid — documented as approximation
- `recommend_cf` = item-item cosine on SVD B item factors (new users have no fitted p_u).
- Therefore it is NOT compared directly to Stage 4's Surprise `predict()` results.
- `recommend_hybrid` blends that CF approximation with genre-only CBF; it approximates, but does not exactly reproduce, Stage 4's Hybrid A.
