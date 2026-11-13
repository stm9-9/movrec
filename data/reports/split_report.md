# Train/Test Split Report

Method: per-user 80/20 holdout (fixed `random_state=42`), computed BEFORE model fitting.

| metric | value |
|---|---|
| total_ratings | 2188 |
| n_train | 1749 |
| n_test | 439 |
| train_frac | 0.7994 |
| n_users | 180 |
| n_users_with_test | 180 |
| n_users_excluded | 0 |

## Ratings per user (distribution)
| ratings/user | users |
|---|---|
| 8 | 2 |
| 9 | 9 |
| 10 | 14 |
| 11 | 27 |
| 12 | 49 |
| 13 | 48 |
| 14 | 23 |
| 15 | 8 |

## Train rating count per movie (from training data only)
| movie_id | train_mean_rating | train_count |
|---|---|---|
| 597 | 3.426 | 108 |
| 671 | 3.667 | 147 |
| 18785 | 3.351 | 148 |
| 18886 | 3.056 | 72 |
| 109445 | 3.255 | 137 |
| 138843 | 3.225 | 138 |
| 157336 | 3.796 | 142 |
| 168259 | 3.574 | 148 |
| 296098 | 3.531 | 143 |
| 299534 | 3.857 | 56 |
| 354912 | 3.412 | 68 |
| 372058 | 3.515 | 103 |
| 396535 | 3.641 | 78 |
| 496243 | 3.600 | 125 |
| 1083862 | 3.169 | 136 |

## Limitations
- Users excluded from top-K evaluation (fewer than 2 ratings): 0
- Only 15 movies exist in the rating matrix; SVD latent factors up to 100 are over-parameterized relative to 15 items (flagged for the CF stage).
- Users with no rating history are cold-start users; they use the popularity fallback (`movie_popularity_train.csv`), never a crash path.
