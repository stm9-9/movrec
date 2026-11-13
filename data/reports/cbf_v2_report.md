# CBF v2 — Improvement Report

## Methods tested
1. **CBF v1 (baseline)** — combined `genres_text + overview`, whitespace TF-IDF, max_features=5000.
2. **genre-only** — L2-normalized one-hot genre vector (19 genres), cosine.
3. **overview-only** — character n-gram TF-IDF `analyzer='char', ngram_range=(3,5)`, max_features=5000, cosine.
4. **v2 w** — separate features concatenated: `[w * genre_onehot | overview_char_tfidf]`, cosine; w ∈ {1,2,4,8,16}.

## Results (real, computed on the same 6 source movies)

| config | build_s | query_s | genre_jaccard | content_sim | f1_balance | lists_with_dup | lists_with_source |
|---|---|---|---|---|---|---|---|
| v1 (combined word TF-IDF) | 0.984 | 0.0672 | 0.4104563492063492 | 0.23665837236514262 | 0.30021858039087 | 0 | 0 |
| genre-only (one-hot) | 0.003 | 0.0135 | 0.975 | 0.10444054452306076 | 0.18867093963934164 | 0 | 0 |
| overview-only (char 3-5 TF-IDF) | 0.013 | 0.474 | 0.28488095238095235 | 0.35745349815229094 | 0.31706751179542236 | 0 | 0 |
| v2 w=1 (genre*1 + overview char) | 0.047 | 0.6447 | 0.9180555555555555 | 0.2729287505807344 | 0.4207675188505911 | 0 | 0 |
| v2 w=2 (genre*2 + overview char) | 0.06 | 0.6944 | 0.975 | 0.21119825269368078 | 0.34719035525255376 | 0 | 0 |
| v2 w=4 (genre*4 + overview char) | 0.075 | 0.5496 | 0.975 | 0.21119825269368078 | 0.34719035525255376 | 0 | 0 |
| v2 w=8 (genre*8 + overview char) | 0.085 | 0.7314 | 0.975 | 0.21119825269368078 | 0.34719035525255376 | 0 | 0 |
| v2 w=16 (genre*16 + overview char) | 0.083 | 0.7081 | 0.975 | 0.21119825269368078 | 0.34719035525255376 | 0 | 0 |

Metrics: `genre_jaccard` = mean Jaccard overlap of genre sets (source vs top-10); `content_sim` = mean cosine of overview char n-gram TF-IDF (source vs top-10); `f1_balance` = harmonic mean of the two; `lists_with_dup` / `lists_with_source` = count of recommendation lists (out of 6) containing a duplicate or the source itself (must be 0).

## Selection
- Chosen: **v2 w=1 (genre*1 + overview char)** (w=1).
- Rule: max F1 among configs improving on v1 (no >0.02 regression).
- v1  : genre_jaccard=0.4105, content_sim=0.2367.
- v2  : genre_jaccard=0.9181, content_sim=0.2729.

> These are proxy quality signals, not a learned relevance label. Final recommendation relevance is validated against held-out ratings in the evaluation stage.

## Limitations / remaining weaknesses
- Genre one-hot + char n-grams are dependency-free; Thai word segmentation (e.g. pythainlp) was considered but not added to avoid an extra dependency.
- Overview content similarity (cosine) is dominated by shared common char n-grams, so the absolute values are informative only in relative terms across configurations.
- Full dense cosine matrix not materialized (kNN top-k is equivalent).

## Outputs
- `outputs/tables/cbf_config_comparison.csv` (aggregate comparison).
- `outputs/tables/cbf_v2_experiment_details.csv` (all configs, all examples).
- `outputs/tables/cbf_v2_top10.csv` (selected config examples).
- `outputs/figures/cbf_config_comparison.png`.
- `data/processed/cbf_model_v2.joblib` (selected model artifact).