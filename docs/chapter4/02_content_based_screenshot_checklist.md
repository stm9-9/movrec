# Chapter 4 — Content-Based Filtering Screenshot Checklist

Stage 2 evidence. All items are real outputs produced by `python scripts/run_cbf.py`.

Evidence directory: `outputs/screenshots/chapter4/02_content_based/`

## Items to screenshot

- [ ] **1. TF-IDF construction + feature/matrix dimensions**
      File: `outputs/screenshots/chapter4/02_content_based/model_summary.txt`
      Shows: movies used (9,841), TF-IDF `max_features=5000`, TF-IDF matrix `9841 x 5000`,
      pairwise cosine matrix dimension `9841 x 9841`, kNN settings.

- [ ] **2. Successful execution output**
      File: `outputs/screenshots/chapter4/02_content_based/execution_output.txt`
      Shows the full console run (summary + all example recommendations).

- [ ] **3. Cosine similarity results (recommendation examples)**
      File: `outputs/screenshots/chapter4/02_content_based/cbf_examples.md`
      Shows top-10 similar movies + cosine similarity + genres for 6 source movies.

- [ ] **4. Results table (raw data)**
      File: `outputs/tables/cbf_similar_movies.csv`
      All example rows in one table (source movie, rank, movie, similarity, genres).

- [ ] **5. Figure — cosine similarity bar chart**
      File: `outputs/figures/cbf_top10_examples.png`
      Top-10 cosine similarity for two example movies (Avengers: Endgame, Titanic).

## Example source movies (real data)
Avengers: Endgame (299534), Titanic (597), Parasite (496243), Your Name (372058),
Interstellar (157336), The Conjuring (138843).

## Notes
- The source movie is excluded from its own recommendations (verified by construction).
- Similarity scores are low (~0.15–0.30) because cosine similarity is computed against
  a 9,841-movie corpus; recommendations are still genre-coherent (see examples).

---

# Stage 2.1 — Improved CBF (v2) screenshot items

Selected config: **v2 w=1** (`genre·1` + overview character n-gram TF-IDF).

- [ ] **1. Configuration comparison table**
      File: `outputs/tables/cbf_config_comparison.csv` (also rendered in `data/reports/cbf_v2_report.md`)
      Shows genre Jaccard, overview content similarity, F1 balance, build/query time per config
      (v1, genre-only, overview-only, v2 w=1..16). v2 w=1 improves genre Jaccard 0.41→0.92 and
      content similarity 0.24→0.27 vs v1.

- [ ] **2. Comparison figure**
      File: `outputs/figures/cbf_config_comparison.png`
      Grouped bar chart of genre relevance vs overview content similarity across configurations.

- [ ] **3. Selected v2 recommendation examples**
      File: `outputs/screenshots/chapter4/02_content_based/cbf_v2_examples.md`
      Top-10 recommendations for the same 6 source movies (Avengers: Endgame, Titanic, Parasite,
      Your Name, Interstellar, The Conjuring).

- [ ] **4. All-configuration detail table**
      File: `outputs/tables/cbf_v2_experiment_details.csv`
      Every (config × source × rank) recommendation for transparency.

- [ ] **5. Improvement report**
      File: `data/reports/cbf_v2_report.md`
      Methods, results, selection justification, and remaining limitations.
