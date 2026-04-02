# Phase 1 TF-IDF Recommender Documentation

This document explains the TF-IDF recommender implementation, the exact ranking logic used in code, and why it performed better than alternatives in the current offline evaluation setup.

## 1. What the TF-IDF Recommender Does

Main implementation: `recommender.py`

High-level flow:

1. Load and clean metadata from `NetflixDataset.csv`.
2. Build engineered content/tone/depth features per title.
3. Build a weighted text "soup" per title.
4. Vectorize with TF-IDF (unigrams + bigrams).
5. Compute cosine similarity between all titles.
6. Build candidate pool from seed-title neighbors.
7. Apply business-aware scoring (depth, uniqueness, tone, penalties).
8. Re-rank with MMR for precision/diversity balance.
9. Return top-K recommendations with title and image.

## 2. Implementation Details (Code-Level)

### 2.1 Data Cleaning and Canonicalization

The recommender normalizes and deduplicates title rows before modeling:

- `normalize_text(...)` lowercases and trims text values.
- `parse_tokens(...)` converts comma-separated fields into token lists.
- Duplicate titles are removed (`raw_df[~raw_df['Title'].duplicated()]`).
- Numeric fields are parsed robustly with fallback defaults:
  - `IMDb Score` -> default `6.6`
  - `Awards Received`, `Awards Nominated For`, `IMDb Votes` -> default `0`

### 2.2 Feature Engineering

The model creates quality and intent signals before ranking:

- `depth_score`:
  - Uses normalized IMDb score, awards, and summary length.
  - Formula in code:
    - `depth_score = 0.50 * imdb_norm + 0.30 * awards_norm + 0.20 * summary_norm`

- `uniqueness_score`:
  - Uses rarity of genres/tags plus anti-popularity pressure.
  - Formula in code:
    - `uniqueness_score = 0.65 * rarity_norm + 0.35 * (1 - votes_norm)`

- Tone signals:
  - Keyword banks produce `intellectual_signal_raw`, `action_signal_raw`, `emotional_signal_raw`.
  - Then normalized into:
    - `intellectual_tone_score = 0.7 * intellectual_norm + 0.3 * emotional_norm`
    - `action_tone_score = action_norm`

### 2.3 Weighted Text Soup + TF-IDF

To emphasize the right metadata fields, the implementation repeats fields in the text soup:

- Genre x3
- Tags x3
- Summary x4
- Actors x2
- Director x2
- Writer x2
- ViewerRating x1

Then it runs:

- `TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.90)`
- `tfidf_matrix = vectorizer.fit_transform(model_data['soup'])`
- `cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)`

Why these defaults are important:

- `ngram_range=(1,2)` captures both key words and short phrases.
- `min_df=2` removes one-off noise tokens.
- `max_df=0.90` suppresses ultra-common low-information terms.

### 2.4 Candidate Retrieval and Intent-Aware Filtering

For each seed title, the recommender collects top cosine neighbors (`_get_similar_items`).

It then applies filtering logic:

- Seed titles are excluded from output.
- Language overlap with user-selected languages is enforced.
- Intellectual-mode guardrail removes low-depth action-skewed items when user intent is intellectual.

### 2.5 Final Ranking Formula

Each candidate gets a `base_score` using similarity + quality + intent alignment:

- `combined_similarity = 0.62 * sim_max + 0.38 * sim_avg`
- `mainstream_penalty = max(0, votes_norm - 0.78) * (1 - uniqueness_score) + 1.00 * action_tone_score * (1 - depth_score)`
- `precision_bonus = 0.09 * intellectual_tone_score * intent_intellectual + 0.07 * depth_score * intent_depth`

Final:

- `base_score =`
  - `0.55 * combined_similarity`
  - `+ 0.21 * depth_score`
  - `+ 0.07 * uniqueness_score`
  - `+ 0.13 * intellectual_tone_score`
  - `+ 0.09 * (1 - action_tone_score)`
  - `+ precision_bonus`
  - `- 0.16 * mainstream_penalty`

This is one of the key reasons the TF-IDF approach works well here: retrieval is text-driven, but ranking is not only similarity-driven.

### 2.6 MMR Re-Ranking

After scoring, top candidates are re-ranked with Maximum Marginal Relevance (`_mmr_rerank`):

- Objective:
  - `mmr_score = lambda * relevance - (1 - lambda) * redundancy`
- Intent-aware lambda defaults:
  - More intellectual intent -> higher lambda (precision-heavy)
  - More action intent -> slightly lower lambda (more diversity)

This prevents near-duplicates while preserving strong relevance.

## 3. Why TF-IDF Was Superior (Current Evidence)

Evaluation scripts:

- `evaluate_recommender.py` (TF-IDF quality)
- `evaluate_semantic_transition.py` (TF-IDF vs semantic)
- `evaluate_phase2_transition.py` (TF-IDF vs semantic vs phase2)

### 3.1 Current Offline Results (Apr 2, 2026)

From `evaluate_semantic_transition.py`:

| Profile | TF-IDF Precision | Semantic Precision | Delta (Semantic - TF-IDF) |
|---|---:|---:|---:|
| Intellectual Sci-Fi | 0.642 | 0.603 | -0.039 |
| Crime Drama Depth | 0.677 | 0.661 | -0.016 |

Interpretation:

- TF-IDF currently outperforms semantic retrieval on both benchmark profiles using the project precision proxy.
- The largest gain is on the intellectual profile, where TF-IDF also preserves higher diversity.

From `evaluate_phase2_transition.py`:

| Profile | TF-IDF Precision | Phase2 Precision | Delta (Phase2 - TF-IDF) |
|---|---:|---:|---:|
| Intellectual Sci-Fi | 0.642 | 0.555 | -0.086 |
| Crime Drama Depth | 0.677 | 0.683 | +0.006 |

Interpretation:

- TF-IDF is better on Intellectual Sci-Fi.
- Phase2 is slightly better on Crime Drama Depth.
- So TF-IDF is not universally best across every profile, but is currently the strongest all-around baseline in this setup.

## 4. Why It Won in This Project

The current TF-IDF system had practical advantages over other solutions in this repository state:

1. Strong lexical precision on metadata-rich catalog text.
2. Explicit weighting of fields (summary/tags/genre emphasis) that matches recommendation intent.
3. Additional quality signals (depth/uniqueness/tone) that semantic-only retrieval does not natively enforce.
4. Lower action leakage in intellectual profiles due to explicit penalties and guardrails.
5. Deterministic, fast, and easy to debug behavior (no embedding model variance/download dependency).
6. Better precision proxy results on 2/2 TF-IDF-vs-semantic benchmark profiles.

## 5. Tradeoffs and Limits

TF-IDF can underperform when:

- Strong semantic paraphrase matching is needed across very different wording.
- Important relevance is implicit and not expressed in metadata terms.

Phase2 can surpass TF-IDF in some profile buckets once ranking features and training labels are tuned well, as seen in Crime Drama Depth (+0.006).

## 6. Reproducing the Numbers

From `app` directory:

```bash
source .venv/bin/activate
python evaluate_recommender.py
python evaluate_semantic_transition.py
python evaluate_phase2_transition.py
```

## 7. Runtime Backend Selection

The Flask app defaults to TF-IDF backend unless overridden:

- `RECOMMENDER_BACKEND=tfidf` (default)
- optional: `semantic`
- optional: `phase2`

This default is defined in `app.py` and also acts as a safe fallback if Phase2 initialization fails.
