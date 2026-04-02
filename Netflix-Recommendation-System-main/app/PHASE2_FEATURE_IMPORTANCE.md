# Phase 2 LambdaMART Documentation

This document explains how Phase 2 works in the Netflix Recommendation System and shows the current trained feature importance rates.

## 1. What Phase 2 Does

Phase 2 is a learning-to-rank reranker built on LightGBM LambdaMART.

High-level flow:

1. Build query profile from selected seed titles.
2. Retrieve semantic candidates using embedding similarity.
3. Compute pairwise ranking features for each candidate.
4. Predict ranking score with LambdaMART.
5. Apply MMR diversification to avoid near-duplicates.
6. Return final recommendations.

Main implementation: `phase2_lambdamart.py`

## 2. Training Pipeline Summary

Training script: `train_phase2_lambdamart.py`

Default objective:

- `objective=lambdarank`
- `metric=ndcg`

Training data generation:

- Create weakly-supervised listwise samples from semantic candidate pools.
- Convert weak relevance into ordinal labels (0-4) by per-query quantiles.
- Train/validation split is query-based (not row-based).

Artifacts produced:

- `artifacts/lambdamart_ranker.joblib`
- `artifacts/lambdamart_training_samples.parquet`
- `artifacts/title_embeddings.npy`

## 3. Feature Importance Rates

Source artifact used for this report:

- `artifacts/lambdamart_ranker.joblib`
- Generated on current workspace state (Apr 2, 2026)

How rates are computed:

- `gain_pct`: normalized LightGBM gain importance, where
  - `gain_pct = feature_gain / sum(all_feature_gain) * 100`
- `split_pct`: normalized split frequency, where
  - `split_pct = feature_split_count / sum(all_feature_splits) * 100`

Notes:

- `gain_pct` is usually better for understanding contribution to ranking quality.
- `split_pct` shows how often a feature is used in tree splits.

### Current Importance Table (sorted by gain)

| Feature | Gain % | Split % | Interpretation |
|---|---:|---:|---|
| genre_jaccard | 43.4868 | 10.9005 | Dominant signal: overlap between seed genres and candidate genres. |
| tag_jaccard | 30.9784 | 15.0582 | Strong thematic match from textual tags/keywords. |
| language_overlap_ratio | 10.3246 | 8.5170 | Language alignment is highly influential. |
| country_overlap_ratio | 6.4639 | 11.7518 | Regional/origin alignment contributes meaningfully. |
| type_match | 3.4010 | 6.6487 | Series-vs-movie consistency helps relevance. |
| runtime_gap_norm | 1.5974 | 6.4651 | Duration similarity has moderate effect. |
| semantic_similarity | 0.6752 | 11.3396 | Used often, but with lower marginal gain after other features. |
| votes_norm | 0.5327 | 2.7151 | Popularity signal with small gain contribution. |
| action_gap_norm | 0.4923 | 3.1900 | Difference from action tendency in query profile. |
| release_year_gap_norm | 0.4354 | 7.2222 | Temporal proximity contributes modestly. |
| intellectual_gap_norm | 0.4140 | 1.3082 | Difference from intellectual profile. |
| depth_score | 0.3705 | 4.6998 | Candidate depth level contributes modestly. |
| uniqueness_score | 0.3339 | 2.2581 | Candidate uniqueness signal contributes modestly. |
| intellectual_tone_score | 0.1685 | 2.0789 | Absolute intellectual tone has limited direct gain. |
| writer_overlap | 0.1540 | 1.1828 | Writer overlap helps in narrower cases. |
| director_overlap | 0.0729 | 0.8423 | Director overlap contributes in fewer cases. |
| depth_gap_norm | 0.0636 | 2.7778 | Gap-to-profile depth contributes weakly. |
| action_tone_score | 0.0182 | 0.4615 | Absolute action tone has very low gain impact. |
| actor_overlap_ratio | 0.0169 | 0.5824 | Actor overlap has minimal impact in current model. |

## 4. Practical Reading of This Model

In the current trained model, ranking is mostly driven by:

1. Genre overlap
2. Tag overlap
3. Language alignment
4. Country alignment

This indicates the model is currently more metadata- and profile-alignment-driven than cast/crew-driven.

## 5. Recomputing Importances

Use this command from the `app` directory to recompute rates from the latest artifact:

```bash
./.venv/bin/python - <<'PY'
import joblib
import numpy as np
from phase2_lambdamart import FEATURE_COLUMNS

artifact = joblib.load('artifacts/lambdamart_ranker.joblib')
model = artifact['model']
booster = model.booster_
features = artifact.get('feature_columns', FEATURE_COLUMNS)
split = booster.feature_importance(importance_type='split')
gain = booster.feature_importance(importance_type='gain')

split_total = float(np.sum(split)) or 1.0
gain_total = float(np.sum(gain)) or 1.0
rows = []
for f, s, g in zip(features, split, gain):
    rows.append((f, int(s), float(g), (float(s)/split_total)*100.0, (float(g)/gain_total)*100.0))

rows.sort(key=lambda x: x[2], reverse=True)
print('feature\tsplit\tgain\tsplit_pct\tgain_pct')
for r in rows:
    print(f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.4f}\t{r[4]:.4f}")
PY
```
