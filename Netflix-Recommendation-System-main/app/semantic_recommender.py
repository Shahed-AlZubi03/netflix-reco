from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .ml_feature_pipeline import (
        build_train_ready_feature_store,
        clean_text,
        normalize_title_key,
        save_feature_store,
    )
except ImportError:
    from ml_feature_pipeline import (
        build_train_ready_feature_store,
        clean_text,
        normalize_title_key,
        save_feature_store,
    )


DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip().lower()


class SemanticNetflixRecommender:
    def __init__(
        self,
        csv_path: str,
        feature_store_path: str | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_batch_size: int = 128,
    ):
        raw_df = pd.read_csv(csv_path, encoding='latin-1')
        raw_df['Title'] = raw_df['Title'].astype(str).apply(clean_text)
        raw_df = raw_df[~raw_df['Title'].duplicated()].reset_index(drop=True)

        self.detail_columns = list(raw_df.columns)
        self.display_data = raw_df.copy()
        self.model_data = self._load_or_build_feature_store(
            raw_df=raw_df,
            feature_store_path=feature_store_path,
        )
        self.model_data = self._prepare_runtime_columns(self.model_data)

        self.embedding_model_name = embedding_model_name
        self.embedding_batch_size = embedding_batch_size
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_matrix = self.embedding_model.encode(
            self.model_data['embedding_text'].tolist(),
            batch_size=embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.title_index_map = pd.Series(self.model_data.index, index=self.model_data['title_key']).to_dict()

    @property
    def available_titles(self) -> List[str]:
        return sorted(self.display_data['Title'].tolist())

    @property
    def available_languages(self) -> List[str]:
        return sorted({lang for lang_set in self.model_data['language_set'] for lang in lang_set})

    def _load_or_build_feature_store(self, raw_df: pd.DataFrame, feature_store_path: str | None) -> pd.DataFrame:
        if not feature_store_path:
            return build_train_ready_feature_store(raw_df)

        feature_store = Path(feature_store_path)
        if feature_store.exists():
            if feature_store.suffix.lower() == '.parquet':
                return pd.read_parquet(feature_store)
            if feature_store.suffix.lower() == '.csv':
                return pd.read_csv(feature_store)

        built = build_train_ready_feature_store(raw_df)
        save_feature_store(
            built,
            output_path=str(feature_store),
            csv_fallback_path=str(feature_store.with_suffix('.csv')),
        )
        return built

    def _prepare_runtime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        if 'series_or_movie' not in prepared.columns:
            prepared['series_or_movie'] = prepared.get('Series or Movie', '').astype(str)
        if 'language_tokens' not in prepared.columns:
            if 'language_tokens_json' in prepared.columns:
                prepared['language_tokens'] = prepared['language_tokens_json']
            elif 'Languages' in prepared.columns:
                prepared['language_tokens'] = prepared['Languages']
            else:
                prepared['language_tokens'] = ''
        if 'imdb_score' not in prepared.columns and 'IMDb Score' in prepared.columns:
            prepared['imdb_score'] = prepared['IMDb Score']
        if 'depth_score' not in prepared.columns:
            prepared['depth_score'] = 0.0
        if 'uniqueness_score' not in prepared.columns:
            prepared['uniqueness_score'] = 0.0
        if 'intellectual_tone_score' not in prepared.columns:
            prepared['intellectual_tone_score'] = 0.0
        if 'action_tone_score' not in prepared.columns:
            prepared['action_tone_score'] = 0.0
        if 'votes_norm' not in prepared.columns:
            prepared['votes_norm'] = 0.0
        if 'embedding_text' not in prepared.columns:
            prepared['embedding_text'] = prepared['Title'].fillna('').astype(str)

        prepared['title_key'] = prepared['Title'].apply(normalize_title_key)
        prepared['series_or_movie'] = prepared['series_or_movie'].fillna('').astype(str).str.lower().str.strip()

        def parse_json_list(value: object) -> List[str]:
            if isinstance(value, list):
                return [str(item).strip().lower() for item in value if str(item).strip()]
            if isinstance(value, set):
                return [str(item).strip().lower() for item in value if str(item).strip()]
            text = str(value).strip()
            if not text:
                return []
            if text.startswith('[') and text.endswith(']'):
                try:
                    parsed = json.loads(text)
                    return [str(item).strip().lower() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    pass
            return [part.strip().lower() for part in text.split(',') if part.strip()]

        prepared['language_tokens'] = prepared['language_tokens'].apply(parse_json_list)
        prepared['language_set'] = prepared['language_tokens'].apply(set)
        prepared['depth_score'] = pd.to_numeric(prepared['depth_score'], errors='coerce').fillna(0.0)
        prepared['uniqueness_score'] = pd.to_numeric(prepared['uniqueness_score'], errors='coerce').fillna(0.0)
        prepared['intellectual_tone_score'] = pd.to_numeric(
            prepared['intellectual_tone_score'],
            errors='coerce',
        ).fillna(0.0)
        prepared['action_tone_score'] = pd.to_numeric(prepared['action_tone_score'], errors='coerce').fillna(0.0)
        prepared['votes_norm'] = pd.to_numeric(prepared['votes_norm'], errors='coerce').fillna(0.0)
        prepared['imdb_score'] = pd.to_numeric(prepared['imdb_score'], errors='coerce').fillna(0.0)
        prepared['embedding_text'] = prepared['embedding_text'].fillna('').astype(str)
        return prepared

    def _resolve_seed_indices(self, seed_titles: List[str]) -> List[int]:
        indices: List[int] = []
        for title in seed_titles:
            key = normalize_title_key(title)
            if key in self.title_index_map:
                indices.append(int(self.title_index_map[key]))
        return list(dict.fromkeys(indices))

    def _dominant_seed_type(self, seed_indices: List[int]) -> str:
        if not seed_indices:
            return ''
        types = [self.model_data.loc[idx, 'series_or_movie'] for idx in seed_indices]
        counts = Counter(types)
        return counts.most_common(1)[0][0] if counts else ''

    def _mmr_rerank(
        self,
        candidate_indices: List[int],
        base_scores: np.ndarray,
        top_k: int,
        mmr_lambda: float,
    ) -> List[int]:
        if not candidate_indices:
            return []

        vectors = self.embedding_matrix[candidate_indices]
        pairwise = cosine_similarity(vectors, vectors)
        min_score = float(base_scores.min())
        max_score = float(base_scores.max())
        if max_score == min_score:
            normalized = np.ones_like(base_scores)
        else:
            normalized = (base_scores - min_score) / (max_score - min_score)

        selected_positions: List[int] = []
        remaining = set(range(len(candidate_indices)))

        while remaining and len(selected_positions) < top_k:
            best_pos = None
            best_value = None
            for pos in remaining:
                if not selected_positions:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = float(max(pairwise[pos, chosen] for chosen in selected_positions))
                mmr_value = float(mmr_lambda * normalized[pos] - (1.0 - mmr_lambda) * diversity_penalty)
                if best_value is None or mmr_value > best_value:
                    best_value = mmr_value
                    best_pos = pos

            selected_positions.append(int(best_pos))
            remaining.remove(int(best_pos))

        return [candidate_indices[pos] for pos in selected_positions]

    def recommend(
        self,
        seed_titles: List[str],
        selected_languages: List[str],
        top_k: int = 60,
        candidate_pool_limit: int = 250,
        enforce_type_match: bool = True,
        mmr_lambda: float = 0.94,
    ) -> pd.DataFrame:
        seed_indices = self._resolve_seed_indices(seed_titles)
        if not seed_indices:
            return pd.DataFrame(columns=['Title', 'Image'])

        selected_language_set: Set[str] = {
            normalize_text(language) for language in selected_languages if normalize_text(language)
        }

        seed_vectors = self.embedding_matrix[seed_indices]
        all_vectors = self.embedding_matrix
        similarity_matrix = cosine_similarity(seed_vectors, all_vectors)
        sim_max = similarity_matrix.max(axis=0)
        sim_avg = similarity_matrix.mean(axis=0)
        semantic_similarity = 0.62 * sim_max + 0.38 * sim_avg

        score_df = self.model_data.copy()
        score_df['candidate_idx'] = score_df.index
        score_df['semantic_similarity'] = semantic_similarity

        seed_idx_set = set(seed_indices)
        score_df = score_df[~score_df['candidate_idx'].isin(seed_idx_set)]

        if selected_language_set:
            score_df = score_df[
                score_df['language_set'].apply(lambda language_set: bool(language_set & selected_language_set))
            ]

        if enforce_type_match:
            dominant_type = self._dominant_seed_type(seed_indices)
            if dominant_type:
                score_df = score_df[score_df['series_or_movie'] == dominant_type]

        if score_df.empty:
            return pd.DataFrame(columns=['Title', 'Image'])

        score_df['ranking_score'] = (
            0.64 * score_df['semantic_similarity']
            + 0.14 * score_df['depth_score']
            + 0.08 * score_df['uniqueness_score']
            + 0.08 * score_df['intellectual_tone_score']
            + 0.06 * (1 - score_df['action_tone_score'])
            - 0.05 * score_df['votes_norm']
        )

        score_df = score_df.sort_values(by=['ranking_score', 'imdb_score'], ascending=False)
        score_df = score_df.head(candidate_pool_limit).reset_index(drop=True)

        candidate_indices = score_df['candidate_idx'].astype(int).tolist()
        candidate_scores = score_df['ranking_score'].astype(float).to_numpy()
        selected_indices = self._mmr_rerank(
            candidate_indices=candidate_indices,
            base_scores=candidate_scores,
            top_k=top_k,
            mmr_lambda=mmr_lambda,
        )

        score_lookup = score_df.set_index('candidate_idx')['ranking_score'].to_dict()
        selected_rows = self.model_data.loc[selected_indices].copy()
        selected_rows['ranking_score'] = selected_rows.index.map(score_lookup).astype(float)
        selected_rows = selected_rows.sort_values(by='ranking_score', ascending=False)

        result_rows = []
        for row in selected_rows.itertuples(index=False):
            display_row = self.display_data[self.display_data['Title'] == row.Title]
            image_url = display_row['Image'].iloc[0] if not display_row.empty else ''
            result_rows.append(
                {
                    'Title': row.Title,
                    'Image': image_url,
                    'ranking_score': float(row.ranking_score),
                }
            )

        return pd.DataFrame(result_rows)

    def get_movie_details(self, movie_title: str) -> List[str]:
        details = self.display_data[self.display_data['Title'] == movie_title]
        if details.empty:
            return [movie_title] + ['Not available'] * 21
        return details[self.detail_columns].iloc[0].tolist()

    def profile_metrics(self, recommendations: pd.DataFrame) -> Dict[str, float]:
        if recommendations.empty:
            return {
                'avg_depth': 0.0,
                'avg_uniqueness': 0.0,
                'avg_intellectual_tone': 0.0,
                'avg_action_tone': 0.0,
            }

        joined = recommendations.merge(
            self.model_data[
                ['Title', 'depth_score', 'uniqueness_score', 'intellectual_tone_score', 'action_tone_score']
            ],
            on='Title',
            how='left',
        )
        return {
            'avg_depth': float(joined['depth_score'].mean()),
            'avg_uniqueness': float(joined['uniqueness_score'].mean()),
            'avg_intellectual_tone': float(joined['intellectual_tone_score'].mean()),
            'avg_action_tone': float(joined['action_tone_score'].mean()),
        }
