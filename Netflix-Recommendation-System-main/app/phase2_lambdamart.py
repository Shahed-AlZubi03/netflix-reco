from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .ml_feature_pipeline import build_train_ready_feature_store, clean_text, normalize_title_key, save_feature_store
except ImportError:
    from ml_feature_pipeline import build_train_ready_feature_store, clean_text, normalize_title_key, save_feature_store


DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


FEATURE_COLUMNS = [
    'semantic_similarity',
    'genre_jaccard',
    'tag_jaccard',
    'language_overlap_ratio',
    'country_overlap_ratio',
    'actor_overlap_ratio',
    'director_overlap',
    'writer_overlap',
    'type_match',
    'runtime_gap_norm',
    'release_year_gap_norm',
    'depth_score',
    'uniqueness_score',
    'intellectual_tone_score',
    'action_tone_score',
    'votes_norm',
    'depth_gap_norm',
    'intellectual_gap_norm',
    'action_gap_norm',
]


@dataclass
class QueryProfile:
    indices: List[int]
    genre_set: Set[str]
    tag_set: Set[str]
    language_set: Set[str]
    country_set: Set[str]
    actor_set: Set[str]
    director_set: Set[str]
    writer_set: Set[str]
    dominant_type: str
    runtime_mean: float
    release_year_mean: float
    depth_mean: float
    intellectual_mean: float
    action_mean: float


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip().lower()


def parse_token_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if isinstance(value, set):
        return [normalize_text(item) for item in value if normalize_text(item)]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = json.loads(text)
            return [normalize_text(item) for item in parsed if normalize_text(item)]
        except json.JSONDecodeError:
            pass

    return [normalize_text(part) for part in text.split(',') if normalize_text(part)]


def parse_people_tokens(value: object) -> Set[str]:
    return set(parse_token_list(value))


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def overlap_ratio(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / max(1, len(left)))


class LambdaMARTPhase2Recommender:
    def __init__(
        self,
        csv_path: str,
        feature_store_path: str = 'artifacts/title_feature_store.parquet',
        model_artifact_path: str = 'artifacts/lambdamart_ranker.joblib',
        embeddings_cache_path: str = 'artifacts/title_embeddings.npy',
        allow_embedding_rebuild: bool = True,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_batch_size: int = 128,
        auto_train_if_missing: bool = False,
    ):
        raw_df = pd.read_csv(csv_path, encoding='latin-1')
        raw_df['Title'] = raw_df['Title'].astype(str).apply(clean_text)
        raw_df = raw_df[~raw_df['Title'].duplicated()].reset_index(drop=True)

        self.csv_path = csv_path
        self.feature_store_path = feature_store_path
        self.model_artifact_path = model_artifact_path
        self.embeddings_cache_path = embeddings_cache_path
        self.allow_embedding_rebuild = allow_embedding_rebuild
        self.detail_columns = list(raw_df.columns)
        self.display_data = raw_df.copy()

        self.model_data = self._load_or_build_feature_store(raw_df)
        self.model_data = self._prepare_runtime_columns(self.model_data)

        self.embedding_model_name = embedding_model_name
        self.embedding_batch_size = embedding_batch_size
        self.embedding_model = None
        self.embedding_matrix = self._load_or_build_embeddings(
            embeddings_cache_path=embeddings_cache_path,
            allow_rebuild=allow_embedding_rebuild,
        )

        self.title_index_map = pd.Series(self.model_data.index, index=self.model_data['title_key']).to_dict()
        self.model = None

        model_file = Path(model_artifact_path)
        if model_file.exists():
            self._load_ranker(model_file)
        elif auto_train_if_missing:
            self.train_ranker()

    def _build_embeddings_with_model(self) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        matrix = self.embedding_model.encode(
            self.model_data['embedding_text'].tolist(),
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(matrix, dtype=np.float32)

    def _load_or_build_embeddings(self, embeddings_cache_path: str, allow_rebuild: bool) -> np.ndarray:
        cache_path = Path(embeddings_cache_path)
        if cache_path.exists():
            matrix = np.load(cache_path)
            if len(matrix.shape) == 2 and matrix.shape[0] == len(self.model_data):
                return np.asarray(matrix, dtype=np.float32)

            if not allow_rebuild:
                raise RuntimeError(
                    f'Embedding cache shape mismatch at {cache_path}. '
                    'Run training to regenerate embeddings.'
                )

        if not allow_rebuild:
            raise RuntimeError(
                f'Embedding cache not found at {cache_path}. '
                'Run train_phase2_lambdamart.py first to build it.'
            )

        matrix = self._build_embeddings_with_model()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
        return matrix

    @property
    def available_titles(self) -> List[str]:
        return sorted(self.display_data['Title'].tolist())

    @property
    def available_languages(self) -> List[str]:
        languages = set()
        for language_set in self.model_data['language_set']:
            languages.update(language_set)
        return sorted(languages)

    def _load_or_build_feature_store(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        feature_path = Path(self.feature_store_path)
        if feature_path.exists():
            if feature_path.suffix.lower() == '.parquet':
                return pd.read_parquet(feature_path)
            if feature_path.suffix.lower() == '.csv':
                return pd.read_csv(feature_path)

        built = build_train_ready_feature_store(raw_df)
        save_feature_store(
            built,
            output_path=str(feature_path),
            csv_fallback_path=str(feature_path.with_suffix('.csv')),
        )
        return built

    def _prepare_runtime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        if 'series_or_movie' not in prepared.columns:
            prepared['series_or_movie'] = prepared.get('Series or Movie', '').astype(str)
        if 'embedding_text' not in prepared.columns:
            prepared['embedding_text'] = prepared['Title'].fillna('').astype(str)

        prepared['title_key'] = prepared['Title'].apply(normalize_title_key)
        prepared['series_or_movie'] = prepared['series_or_movie'].fillna('').astype(str).str.lower().str.strip()

        prepared['genre_set'] = prepared.get('genre_tokens', prepared.get('genre_tokens_json', '')).apply(
            lambda value: set(parse_token_list(value))
        )
        prepared['tag_set'] = prepared.get('tag_tokens', prepared.get('tag_tokens_json', '')).apply(
            lambda value: set(parse_token_list(value))
        )
        prepared['language_set'] = prepared.get('language_tokens', prepared.get('language_tokens_json', '')).apply(
            lambda value: set(parse_token_list(value))
        )
        prepared['country_set'] = prepared.get('country_tokens', prepared.get('country_tokens_json', '')).apply(
            lambda value: set(parse_token_list(value))
        )
        prepared['actor_set'] = prepared.get('Actors', '').apply(parse_people_tokens)
        prepared['director_set'] = prepared.get('Director', '').apply(parse_people_tokens)
        prepared['writer_set'] = prepared.get('Writer', '').apply(parse_people_tokens)

        numeric_columns = [
            'runtime_minutes',
            'release_year',
            'depth_score',
            'uniqueness_score',
            'intellectual_tone_score',
            'action_tone_score',
            'votes_norm',
            'imdb_score',
        ]
        defaults = {
            'runtime_minutes': 90.0,
            'release_year': 2005.0,
            'depth_score': 0.0,
            'uniqueness_score': 0.0,
            'intellectual_tone_score': 0.0,
            'action_tone_score': 0.0,
            'votes_norm': 0.0,
            'imdb_score': 0.0,
        }
        for column in numeric_columns:
            if column not in prepared.columns:
                prepared[column] = defaults[column]
            prepared[column] = pd.to_numeric(prepared[column], errors='coerce').fillna(defaults[column])

        prepared['embedding_text'] = prepared['embedding_text'].fillna('').astype(str)
        return prepared

    def _resolve_seed_indices(self, seed_titles: List[str]) -> List[int]:
        indices: List[int] = []
        for title in seed_titles:
            key = normalize_title_key(title)
            if key in self.title_index_map:
                indices.append(int(self.title_index_map[key]))
        return list(dict.fromkeys(indices))

    def _build_query_profile(self, seed_indices: List[int]) -> QueryProfile:
        seed_rows = self.model_data.loc[seed_indices]

        def merge_set(column: str) -> Set[str]:
            merged: Set[str] = set()
            for entry in seed_rows[column].tolist():
                merged.update(entry)
            return merged

        type_counts = seed_rows['series_or_movie'].value_counts()
        dominant_type = str(type_counts.index[0]) if not type_counts.empty else ''

        return QueryProfile(
            indices=seed_indices,
            genre_set=merge_set('genre_set'),
            tag_set=merge_set('tag_set'),
            language_set=merge_set('language_set'),
            country_set=merge_set('country_set'),
            actor_set=merge_set('actor_set'),
            director_set=merge_set('director_set'),
            writer_set=merge_set('writer_set'),
            dominant_type=dominant_type,
            runtime_mean=float(seed_rows['runtime_minutes'].mean()),
            release_year_mean=float(seed_rows['release_year'].mean()),
            depth_mean=float(seed_rows['depth_score'].mean()),
            intellectual_mean=float(seed_rows['intellectual_tone_score'].mean()),
            action_mean=float(seed_rows['action_tone_score'].mean()),
        )

    def _pair_features(self, query_profile: QueryProfile, candidate_row: pd.Series, semantic_similarity: float) -> Dict[str, float]:
        genre_jaccard = jaccard_similarity(query_profile.genre_set, candidate_row['genre_set'])
        tag_jaccard = jaccard_similarity(query_profile.tag_set, candidate_row['tag_set'])
        language_overlap_ratio = overlap_ratio(query_profile.language_set, candidate_row['language_set'])
        country_overlap_ratio = overlap_ratio(query_profile.country_set, candidate_row['country_set'])
        actor_overlap_ratio = overlap_ratio(query_profile.actor_set, candidate_row['actor_set'])
        director_overlap = 1.0 if (query_profile.director_set & candidate_row['director_set']) else 0.0
        writer_overlap = 1.0 if (query_profile.writer_set & candidate_row['writer_set']) else 0.0
        type_match = 1.0 if query_profile.dominant_type and candidate_row['series_or_movie'] == query_profile.dominant_type else 0.0

        runtime_gap_norm = min(abs(float(candidate_row['runtime_minutes']) - query_profile.runtime_mean) / 180.0, 1.0)
        release_year_gap_norm = min(abs(float(candidate_row['release_year']) - query_profile.release_year_mean) / 40.0, 1.0)

        depth_gap_norm = min(abs(float(candidate_row['depth_score']) - query_profile.depth_mean), 1.0)
        intellectual_gap_norm = min(abs(float(candidate_row['intellectual_tone_score']) - query_profile.intellectual_mean), 1.0)
        action_gap_norm = min(abs(float(candidate_row['action_tone_score']) - query_profile.action_mean), 1.0)

        return {
            'semantic_similarity': float(semantic_similarity),
            'genre_jaccard': float(genre_jaccard),
            'tag_jaccard': float(tag_jaccard),
            'language_overlap_ratio': float(language_overlap_ratio),
            'country_overlap_ratio': float(country_overlap_ratio),
            'actor_overlap_ratio': float(actor_overlap_ratio),
            'director_overlap': float(director_overlap),
            'writer_overlap': float(writer_overlap),
            'type_match': float(type_match),
            'runtime_gap_norm': float(runtime_gap_norm),
            'release_year_gap_norm': float(release_year_gap_norm),
            'depth_score': float(candidate_row['depth_score']),
            'uniqueness_score': float(candidate_row['uniqueness_score']),
            'intellectual_tone_score': float(candidate_row['intellectual_tone_score']),
            'action_tone_score': float(candidate_row['action_tone_score']),
            'votes_norm': float(candidate_row['votes_norm']),
            'depth_gap_norm': float(depth_gap_norm),
            'intellectual_gap_norm': float(intellectual_gap_norm),
            'action_gap_norm': float(action_gap_norm),
        }

    def _weak_relevance(self, features: Dict[str, float]) -> float:
        relevance = (
            1.80 * features['semantic_similarity']
            + 1.55 * features['genre_jaccard']
            + 1.25 * features['tag_jaccard']
            + 0.90 * features['language_overlap_ratio']
            + 0.30 * features['country_overlap_ratio']
            + 0.40 * features['actor_overlap_ratio']
            + 0.40 * features['director_overlap']
            + 0.25 * features['writer_overlap']
            + 0.45 * features['type_match']
            + 0.35 * features['depth_score']
            + 0.10 * features['intellectual_tone_score']
            - 0.35 * features['runtime_gap_norm']
            - 0.20 * features['release_year_gap_norm']
            - 0.25 * features['depth_gap_norm']
            - 0.20 * features['action_gap_norm']
            - 0.14 * features['votes_norm']
        )

        if features['type_match'] == 0.0:
            relevance -= 0.25
        if features['language_overlap_ratio'] == 0.0 and features['semantic_similarity'] < 0.22:
            relevance -= 0.30
        return float(relevance)

    def build_listwise_training_samples(
        self,
        candidate_pool_limit: int = 180,
        max_queries: int = 2000,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_seed)
        all_indices = np.arange(len(self.model_data))
        if max_queries > 0 and max_queries < len(all_indices):
            query_indices = rng.choice(all_indices, size=max_queries, replace=False)
            query_indices = np.sort(query_indices)
        else:
            query_indices = all_indices

        rows: List[Dict[str, float]] = []

        for query_idx in query_indices:
            query_vector = self.embedding_matrix[query_idx : query_idx + 1]
            sims = cosine_similarity(query_vector, self.embedding_matrix)[0]
            sims[query_idx] = -1.0
            candidate_indices = np.argsort(-sims)[:candidate_pool_limit]

            query_profile = self._build_query_profile([int(query_idx)])
            per_query_entries: List[Dict[str, float]] = []
            raw_scores: List[float] = []

            for candidate_idx in candidate_indices:
                candidate_row = self.model_data.iloc[int(candidate_idx)]
                feature_values = self._pair_features(
                    query_profile=query_profile,
                    candidate_row=candidate_row,
                    semantic_similarity=float(sims[int(candidate_idx)]),
                )
                raw_score = self._weak_relevance(feature_values)
                raw_scores.append(raw_score)

                entry = {
                    'query_idx': int(query_idx),
                    'candidate_idx': int(candidate_idx),
                    **feature_values,
                }
                per_query_entries.append(entry)

            score_array = np.array(raw_scores, dtype=float)
            if len(score_array) == 0:
                continue

            if np.allclose(score_array, score_array[0]):
                labels = np.zeros(len(score_array), dtype=int)
            else:
                thresholds = np.quantile(score_array, [0.20, 0.40, 0.60, 0.80])
                labels = np.zeros(len(score_array), dtype=int)
                labels += score_array > thresholds[0]
                labels += score_array > thresholds[1]
                labels += score_array > thresholds[2]
                labels += score_array > thresholds[3]

            for entry, label in zip(per_query_entries, labels.tolist()):
                entry['label'] = int(label)
                rows.append(entry)

        return pd.DataFrame(rows)

    def train_ranker(
        self,
        training_data_path: str = 'artifacts/lambdamart_training_samples.parquet',
        model_artifact_path: str | None = None,
        candidate_pool_limit: int = 180,
        max_queries: int = 2000,
        random_seed: int = 42,
    ) -> Dict[str, float]:
        training_df = self.build_listwise_training_samples(
            candidate_pool_limit=candidate_pool_limit,
            max_queries=max_queries,
            random_seed=random_seed,
        )
        if training_df.empty:
            raise RuntimeError('No training samples were generated for LambdaMART training.')

        training_df = training_df.sort_values(by=['query_idx', 'candidate_idx']).reset_index(drop=True)

        training_path = Path(training_data_path)
        training_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            training_df.to_parquet(training_path, index=False)
        except Exception:
            training_df.to_csv(training_path.with_suffix('.csv'), index=False)

        query_ids = training_df['query_idx'].unique().tolist()
        rng = np.random.default_rng(random_seed)
        rng.shuffle(query_ids)

        split_point = max(1, int(len(query_ids) * 0.85))
        train_queries = set(query_ids[:split_point])
        valid_queries = set(query_ids[split_point:]) if len(query_ids) > 1 else set()

        train_df = training_df[training_df['query_idx'].isin(train_queries)].copy()
        if valid_queries:
            valid_df = training_df[training_df['query_idx'].isin(valid_queries)].copy()
        else:
            valid_df = train_df.sample(frac=0.1, random_state=random_seed)

        train_group = train_df.groupby('query_idx').size().tolist()
        valid_group = valid_df.groupby('query_idx').size().tolist()

        x_train = train_df[FEATURE_COLUMNS]
        y_train = train_df['label'].astype(int)
        x_valid = valid_df[FEATURE_COLUMNS]
        y_valid = valid_df['label'].astype(int)

        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            n_estimators=360,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            colsample_bytree=0.85,
            subsample=0.85,
            random_state=random_seed,
            verbosity=-1,
        )

        ranker.fit(
            x_train,
            y_train,
            group=train_group,
            eval_set=[(x_valid, y_valid)],
            eval_group=[valid_group],
            eval_at=[5, 10],
        )

        model_path = Path(model_artifact_path or self.model_artifact_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            'model': ranker,
            'feature_columns': FEATURE_COLUMNS,
            'embedding_model_name': self.embedding_model_name,
        }
        joblib.dump(artifact, model_path)
        self.model = ranker

        cache_path = Path(self.embeddings_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.asarray(self.embedding_matrix, dtype=np.float32))

        summary = {
            'training_rows': float(len(training_df)),
            'training_queries': float(len(query_ids)),
            'train_rows': float(len(train_df)),
            'valid_rows': float(len(valid_df)),
        }
        return summary

    def _load_ranker(self, model_path: Path) -> None:
        artifact = joblib.load(model_path)
        self.model = artifact['model']

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
            best_position = None
            best_value = None

            for position in remaining:
                if not selected_positions:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = float(max(pairwise[position, selected] for selected in selected_positions))

                mmr_value = float(mmr_lambda * normalized[position] - (1.0 - mmr_lambda) * diversity_penalty)
                if best_value is None or mmr_value > best_value:
                    best_value = mmr_value
                    best_position = position

            selected_positions.append(int(best_position))
            remaining.remove(int(best_position))

        return [candidate_indices[position] for position in selected_positions]

    def _predict_candidate_scores(self, query_profile: QueryProfile, candidate_indices: List[int], semantic_scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('LambdaMART model is not loaded. Train or load a model before recommending.')

        feature_rows: List[Dict[str, float]] = []
        for candidate_idx in candidate_indices:
            candidate_row = self.model_data.loc[int(candidate_idx)]
            feature_values = self._pair_features(
                query_profile=query_profile,
                candidate_row=candidate_row,
                semantic_similarity=float(semantic_scores[int(candidate_idx)]),
            )
            feature_rows.append(feature_values)

        feature_df = pd.DataFrame(feature_rows)
        feature_df = feature_df[FEATURE_COLUMNS]
        return self.model.predict(feature_df)

    def recommend(
        self,
        seed_titles: List[str],
        selected_languages: List[str],
        top_k: int = 60,
        candidate_pool_limit: int = 250,
        enforce_type_match: bool = True,
        mmr_lambda: float = 0.92,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError('LambdaMART model is not loaded. Train it with train_ranker() first.')

        seed_indices = self._resolve_seed_indices(seed_titles)
        if not seed_indices:
            return pd.DataFrame(columns=['Title', 'Image'])

        query_profile = self._build_query_profile(seed_indices)
        selected_language_set: Set[str] = {
            normalize_text(language) for language in selected_languages if normalize_text(language)
        }

        seed_vectors = self.embedding_matrix[seed_indices]
        centroid = seed_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        semantic_scores = cosine_similarity(centroid.reshape(1, -1), self.embedding_matrix)[0]

        score_df = self.model_data.copy()
        score_df['candidate_idx'] = score_df.index
        score_df['semantic_similarity'] = semantic_scores
        score_df = score_df[~score_df['candidate_idx'].isin(seed_indices)]

        if selected_language_set:
            score_df = score_df[
                score_df['language_set'].apply(lambda language_set: bool(language_set & selected_language_set))
            ]

        if enforce_type_match and query_profile.dominant_type:
            score_df = score_df[score_df['series_or_movie'] == query_profile.dominant_type]

        if score_df.empty:
            return pd.DataFrame(columns=['Title', 'Image'])

        score_df = score_df.sort_values(by='semantic_similarity', ascending=False)
        score_df = score_df.head(max(candidate_pool_limit, top_k * 2)).reset_index(drop=True)

        candidate_indices = score_df['candidate_idx'].astype(int).tolist()
        predicted_scores = self._predict_candidate_scores(
            query_profile=query_profile,
            candidate_indices=candidate_indices,
            semantic_scores=semantic_scores,
        )

        score_df['ranking_score'] = predicted_scores
        score_df = score_df.sort_values(by='ranking_score', ascending=False).reset_index(drop=True)

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
