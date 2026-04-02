import re
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(title))


def parse_tokens(value: object) -> List[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    return [part.strip() for part in normalized.split(',') if part.strip()]


def to_feature_token(value: object) -> str:
    token = normalize_text(value)
    token = re.sub(r'[^a-z0-9]+', '_', token)
    return token.strip('_')


def build_prefixed_tokens(tokens: Iterable[str], prefix: str) -> str:
    prefixed: List[str] = []
    for token in tokens:
        normalized = to_feature_token(token)
        if normalized:
            prefixed.append(f'{prefix}_{normalized}')
    return ' '.join(prefixed)


def min_max_scale(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def build_rarity_scores(token_series: pd.Series) -> pd.Series:
    token_frequency: Dict[str, int] = {}
    for tokens in token_series:
        for token in tokens:
            token_frequency[token] = token_frequency.get(token, 0) + 1

    rarity_scores: List[float] = []
    for tokens in token_series:
        if not tokens:
            rarity_scores.append(0.0)
            continue
        rarity_scores.append(sum(1.0 / token_frequency[token] for token in tokens) / len(tokens))

    return pd.Series(rarity_scores, index=token_series.index)


def keyword_hit_score(text: str, keywords: Iterable[str]) -> float:
    if not text:
        return 0.0
    hit_count = sum(1 for keyword in keywords if keyword in text)
    return float(hit_count)


class NetflixRecommender:
    def __init__(self, csv_path: str):
        raw_df = pd.read_csv(csv_path, encoding='latin-1')
        raw_df.rename(columns={'View Rating': 'ViewerRating'}, inplace=True)
        raw_df['Title'] = raw_df['Title'].astype(str).str.strip()
        raw_df = raw_df[~raw_df['Title'].duplicated()].reset_index(drop=True)

        self.detail_columns = list(raw_df.columns)
        self.display_data = raw_df.copy()
        self.model_data = raw_df.copy()

        self._build_features()

    def _infer_seed_intent(self, seed_titles: List[str]) -> Dict[str, float]:
        seed_keys = {normalize_title_key(title) for title in seed_titles}
        seed_rows = self.model_data[self.model_data['title_key'].isin(seed_keys)]

        if seed_rows.empty:
            return {
                'intellectual_focus': 0.5,
                'action_focus': 0.5,
                'depth_focus': 0.5,
                'mmr_lambda': 0.97,
            }

        intellectual_focus = float(seed_rows['intellectual_tone_score'].mean())
        action_focus = float(seed_rows['action_tone_score'].mean())
        depth_focus = float(seed_rows['depth_score'].mean())

        intent_gap = intellectual_focus - action_focus
        if intent_gap >= 0.10:
            mmr_lambda = 1.00
        elif intent_gap <= -0.10:
            mmr_lambda = 0.95
        else:
            mmr_lambda = 0.98

        return {
            'intellectual_focus': intellectual_focus,
            'action_focus': action_focus,
            'depth_focus': depth_focus,
            'mmr_lambda': mmr_lambda,
        }

    @property
    def available_titles(self) -> List[str]:
        return sorted(self.display_data['Title'].tolist())

    @property
    def available_languages(self) -> List[str]:
        return sorted({lang for lang_set in self.model_data['language_set'] for lang in lang_set})

    def _build_features(self) -> None:
        text_columns = [
            'Genre', 'Tags', 'Actors', 'ViewerRating', 'Summary',
            'Director', 'Writer', 'Languages'
        ]
        for column in text_columns:
            self.model_data[column] = self.model_data[column].apply(normalize_text)

        self.model_data['IMDb Score'] = pd.to_numeric(self.model_data['IMDb Score'], errors='coerce').fillna(6.6)
        self.model_data['Awards Received'] = pd.to_numeric(self.model_data['Awards Received'], errors='coerce').fillna(0)
        self.model_data['Awards Nominated For'] = pd.to_numeric(self.model_data['Awards Nominated For'], errors='coerce').fillna(0)
        self.model_data['IMDb Votes'] = pd.to_numeric(
            self.model_data['IMDb Votes'].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        ).fillna(0)

        self.model_data['genre_tokens'] = self.model_data['Genre'].apply(parse_tokens)
        self.model_data['tag_tokens'] = self.model_data['Tags'].apply(parse_tokens)
        self.model_data['language_set'] = self.model_data['Languages'].apply(lambda value: set(parse_tokens(value)))
        self.model_data['summary_word_count'] = self.model_data['Summary'].str.split().str.len().fillna(0)

        imdb_norm = min_max_scale(self.model_data['IMDb Score'])
        awards_norm = min_max_scale(self.model_data['Awards Received'] + self.model_data['Awards Nominated For'])
        summary_norm = min_max_scale(self.model_data['summary_word_count'])
        self.model_data['depth_score'] = 0.50 * imdb_norm + 0.30 * awards_norm + 0.20 * summary_norm

        genre_rarity = build_rarity_scores(self.model_data['genre_tokens'])
        tag_rarity = build_rarity_scores(self.model_data['tag_tokens'])
        rarity_norm = min_max_scale(0.60 * genre_rarity + 0.40 * tag_rarity)
        votes_norm = min_max_scale(self.model_data['IMDb Votes'])
        self.model_data['uniqueness_score'] = 0.65 * rarity_norm + 0.35 * (1 - votes_norm)
        self.model_data['votes_norm'] = votes_norm

        intellectual_keywords = {
            'philosoph', 'existential', 'dystopian', 'consciousness', 'identity', 'memory',
            'mind', 'time', 'space', 'society', 'moral', 'meaning', 'character', 'drama',
            'psychological', 'human', 'humanity', 'nonlinear', 'mystery', 'political'
        }
        action_keywords = {
            'action', 'adventure', 'superhero', 'battle', 'war', 'alien invasion',
            'monster', 'robot', 'mecha', 'fight', 'explosive', 'spectacle', 'mission',
            'mercenary', 'survival action'
        }
        emotional_keywords = {
            'grief', 'loss', 'family', 'relationship', 'sacrifice', 'redemption',
            'betrayal', 'hope', 'trauma', 'friendship', 'romance', 'father', 'mother'
        }

        text_blob = (
            self.model_data['Genre'] + ' ' + self.model_data['Tags'] + ' ' + self.model_data['Summary']
        )
        self.model_data['intellectual_signal_raw'] = text_blob.apply(
            lambda value: keyword_hit_score(value, intellectual_keywords)
        )
        self.model_data['action_signal_raw'] = text_blob.apply(
            lambda value: keyword_hit_score(value, action_keywords)
        )
        self.model_data['emotional_signal_raw'] = text_blob.apply(
            lambda value: keyword_hit_score(value, emotional_keywords)
        )

        intellectual_norm = min_max_scale(self.model_data['intellectual_signal_raw'])
        action_norm = min_max_scale(self.model_data['action_signal_raw'])
        emotional_norm = min_max_scale(self.model_data['emotional_signal_raw'])

        self.model_data['intellectual_tone_score'] = (0.7 * intellectual_norm + 0.3 * emotional_norm)
        self.model_data['action_tone_score'] = action_norm

        weighted_soup = []
        for _, row in self.model_data.iterrows():
            genre_signature = build_prefixed_tokens(row['genre_tokens'], 'genre')
            tag_signature = build_prefixed_tokens(row['tag_tokens'], 'tag')
            language_signature = build_prefixed_tokens(sorted(row['language_set']), 'lang')
            type_signature = ''
            series_or_movie = row.get('Series or Movie', '')
            type_token = to_feature_token(series_or_movie)
            if type_token:
                type_signature = f'type_{type_token}'

            chunks = [
                " ".join([row['Genre']] * 3),
                " ".join([row['Tags']] * 3),
                " ".join([row['Summary']] * 4),
                " ".join([row['Actors']] * 2),
                " ".join([row['Director']] * 2),
                " ".join([row['Writer']] * 2),
                " ".join([row['Languages']] * 2),
                row['ViewerRating'],
                " ".join([genre_signature] * 2),
                tag_signature,
                language_signature,
                type_signature,
            ]
            weighted_soup.append(" ".join(chunk for chunk in chunks if chunk).strip())

        self.model_data['soup'] = weighted_soup
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            strip_accents='unicode',
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.model_data['soup'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        self.model_data['title_key'] = self.model_data['Title'].apply(normalize_title_key)
        self.title_index_map = pd.Series(self.model_data.index, index=self.model_data['title_key']).to_dict()

    def _get_similar_items(self, seed_title: str, max_items: int = 120) -> List[tuple[int, float]]:
        key = normalize_title_key(seed_title)
        if key not in self.title_index_map:
            return []
        idx = self.title_index_map[key]
        scores = list(enumerate(self.cosine_sim[idx]))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(candidate_idx, float(score)) for candidate_idx, score in scores[1:max_items + 1]]

    def _resolve_seed_indices(self, seed_titles: List[str]) -> List[int]:
        seen: Set[int] = set()
        indices: List[int] = []

        for title in seed_titles:
            idx = self.title_index_map.get(normalize_title_key(title))
            if idx is None or idx in seen:
                continue
            seen.add(int(idx))
            indices.append(int(idx))

        return indices

    def _aggregate_seed_signals(self, seed_indices: List[int]) -> tuple[np.ndarray, np.ndarray]:
        seed_sim_matrix = self.cosine_sim[seed_indices]
        if seed_sim_matrix.ndim == 1:
            seed_sim_matrix = seed_sim_matrix.reshape(1, -1)

        sim_max = seed_sim_matrix.max(axis=0)
        sim_avg = seed_sim_matrix.mean(axis=0)

        # Encourage candidates that are relevant to multiple seeds, not only one outlier seed.
        support_ratio = (seed_sim_matrix >= 0.10).mean(axis=0)
        combined_similarity = 0.58 * sim_max + 0.30 * sim_avg + 0.12 * support_ratio

        return combined_similarity.astype(float), support_ratio.astype(float)

    def _mmr_rerank(self, pool_df: pd.DataFrame, top_k: int, mmr_lambda: float) -> pd.DataFrame:
        if pool_df.empty:
            return pool_df

        candidate_indices = pool_df['candidate_idx'].tolist()
        pairwise = cosine_similarity(self.tfidf_matrix[candidate_indices], self.tfidf_matrix[candidate_indices])

        min_score = pool_df['base_score'].min()
        max_score = pool_df['base_score'].max()
        score_span = max_score - min_score
        if score_span == 0:
            base_norm = [1.0] * len(pool_df)
        else:
            base_norm = [float((score - min_score) / score_span) for score in pool_df['base_score'].tolist()]

        selected_positions: List[int] = []
        remaining_positions = set(range(len(candidate_indices)))

        while remaining_positions and len(selected_positions) < top_k:
            best_position = None
            best_mmr_score = None

            for position in remaining_positions:
                if not selected_positions:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = max(pairwise[position, selected] for selected in selected_positions)

                mmr_score = mmr_lambda * base_norm[position] - (1.0 - mmr_lambda) * diversity_penalty
                if best_mmr_score is None or mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_position = position

            selected_positions.append(best_position)
            remaining_positions.remove(best_position)

        ranked = pool_df.iloc[selected_positions].copy()
        ranked.reset_index(drop=True, inplace=True)
        return ranked

    def recommend(
        self,
        seed_titles: List[str],
        selected_languages: List[str],
        top_k: int = 60,
        candidate_pool_limit: int = 220,
        mmr_lambda: float | None = None,
    ) -> pd.DataFrame:
        language_set: Set[str] = {normalize_text(lang) for lang in selected_languages if normalize_text(lang)}
        seed_indices = self._resolve_seed_indices(seed_titles)
        if not seed_indices:
            return pd.DataFrame(columns=['Title', 'Image'])

        intent = self._infer_seed_intent(seed_titles)
        if mmr_lambda is None:
            mmr_lambda = intent['mmr_lambda']

        intellectual_mode = intent['intellectual_focus'] >= (intent['action_focus'] + 0.05)

        similarity_scores, support_ratios = self._aggregate_seed_signals(seed_indices)
        seed_idx_set = set(seed_indices)
        candidate_rows = []

        for candidate_idx, candidate in self.model_data.iterrows():
            if candidate_idx in seed_idx_set:
                continue

            if language_set and not (candidate['language_set'] & language_set):
                continue

            # In intellectual mode, drop action-skewed low-depth picks that caused noise.
            if intellectual_mode and (
                candidate['action_tone_score'] > candidate['intellectual_tone_score'] + 0.12
                and candidate['depth_score'] < 0.62
            ):
                continue

            combined_similarity = float(similarity_scores[candidate_idx])
            seed_support_bonus = 0.04 * float(support_ratios[candidate_idx])

            mainstream_penalty = (
                max(0.0, candidate['votes_norm'] - 0.78) * (1 - candidate['uniqueness_score'])
                + 1.00 * candidate['action_tone_score'] * (1 - candidate['depth_score'])
            )

            precision_bonus = (
                0.09 * candidate['intellectual_tone_score'] * intent['intellectual_focus']
                + 0.07 * candidate['depth_score'] * intent['depth_focus']
            )

            base_score = (
                0.55 * combined_similarity
                + 0.21 * candidate['depth_score']
                + 0.07 * candidate['uniqueness_score']
                + 0.13 * candidate['intellectual_tone_score']
                + 0.09 * (1 - candidate['action_tone_score'])
                + precision_bonus
                + seed_support_bonus
                - 0.16 * mainstream_penalty
            )

            candidate_rows.append({
                'candidate_idx': candidate_idx,
                'base_score': base_score,
                'IMDb Score': candidate['IMDb Score'],
            })

        if not candidate_rows:
            return pd.DataFrame(columns=['Title', 'Image'])

        pool_df = pd.DataFrame(candidate_rows)
        pool_df = pool_df.sort_values(by=['base_score', 'IMDb Score'], ascending=False)
        pool_df = pool_df.head(candidate_pool_limit).reset_index(drop=True)
        reranked_df = self._mmr_rerank(pool_df, top_k=top_k, mmr_lambda=mmr_lambda)

        result_rows = []
        for _, row in reranked_df.iterrows():
            candidate_idx = int(row['candidate_idx'])
            display_row = self.display_data.loc[candidate_idx]
            result_rows.append({
                'Title': display_row['Title'],
                'Image': display_row['Image'],
                'base_score': float(row['base_score']),
            })

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
            self.model_data[['Title', 'depth_score', 'uniqueness_score', 'intellectual_tone_score', 'action_tone_score']],
            on='Title',
            how='left',
        )

        return {
            'avg_depth': float(joined['depth_score'].mean()),
            'avg_uniqueness': float(joined['uniqueness_score'].mean()),
            'avg_intellectual_tone': float(joined['intellectual_tone_score'].mean()),
            'avg_action_tone': float(joined['action_tone_score'].mean()),
        }
