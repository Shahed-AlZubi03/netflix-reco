import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


try:
    from ftfy import fix_text as fix_mojibake
except ImportError:
    fix_mojibake = None


REQUIRED_COLUMNS = [
    'Title',
    'Genre',
    'Tags',
    'Languages',
    'Country Availability',
    'Runtime',
    'Director',
    'Writer',
    'Actors',
    'View Rating',
    'IMDb Score',
    'Awards Received',
    'Awards Nominated For',
    'Release Date',
    'Netflix Release Date',
    'Summary',
    'Series or Movie',
    'IMDb Votes',
    'Image',
]


TOKEN_ALIASES: Dict[str, str] = {
    'tv programmes': 'tv shows',
    'tv programme': 'tv shows',
    'us tv programmes': 'us tv shows',
    'uk tv programmes': 'uk tv shows',
    'british programmes': 'british shows',
    'crime tv programmes': 'crime tv shows',
    'tv sci fi and fantasy': 'tv sci-fi and fantasy',
    'sci fi and fantasy': 'sci-fi and fantasy',
    'action and adventure': 'action adventure',
    'children and family movies': 'children family movies',
}


INTELLECTUAL_KEYWORDS = {
    'philosoph',
    'existential',
    'dystopian',
    'consciousness',
    'identity',
    'memory',
    'mind',
    'time',
    'space',
    'society',
    'moral',
    'meaning',
    'character',
    'drama',
    'psychological',
    'human',
    'humanity',
    'nonlinear',
    'mystery',
    'political',
}


ACTION_KEYWORDS = {
    'action',
    'adventure',
    'superhero',
    'battle',
    'war',
    'alien invasion',
    'monster',
    'robot',
    'mecha',
    'fight',
    'explosive',
    'spectacle',
    'mission',
    'mercenary',
    'survival action',
}


EMOTIONAL_KEYWORDS = {
    'grief',
    'loss',
    'family',
    'relationship',
    'sacrifice',
    'redemption',
    'betrayal',
    'hope',
    'trauma',
    'friendship',
    'romance',
    'father',
    'mother',
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ''
    text = str(value)
    if fix_mojibake is not None:
        text = fix_mojibake(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2019', "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_title_key(title: str) -> str:
    title = clean_text(title).lower()
    return re.sub(r'[^a-z0-9]', '', title)


def normalize_token(token: str) -> str:
    token = clean_text(token).lower()
    token = token.replace('&', ' and ')
    token = token.replace('/', ' ')
    token = token.replace("'", '')
    token = re.sub(r'[^a-z0-9\s-]', ' ', token)
    token = re.sub(r'\s+', ' ', token).strip()
    return TOKEN_ALIASES.get(token, token)


def split_and_canonicalize(value: object) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    tokens = [normalize_token(part) for part in text.split(',')]
    tokens = [token for token in tokens if token]
    deduped = list(dict.fromkeys(tokens))
    return deduped


def parse_runtime_minutes(value: object) -> float:
    text = clean_text(value).lower()
    if not text:
        return float('nan')

    if '< 30' in text:
        return 25.0
    if '1-2 hour' in text or '1 - 2 hour' in text:
        return 90.0
    if '> 2' in text and ('hour' in text or 'hr' in text):
        return 150.0

    range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*(minutes|minute|mins|min|hours|hour|hrs|hr)', text)
    if range_match:
        start_value = float(range_match.group(1))
        end_value = float(range_match.group(2))
        unit = range_match.group(3)
        midpoint = (start_value + end_value) / 2.0
        if unit.startswith('hour') or unit.startswith('hr'):
            return midpoint * 60.0
        return midpoint

    single_match = re.search(r'(\d+(?:\.\d+)?)\s*(minutes|minute|mins|min|hours|hour|hrs|hr)', text)
    if single_match:
        numeric_value = float(single_match.group(1))
        unit = single_match.group(2)
        if unit.startswith('hour') or unit.startswith('hr'):
            return numeric_value * 60.0
        return numeric_value

    return float('nan')


def parse_numeric(value: object) -> float:
    text = clean_text(value).replace(',', '')
    text = text.replace('$', '')
    if not text:
        return float('nan')
    try:
        return float(text)
    except ValueError:
        return float('nan')


def min_max_scale(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def build_rarity_scores(token_series: Iterable[List[str]]) -> pd.Series:
    token_frequency: Dict[str, int] = {}
    token_series = list(token_series)
    for tokens in token_series:
        for token in tokens:
            token_frequency[token] = token_frequency.get(token, 0) + 1

    rarity_scores: List[float] = []
    for tokens in token_series:
        if not tokens:
            rarity_scores.append(0.0)
            continue
        rarity_scores.append(sum(1.0 / token_frequency[token] for token in tokens) / len(tokens))

    return pd.Series(rarity_scores)


def keyword_hit_score(text: str, keywords: Iterable[str]) -> float:
    if not text:
        return 0.0
    return float(sum(1 for keyword in keywords if keyword in text))


def build_embedding_text(row: pd.Series) -> str:
    chunks = [
        f"title {clean_text(row.get('Title', ''))}",
        f"type {clean_text(row.get('Series or Movie', ''))}",
        f"genre {row.get('genre_text', '')}",
        f"tags {row.get('tag_text', '')}",
        f"languages {row.get('language_text', '')}",
        f"summary {clean_text(row.get('Summary', ''))}",
        f"actors {clean_text(row.get('Actors', ''))}",
        f"director {clean_text(row.get('Director', ''))}",
        f"writer {clean_text(row.get('Writer', ''))}",
    ]
    return '. '.join(chunk for chunk in chunks if chunk.strip()).strip()


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ''
    return df


def build_train_ready_feature_store(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_required_columns(raw_df.copy())
    df = df[REQUIRED_COLUMNS].copy()

    for column in ['Title', 'Summary', 'Director', 'Writer', 'Actors', 'Series or Movie', 'View Rating', 'Image']:
        df[column] = df[column].apply(clean_text)

    df = df[df['Title'] != ''].copy()
    df = df[~df['Title'].duplicated()].reset_index(drop=True)
    df['title_key'] = df['Title'].apply(normalize_title_key)

    df['genre_tokens'] = df['Genre'].apply(split_and_canonicalize)
    df['tag_tokens'] = df['Tags'].apply(split_and_canonicalize)
    df['language_tokens'] = df['Languages'].apply(split_and_canonicalize)
    df['country_tokens'] = df['Country Availability'].apply(split_and_canonicalize)

    df['genre_text'] = df['genre_tokens'].apply(lambda tokens: ', '.join(tokens))
    df['tag_text'] = df['tag_tokens'].apply(lambda tokens: ', '.join(tokens))
    df['language_text'] = df['language_tokens'].apply(lambda tokens: ', '.join(tokens))

    df['runtime_minutes'] = df['Runtime'].apply(parse_runtime_minutes)
    df['runtime_minutes'] = df['runtime_minutes'].fillna(df['runtime_minutes'].median())

    release_dates = pd.to_datetime(
        df['Release Date'].apply(clean_text),
        errors='coerce',
        dayfirst=True,
        format='mixed',
    )
    netflix_dates = pd.to_datetime(
        df['Netflix Release Date'].apply(clean_text),
        errors='coerce',
        dayfirst=True,
        format='mixed',
    )

    df['release_date'] = release_dates
    df['netflix_release_date'] = netflix_dates
    df['release_year'] = df['release_date'].dt.year.fillna(0).astype(int)
    df['netflix_release_year'] = df['netflix_release_date'].dt.year.fillna(0).astype(int)
    df['days_to_netflix'] = (df['netflix_release_date'] - df['release_date']).dt.days.fillna(0)

    df['imdb_score'] = df['IMDb Score'].apply(parse_numeric).fillna(6.6)
    df['awards_received'] = df['Awards Received'].apply(parse_numeric).fillna(0.0)
    df['awards_nominated_for'] = df['Awards Nominated For'].apply(parse_numeric).fillna(0.0)
    df['awards_total'] = df['awards_received'] + df['awards_nominated_for']
    df['imdb_votes'] = df['IMDb Votes'].apply(parse_numeric).fillna(0.0)
    df['imdb_votes_log'] = np.log1p(df['imdb_votes'])

    df['summary_word_count'] = df['Summary'].str.split().str.len().fillna(0)

    imdb_norm = min_max_scale(df['imdb_score'])
    awards_norm = min_max_scale(df['awards_total'])
    summary_norm = min_max_scale(df['summary_word_count'])
    df['depth_score'] = 0.50 * imdb_norm + 0.30 * awards_norm + 0.20 * summary_norm

    genre_rarity = build_rarity_scores(df['genre_tokens'])
    tag_rarity = build_rarity_scores(df['tag_tokens'])
    rarity_norm = min_max_scale(0.60 * genre_rarity + 0.40 * tag_rarity)
    votes_norm = min_max_scale(df['imdb_votes'])
    df['votes_norm'] = votes_norm
    df['uniqueness_score'] = 0.65 * rarity_norm + 0.35 * (1 - votes_norm)

    text_blob = (df['genre_text'] + ' ' + df['tag_text'] + ' ' + df['Summary'].str.lower()).fillna('')
    intellectual_norm = min_max_scale(text_blob.apply(lambda text: keyword_hit_score(text, INTELLECTUAL_KEYWORDS)))
    action_norm = min_max_scale(text_blob.apply(lambda text: keyword_hit_score(text, ACTION_KEYWORDS)))
    emotional_norm = min_max_scale(text_blob.apply(lambda text: keyword_hit_score(text, EMOTIONAL_KEYWORDS)))
    df['intellectual_tone_score'] = 0.7 * intellectual_norm + 0.3 * emotional_norm
    df['action_tone_score'] = action_norm

    df['series_or_movie'] = df['Series or Movie'].str.lower().str.strip()
    df['language_set'] = df['language_tokens'].apply(set)

    df['embedding_text'] = df.apply(build_embedding_text, axis=1)

    df['genre_tokens_json'] = df['genre_tokens'].apply(json.dumps)
    df['tag_tokens_json'] = df['tag_tokens'].apply(json.dumps)
    df['language_tokens_json'] = df['language_tokens'].apply(json.dumps)
    df['country_tokens_json'] = df['country_tokens'].apply(json.dumps)

    return df


def save_feature_store(df: pd.DataFrame, output_path: str, csv_fallback_path: str | None = None) -> str:
    serializable_df = df.copy()
    for column in serializable_df.columns:
        non_null_values = serializable_df[column].dropna()
        if non_null_values.empty:
            continue
        sample_value = non_null_values.iloc[0]
        if isinstance(sample_value, set):
            serializable_df[column] = serializable_df[column].apply(
                lambda value: json.dumps(sorted(value)) if isinstance(value, set) else value
            )
        elif isinstance(sample_value, (list, dict)):
            serializable_df[column] = serializable_df[column].apply(
                lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == '.parquet':
        try:
            serializable_df.to_parquet(output, index=False)
            return str(output)
        except Exception:
            pass

    if output.suffix.lower() == '.csv':
        serializable_df.to_csv(output, index=False)
        return str(output)

    if csv_fallback_path:
        fallback = Path(csv_fallback_path)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        serializable_df.to_csv(fallback, index=False)
        return str(fallback)

    fallback = output.with_suffix('.csv')
    serializable_df.to_csv(fallback, index=False)
    return str(fallback)


def build_feature_store_from_csv(csv_path: str, output_path: str, csv_fallback_path: str | None = None) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path, encoding='latin-1')
    feature_df = build_train_ready_feature_store(raw_df)
    saved_path = save_feature_store(feature_df, output_path=output_path, csv_fallback_path=csv_fallback_path)
    print(f'Feature store rows: {len(feature_df)}')
    print(f'Feature store saved to: {saved_path}')
    return feature_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a train-ready title feature store.')
    parser.add_argument('--csv-path', default='NetflixDataset.csv', help='Path to source CSV dataset.')
    parser.add_argument(
        '--output-path',
        default='artifacts/title_feature_store.parquet',
        help='Path where the feature store should be written.',
    )
    parser.add_argument(
        '--csv-fallback-path',
        default='artifacts/title_feature_store.csv',
        help='CSV path to use when parquet writer dependencies are unavailable.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_feature_store_from_csv(
        csv_path=args.csv_path,
        output_path=args.output_path,
        csv_fallback_path=args.csv_fallback_path,
    )


if __name__ == '__main__':
    main()