from statistics import mean

from sklearn.metrics.pairwise import cosine_similarity

from recommender import NetflixRecommender, normalize_text


PROFILES = [
    {
        'name': 'Intellectual Sci-Fi',
        'seeds': ['Interstellar', 'The Umbrella Academy', 'The Magicians', 'The Godfather'],
        'languages': ['English'],
        'prefer': [
            'philosoph', 'existential', 'dystopian', 'mind', 'consciousness',
            'identity', 'character', 'drama', 'mystery', 'space', 'slow-burn'
        ],
        'avoid': ['superhero', 'explosive', 'action', 'monster', 'mecha', 'spectacle'],
    },
    {
        'name': 'Crime Drama Depth',
        'seeds': ['The Godfather', 'Pulp Fiction', 'No Country for Old Men', 'In Bruges'],
        'languages': ['English'],
        'prefer': ['crime', 'drama', 'character', 'moral', 'dark', 'dialogue'],
        'avoid': ['family comedy', 'kids', 'superhero', 'animated'],
    },
]


def keyword_alignment_score(df, prefer_keywords, avoid_keywords):
    if df.empty:
        return 0.0

    scores = []
    for row in df.itertuples(index=False):
        text = normalize_text(getattr(row, 'Genre', '')) + ' ' + normalize_text(getattr(row, 'Tags', ''))
        text += ' ' + normalize_text(getattr(row, 'Summary', ''))

        prefer_hits = sum(1 for keyword in prefer_keywords if keyword in text)
        avoid_hits = sum(1 for keyword in avoid_keywords if keyword in text)
        raw = prefer_hits - avoid_hits
        normalized = max(-3.0, min(3.0, float(raw)))
        scores.append((normalized + 3.0) / 6.0)

    return mean(scores)


def diversity_score(recommender, title_list):
    if len(title_list) < 2:
        return 1.0

    idx_map = recommender.model_data.set_index('Title').index
    indices = []
    for title in title_list:
        matches = idx_map[idx_map == title]
        if len(matches) == 0:
            continue
        indices.append(recommender.model_data[recommender.model_data['Title'] == title].index[0])

    if len(indices) < 2:
        return 1.0

    similarity_matrix = cosine_similarity(recommender.tfidf_matrix[indices], recommender.tfidf_matrix[indices])

    similarities = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            similarities.append(float(similarity_matrix[i, j]))

    if not similarities:
        return 1.0

    return 1.0 - mean(similarities)


def action_leakage_rate(recommender, title_list):
    if not title_list:
        return 0.0

    rows = recommender.model_data[recommender.model_data['Title'].isin(title_list)]
    if rows.empty:
        return 0.0

    leaked = rows[
        (rows['action_tone_score'] > rows['intellectual_tone_score'] + 0.15)
        & (rows['depth_score'] < 0.56)
    ]
    return float(len(leaked) / len(rows))


def precision_proxy(metrics):
    return (
        0.42 * metrics['alignment']
        + 0.24 * metrics['depth']
        + 0.18 * metrics['intellectual']
        + 0.10 * (1.0 - metrics['action'])
        + 0.06 * (1.0 - metrics['leakage'])
    )


def evaluate_profile(recommender, profile):
    recommendations = recommender.recommend(
        seed_titles=profile['seeds'],
        selected_languages=profile['languages'],
        top_k=10,
    )

    if recommendations.empty:
        return {
            'name': profile['name'],
            'count': 0,
            'alignment': 0.0,
            'depth': 0.0,
            'uniqueness': 0.0,
            'intellectual': 0.0,
            'action': 0.0,
            'diversity': 0.0,
            'leakage': 0.0,
            'precision': 0.0,
        }, recommendations

    details = recommender.display_data.merge(recommendations[['Title']], on='Title', how='right')
    alignment = keyword_alignment_score(details, profile['prefer'], profile['avoid'])
    profile_metrics = recommender.profile_metrics(recommendations)
    diversity = diversity_score(recommender, recommendations['Title'].tolist())
    leakage = action_leakage_rate(recommender, recommendations['Title'].tolist())

    result = {
        'name': profile['name'],
        'count': len(recommendations),
        'alignment': alignment,
        'depth': profile_metrics['avg_depth'],
        'uniqueness': profile_metrics['avg_uniqueness'],
        'intellectual': profile_metrics['avg_intellectual_tone'],
        'action': profile_metrics['avg_action_tone'],
        'diversity': diversity,
        'leakage': leakage,
    }
    result['precision'] = precision_proxy(result)
    return result, recommendations


def main():
    recommender = NetflixRecommender('NetflixDataset.csv')

    print('Offline recommendation quality report')
    print('-' * 95)
    print(
        f"{'Profile':<24} {'N':<3} {'Align':<8} {'Depth':<8} {'Unique':<8} "
        f"{'Intel':<8} {'Action':<8} {'Leak':<8} {'Div':<7} {'Prec':<8}"
    )
    print('-' * 95)

    for profile in PROFILES:
        metrics, recommendations = evaluate_profile(recommender, profile)
        print(
            f"{metrics['name']:<24} {metrics['count']:<3} {metrics['alignment']:<8.3f} "
            f"{metrics['depth']:<8.3f} {metrics['uniqueness']:<8.3f} "
            f"{metrics['intellectual']:<8.3f} {metrics['action']:<8.3f} "
            f"{metrics['leakage']:<8.3f} {metrics['diversity']:<7.3f} {metrics['precision']:<8.3f}"
        )

        preview = recommendations['Title'].head(5).tolist()
        if preview:
            print('  Top 5:', ' | '.join(preview))
        else:
            print('  Top 5: (none)')

    print('-' * 95)


if __name__ == '__main__':
    main()
