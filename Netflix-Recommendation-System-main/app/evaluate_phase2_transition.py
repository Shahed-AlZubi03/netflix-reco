from statistics import mean

from sklearn.metrics.pairwise import cosine_similarity

from evaluate_recommender import (
    PROFILES,
    action_leakage_rate,
    keyword_alignment_score,
    precision_proxy,
)
from phase2_lambdamart import LambdaMARTPhase2Recommender
from recommender import NetflixRecommender
from semantic_recommender import SemanticNetflixRecommender


def diversity_score_generic(recommender, title_list):
    if len(title_list) < 2:
        return 1.0

    rows = recommender.model_data[recommender.model_data['Title'].isin(title_list)]
    if rows.empty:
        return 1.0

    indices = rows.index.tolist()
    if len(indices) < 2:
        return 1.0

    if hasattr(recommender, 'embedding_matrix'):
        similarity_matrix = cosine_similarity(
            recommender.embedding_matrix[indices],
            recommender.embedding_matrix[indices],
        )
    elif hasattr(recommender, 'tfidf_matrix'):
        similarity_matrix = cosine_similarity(
            recommender.tfidf_matrix[indices],
            recommender.tfidf_matrix[indices],
        )
    else:
        return 1.0

    similarities = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            similarities.append(float(similarity_matrix[i, j]))

    if not similarities:
        return 1.0

    return 1.0 - mean(similarities)


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
            'top5': [],
        }

    details = recommender.display_data.merge(recommendations[['Title']], on='Title', how='right')
    alignment = keyword_alignment_score(details, profile['prefer'], profile['avoid'])
    profile_metrics = recommender.profile_metrics(recommendations)
    diversity = diversity_score_generic(recommender, recommendations['Title'].tolist())
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
        'top5': recommendations['Title'].head(5).tolist(),
    }
    result['precision'] = precision_proxy(result)
    return result


def print_metrics_row(model_name, metrics):
    print(
        f"{model_name:<12} {metrics['count']:<3} {metrics['alignment']:<8.3f} "
        f"{metrics['depth']:<8.3f} {metrics['uniqueness']:<8.3f} "
        f"{metrics['intellectual']:<8.3f} {metrics['action']:<8.3f} "
        f"{metrics['leakage']:<8.3f} {metrics['diversity']:<7.3f} {metrics['precision']:<8.3f}"
    )


def main():
    baseline = NetflixRecommender('NetflixDataset.csv')
    semantic = SemanticNetflixRecommender(
        csv_path='NetflixDataset.csv',
        feature_store_path='artifacts/title_feature_store.parquet',
    )
    phase2 = LambdaMARTPhase2Recommender(
        csv_path='NetflixDataset.csv',
        feature_store_path='artifacts/title_feature_store.parquet',
        model_artifact_path='artifacts/lambdamart_ranker.joblib',
        embeddings_cache_path='artifacts/title_embeddings.npy',
        allow_embedding_rebuild=True,
        auto_train_if_missing=True,
    )

    print('TF-IDF vs Semantic vs Phase2-LambdaMART Report')
    print('-' * 120)
    header = (
        f"{'Model':<12} {'N':<3} {'Align':<8} {'Depth':<8} {'Unique':<8} "
        f"{'Intel':<8} {'Action':<8} {'Leak':<8} {'Div':<7} {'Prec':<8}"
    )

    for profile in PROFILES:
        print(profile['name'])
        print(header)
        print('-' * 120)

        baseline_metrics = evaluate_profile(baseline, profile)
        semantic_metrics = evaluate_profile(semantic, profile)
        phase2_metrics = evaluate_profile(phase2, profile)

        print_metrics_row('TF-IDF', baseline_metrics)
        print_metrics_row('Semantic', semantic_metrics)
        print_metrics_row('Phase2', phase2_metrics)

        print(f"Precision delta (Semantic - TF-IDF): {semantic_metrics['precision'] - baseline_metrics['precision']:+.3f}")
        print(f"Precision delta (Phase2 - TF-IDF): {phase2_metrics['precision'] - baseline_metrics['precision']:+.3f}")
        print(f"Precision delta (Phase2 - Semantic): {phase2_metrics['precision'] - semantic_metrics['precision']:+.3f}")

        print('TF-IDF Top 5 :', ' | '.join(baseline_metrics['top5']) if baseline_metrics['top5'] else '(none)')
        print('Semantic Top 5:', ' | '.join(semantic_metrics['top5']) if semantic_metrics['top5'] else '(none)')
        print('Phase2 Top 5  :', ' | '.join(phase2_metrics['top5']) if phase2_metrics['top5'] else '(none)')
        print('-' * 120)


if __name__ == '__main__':
    main()