import argparse

from phase2_lambdamart import LambdaMARTPhase2Recommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Phase 2 LambdaMART reranker.')
    parser.add_argument('--csv-path', default='NetflixDataset.csv', help='Path to source dataset CSV.')
    parser.add_argument(
        '--feature-store-path',
        default='artifacts/title_feature_store.parquet',
        help='Path to feature store file.',
    )
    parser.add_argument(
        '--model-artifact-path',
        default='artifacts/lambdamart_ranker.joblib',
        help='Path to save trained LambdaMART model artifact.',
    )
    parser.add_argument(
        '--embeddings-cache-path',
        default='artifacts/title_embeddings.npy',
        help='Path to save or load the precomputed embedding matrix.',
    )
    parser.add_argument(
        '--training-data-path',
        default='artifacts/lambdamart_training_samples.parquet',
        help='Path to save generated listwise training samples.',
    )
    parser.add_argument(
        '--candidate-pool-limit',
        type=int,
        default=180,
        help='Number of nearest semantic candidates sampled per query for training.',
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        default=2000,
        help='Maximum number of query titles used to generate listwise training data.',
    )
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed used for sampling.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    recommender = LambdaMARTPhase2Recommender(
        csv_path=args.csv_path,
        feature_store_path=args.feature_store_path,
        model_artifact_path=args.model_artifact_path,
        embeddings_cache_path=args.embeddings_cache_path,
        allow_embedding_rebuild=True,
        auto_train_if_missing=False,
    )

    summary = recommender.train_ranker(
        training_data_path=args.training_data_path,
        model_artifact_path=args.model_artifact_path,
        candidate_pool_limit=args.candidate_pool_limit,
        max_queries=args.max_queries,
        random_seed=args.random_seed,
    )

    print('Phase 2 LambdaMART training complete')
    print(f"training_rows: {int(summary['training_rows'])}")
    print(f"training_queries: {int(summary['training_queries'])}")
    print(f"train_rows: {int(summary['train_rows'])}")
    print(f"valid_rows: {int(summary['valid_rows'])}")
    print(f'model_artifact: {args.model_artifact_path}')
    print(f'embeddings_cache: {args.embeddings_cache_path}')


if __name__ == '__main__':
    main()