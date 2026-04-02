import unittest

import pandas as pd

from ml_feature_pipeline import build_train_ready_feature_store, parse_runtime_minutes, split_and_canonicalize


class FeaturePipelineTests(unittest.TestCase):
    def test_runtime_parser_handles_bucket_values(self):
        self.assertEqual(parse_runtime_minutes('< 30 minutes'), 25.0)
        self.assertEqual(parse_runtime_minutes('1-2 hour'), 90.0)
        self.assertEqual(parse_runtime_minutes('> 2 hrs'), 150.0)

    def test_canonicalization_normalizes_common_tokens(self):
        tokens = split_and_canonicalize('TV Programmes, Sci-Fi & Fantasy, US TV Programmes')
        self.assertIn('tv shows', tokens)
        self.assertIn('sci-fi and fantasy', tokens)
        self.assertIn('us tv shows', tokens)

    def test_feature_store_contains_training_columns(self):
        sample = pd.DataFrame(
            [
                {
                    'Title': 'Sample Title',
                    'Genre': 'Drama, Fantasy',
                    'Tags': 'TV Programmes, TV Sci-Fi & Fantasy',
                    'Languages': 'English, Spanish',
                    'Country Availability': 'United States, Canada',
                    'Runtime': '< 30 minutes',
                    'Director': 'Director A',
                    'Writer': 'Writer A',
                    'Actors': 'Actor A, Actor B',
                    'View Rating': 'TV-14',
                    'IMDb Score': '7.8',
                    'Awards Received': '3',
                    'Awards Nominated For': '10',
                    'Release Date': '12-Jan-20',
                    'Netflix Release Date': '01-02-2020',
                    'Summary': 'A mysterious story about time and identity.',
                    'Series or Movie': 'Series',
                    'IMDb Votes': '12,345',
                    'Image': 'https://example.com/image.jpg',
                }
            ]
        )
        features = build_train_ready_feature_store(sample)
        self.assertEqual(len(features), 1)
        for column in [
            'title_key',
            'runtime_minutes',
            'release_year',
            'netflix_release_year',
            'imdb_score',
            'imdb_votes_log',
            'depth_score',
            'uniqueness_score',
            'embedding_text',
        ]:
            self.assertIn(column, features.columns)


if __name__ == '__main__':
    unittest.main()