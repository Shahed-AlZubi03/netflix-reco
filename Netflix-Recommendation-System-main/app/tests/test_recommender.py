import unittest

from sklearn.metrics.pairwise import cosine_similarity

from recommender import NetflixRecommender, normalize_title_key


class NetflixRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = NetflixRecommender('NetflixDataset.csv')

    def test_recommendation_shape(self):
        seeds = ['Interstellar', 'The Godfather', 'The Umbrella Academy', 'The Magicians']
        recs = self.recommender.recommend(seeds, ['English'], top_k=20)

        self.assertFalse(recs.empty)
        self.assertLessEqual(len(recs), 20)
        self.assertIn('Title', recs.columns)
        self.assertIn('Image', recs.columns)

    def test_recommendations_exclude_seed_titles(self):
        seeds = ['Interstellar', 'The Godfather', 'The Umbrella Academy', 'The Magicians']
        seed_keys = {normalize_title_key(title) for title in seeds}
        recs = self.recommender.recommend(seeds, ['English'], top_k=20)

        rec_keys = {normalize_title_key(title) for title in recs['Title'].tolist()}
        self.assertEqual(seed_keys & rec_keys, set())

    def test_language_filter_is_applied(self):
        seeds = ['Interstellar', 'The Godfather', 'The Umbrella Academy', 'The Magicians']
        recs = self.recommender.recommend(seeds, ['Japanese'], top_k=15)

        for row in recs.itertuples(index=False):
            candidate = self.recommender.model_data[self.recommender.model_data['Title'] == row.Title].iloc[0]
            self.assertIn('japanese', candidate['language_set'])

    def test_movie_details_fallback(self):
        details = self.recommender.get_movie_details('title-that-should-not-exist-123')
        self.assertEqual(len(details), 22)
        self.assertEqual(details[0], 'title-that-should-not-exist-123')

    def test_intellectual_seed_set_has_low_action_leakage(self):
        seeds = ['Interstellar', 'The Magicians', 'The Godfather', 'The Umbrella Academy']
        recs = self.recommender.recommend(seeds, ['English'], top_k=20)

        self.assertFalse(recs.empty)
        rows = self.recommender.model_data[self.recommender.model_data['Title'].isin(recs['Title'])]
        leakage_rows = rows[
            (rows['action_tone_score'] > rows['intellectual_tone_score'] + 0.15)
            & (rows['depth_score'] < 0.56)
        ]
        leakage_rate = len(leakage_rows) / len(rows)
        self.assertLessEqual(leakage_rate, 0.20)

    def test_higher_mmr_lambda_reduces_diversity(self):
        seeds = ['Interstellar', 'The Magicians', 'The Godfather', 'The Umbrella Academy']
        low_lambda_recs = self.recommender.recommend(seeds, ['English'], top_k=15, mmr_lambda=0.70)
        high_lambda_recs = self.recommender.recommend(seeds, ['English'], top_k=15, mmr_lambda=0.92)

        self.assertFalse(low_lambda_recs.empty)
        self.assertFalse(high_lambda_recs.empty)

        def avg_pairwise_similarity(titles):
            indices = self.recommender.model_data[self.recommender.model_data['Title'].isin(titles)].index.tolist()
            if len(indices) < 2:
                return 0.0
            sim = cosine_similarity(self.recommender.tfidf_matrix[indices], self.recommender.tfidf_matrix[indices])
            values = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    values.append(float(sim[i, j]))
            return sum(values) / len(values) if values else 0.0

        low_lambda_similarity = avg_pairwise_similarity(low_lambda_recs['Title'].tolist())
        high_lambda_similarity = avg_pairwise_similarity(high_lambda_recs['Title'].tolist())

        self.assertGreaterEqual(high_lambda_similarity, low_lambda_similarity)


if __name__ == '__main__':
    unittest.main()
