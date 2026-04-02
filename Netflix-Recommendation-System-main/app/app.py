import os

from flask import Flask, redirect, render_template, request, url_for

if __package__:
    from .recommender import NetflixRecommender
else:
    from recommender import NetflixRecommender


app = Flask(__name__)
backend = os.getenv('RECOMMENDER_BACKEND', 'tfidf').strip().lower()
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'NetflixDataset.csv')
feature_store_path = os.path.join(base_dir, 'artifacts', 'title_feature_store.parquet')
model_artifact_path = os.path.join(base_dir, 'artifacts', 'lambdamart_ranker.joblib')
embeddings_cache_path = os.path.join(base_dir, 'artifacts', 'title_embeddings.npy')

if backend == 'phase2':
    try:
        if __package__:
            from .phase2_lambdamart import LambdaMARTPhase2Recommender
        else:
            from phase2_lambdamart import LambdaMARTPhase2Recommender

        recommender = LambdaMARTPhase2Recommender(
            csv_path=csv_path,
            feature_store_path=feature_store_path,
            model_artifact_path=model_artifact_path,
            embeddings_cache_path=embeddings_cache_path,
            allow_embedding_rebuild=False,
            auto_train_if_missing=False,
        )
    except Exception as exc:
        print(f'[WARN] Phase2 backend failed: {exc}. Falling back to TF-IDF.')
        recommender = NetflixRecommender(csv_path)
elif backend == 'semantic':
    if __package__:
        from .semantic_recommender import SemanticNetflixRecommender
    else:
        from semantic_recommender import SemanticNetflixRecommender

    recommender = SemanticNetflixRecommender(
        csv_path=csv_path,
        feature_store_path=feature_store_path,
    )
else:
    recommender = NetflixRecommender(csv_path)


@app.route('/')
def index():
    return render_template(
        'index.html',
        languages=recommender.available_languages,
        titles=recommender.available_titles,
    )


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'GET':
        return redirect(url_for('index'))

    movienames = request.form.getlist('titles')
    selected_languages = request.form.getlist('languages')

    ranked_df = recommender.recommend(
        seed_titles=movienames,
        selected_languages=selected_languages,
        top_k=60,
    )

    images = ranked_df['Image'].tolist() if not ranked_df.empty else []
    titles = ranked_df['Title'].tolist() if not ranked_df.empty else []
    return render_template('result.html', titles=titles, images=images)


@app.route('/about', methods=['GET', 'POST'])
def getvalue():
    if request.method == 'GET':
        return redirect(url_for('index'))
    return recommendations()


@app.route('/moviepage/<name>')
def movie_details(name):
    details = recommender.get_movie_details(name)
    return render_template('moviepage.html', details=details)


if __name__ == '__main__':
    app.run(debug=False)
