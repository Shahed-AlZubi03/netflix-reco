<h1 align="center">Netflix-Recommendation-System</h1>
<p><font size="3">
A web-app which can be used to get recommendations for a series/movie, the app recommends a list of media according to list of entered choices of movies/series in your preferred language using <strong>Python</strong> and <strong>Flask</strong> for backend and <strong>HTML</strong>, <strong>CSS</strong> and <strong>JavaScript</strong> for frontend.
</p>

 # This web-app contains 3 main pages:
- [Home Page](#home-page)
- [Recommendation Page](#recommendation-page)
- [Movie Detail Page](#movie-detail-page)
- [Netflix Page](#netflix-page)

## Home Page
Here the user can choose list of their favourite movies and series and their preferred language. For example, I have entered a list with 2 Horror Movies(Insidious and Insidious Chapter 2), an action series(Supergirl) and a drama series(Suits) as my list of choices and English and Hindi as my preferred languages.
Clicking on the Get Started button the user will see the list of recommendations.
![](/app/static/screenshots/Screenshot-HomePage.png)

## Recommendation Page
Here the user will get poster images of all the recommended movies and series sorted based upon their IMDb Scores.
![](/app/static/screenshots/Screenshot-RecommendationPage1.png)
![](/app/static/screenshots/Screenshot-RecommendationPage2.png)

Clicking on any poster image, the user will be sent to the Movie Details page for the corresponding title.

## Movie Detail Page
Here are the complete details of the user selected title like Genre, Movie Summary, Languages in which movie is available, IMDb scores, Directors, Writers and Actors and so on. User will also find a link at the end of the page for the NEtflix Page of the corresponding title. 
![](/app/static/screenshots/Screenshot-MovieDetailPage1.png)
![](/app/static/screenshots/Screenshot-MovieDetailPage2.png)

## Netflix Page
This page is not a part of my web-app but an example what the user will see as the Netflix Page if they choose to click on the Netflix Link for the title.
You can login into your Netflix account and enjoy watching your selected movie or series from our recommendations.
![](/app/static/screenshots/Screenshot-NetflixPage.png)

# How To Use

To be able to use this web app locally in a development environment you will need the following:

1) You will need [Git](https://git-scm.com) installed on your computer.

2) Then From your terminal, you should do the following:

```cmd
# Clone this repository
git clone https://github.com/garg-priya-creator/Netflix-Recommendation-System.git

# Go into the repository
cd netflix-recommendation-system

# Install flask (if you already haven't)
pip install flask

# Install project dependencies (recommended)
pip install -r app/requirements.txt

```
3) To run this application you don't need to have any special configuration but make sure you don't change the directory of the project otherwise you can recieve errors while you try to run the app.

4) You can run the Netflix React App using the following command from your terminal:

```
# Run the app
>>set FLASK_APP=app.py
>>flask run
```

## Recommendation Quality Workflow

The project now includes an offline quality harness and unit tests so you can iterate on recommendation logic with measurable signals.

From the `app` folder:

```bash
# use one virtual environment only (project-local)
source .venv/bin/activate

# run offline evaluation profiles and quality metrics
python evaluate_recommender.py

# run recommender unit tests
python -m unittest tests/test_recommender.py -v
```

The evaluation script reports profile-level alignment, depth, uniqueness, intellectual/action tone, and diversity.

Important environment note:
- Keep only `Netflix-Recommendation-System-main/app/.venv`.
- If you have another `.venv` at repository root, remove it to avoid interpreter confusion.

## ML Transition (Phase 0 + Phase 1)

The project now includes a transition path from TF-IDF retrieval to semantic retrieval with sentence embeddings.

### Phase 0: Data Hardening + Train-Ready Feature Store

From the `app` folder:

```bash
# Build canonicalized, train-ready per-title features
python ml_feature_pipeline.py \
	--csv-path NetflixDataset.csv \
	--output-path artifacts/title_feature_store.parquet \
	--csv-fallback-path artifacts/title_feature_store.csv
```

This pipeline does the following:
- Normalizes text and attempts to repair encoding artifacts.
- Canonicalizes genres/tags/languages into stable tokens.
- Parses runtime and date fields into numeric features.
- Produces a reusable feature store for training and retrieval.

### Phase 1: Semantic Retrieval + Baseline Validation

Embedding model selected for this repo:
- `sentence-transformers/all-MiniLM-L6-v2`

Why this model:
- Strong semantic quality for short media metadata.
- Lightweight and practical for local development.
- Fast inference for catalogs of this size.

Run transition evaluation against TF-IDF baseline:

```bash
python evaluate_semantic_transition.py
```

This compares TF-IDF and semantic retrieval on the same profile metrics and prints precision deltas.

### Phase 2: LambdaMART Re-Ranking

Phase 2 adds:
- Listwise training sample generation from semantic candidate pools.
- LambdaMART training (`LightGBM LGBMRanker`) on handcrafted + embedding features.
- MMR diversity rerank in final top-k after model scoring.
- Offline profile-bucket evaluation against TF-IDF and semantic baselines.

Train Phase 2 model:

```bash
python train_phase2_lambdamart.py \
	--csv-path NetflixDataset.csv \
	--feature-store-path artifacts/title_feature_store.parquet \
	--model-artifact-path artifacts/lambdamart_ranker.joblib \
	--training-data-path artifacts/lambdamart_training_samples.parquet \
	--candidate-pool-limit 180 \
	--max-queries 2000
```

Run 3-way offline evaluation by profile buckets:

```bash
python evaluate_phase2_transition.py
```

Use Phase 2 in the Flask app:

```bash
export RECOMMENDER_BACKEND=phase2
flask run
```

### Precision vs Diversity Knob

The recommender now applies intent-aware defaults for MMR re-ranking and also supports manual override via code:

- Higher `mmr_lambda` -> higher precision, lower diversity
- Lower `mmr_lambda` -> higher diversity, lower precision

You can tune this in the call to `recommend(...)` in `app/app.py`.

# Author

👤 **Priya Garg**
- Github: https://github.com/garg-priya-creator
- Linkedin: https://www.linkedin.com/in/priya-garg-9220381b3
- Email: priyagarg072@gmail.com

# Show Your Support 

Give a ⭐️ if you like this project!
