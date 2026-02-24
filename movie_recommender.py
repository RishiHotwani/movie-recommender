"""
Movie Recommendation System
============================
Implements three recommendation approaches:
1. Content-Based Filtering (using TF-IDF on genres/descriptions)
2. Collaborative Filtering (User-Item Matrix with SVD)
3. Hybrid Recommender (combines both)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. SAMPLE DATASET
# ─────────────────────────────────────────────

MOVIES = pd.DataFrame({
    'movie_id': range(1, 21),
    'title': [
        'The Dark Knight', 'Inception', 'Interstellar', 'The Matrix',
        'Avengers: Endgame', 'Iron Man', 'The Godfather', 'Pulp Fiction',
        'Forrest Gump', 'The Shawshank Redemption', 'Titanic', 'Avatar',
        'Toy Story', 'Finding Nemo', 'The Lion King', 'Gladiator',
        'Braveheart', 'Schindler\'s List', 'Goodfellas', 'Fight Club'
    ],
    'genres': [
        'Action Crime Drama', 'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi', 'Action Sci-Fi',
        'Action Adventure Sci-Fi', 'Action Sci-Fi', 'Crime Drama', 'Crime Drama Thriller',
        'Drama Romance', 'Drama', 'Drama Romance', 'Action Adventure Sci-Fi',
        'Animation Adventure Comedy', 'Animation Adventure Comedy', 'Animation Adventure Drama',
        'Action Adventure Drama', 'Action Drama History', 'Biography Drama History',
        'Biography Crime Drama', 'Drama Thriller'
    ],
    'description': [
        'Batman fights the Joker in Gotham. Crime and chaos unleashed.',
        'A thief who steals corporate secrets through dream-sharing technology.',
        'A team of explorers travel through a wormhole in space.',
        'A hacker discovers the world is a simulated reality.',
        'The Avengers assemble to undo Thanos universe destruction.',
        'Billionaire Tony Stark builds a powered armor suit.',
        'The story of the Corleone mafia family power and loyalty.',
        'Several interwoven stories of Los Angeles criminals.',
        'A man with low IQ witnesses major historical events.',
        'Two imprisoned men bond over years finding redemption.',
        'A love story aboard the ill-fated RMS Titanic ship.',
        'A paraplegic marine on an alien moon ecological mission.',
        'A cowboy toy is threatened by a new spaceman toy.',
        'A clownfish searches the ocean for his missing son.',
        'A young lion prince exiled after his father is murdered.',
        'A Roman general seeks revenge against a corrupt emperor.',
        'Scottish warrior William Wallace leads revolt against English rule.',
        'A German businessman saves lives of Jewish workers in WWII.',
        'Henry Hill rises through the ranks of organized crime.',
        'An insomniac forms an underground fight club with a soap maker.'
    ],
    'year': [2008,2010,2014,1999,2019,2008,1972,1994,1994,1994,1997,2009,1995,2003,1994,2000,1995,1993,1990,1999],
    'rating': [9.0,8.8,8.6,8.7,8.4,7.9,9.2,8.9,8.8,9.3,7.8,7.9,8.3,8.1,8.5,8.5,8.3,9.0,8.7,8.8]
})

RATINGS = pd.DataFrame({
    'user_id': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5,
                6,6,6,6,6, 7,7,7,7,7, 8,8,8,8,8],
    'movie_id': [1,2,3,4,5, 1,6,7,8,9, 2,3,10,11,12, 7,8,9,13,14, 1,2,4,5,6,
                 15,13,14,9,10, 3,4,5,16,17, 18,19,20,7,8],
    'rating':   [5,5,4,5,4, 4,5,5,4,3, 5,4,5,3,4, 5,4,3,4,4, 5,4,5,4,5,
                 4,5,4,3,4, 5,4,4,4,3, 5,5,4,5,4]
})

USERS = pd.DataFrame({
    'user_id': range(1, 9),
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'favorite_genres': [
        'Action Sci-Fi', 'Crime Drama', 'Sci-Fi Drama', 'Animation Comedy',
        'Action Adventure', 'Animation Drama', 'Sci-Fi Adventure', 'Crime Drama Thriller'
    ]
})


# ─────────────────────────────────────────────
# 2. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────

class ContentBasedRecommender:
    """Recommends movies similar to a given movie using TF-IDF + Cosine Similarity."""

    def __init__(self, movies_df: pd.DataFrame):
        self.movies = movies_df.copy()
        self.similarity_matrix = None
        self._build_model()

    def _build_model(self):
        # Combine genres + description into one feature string
        self.movies['features'] = (
            self.movies['genres'] + ' ' + self.movies['description']
        )
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(self.movies['features'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        print("✅ Content-Based model built (TF-IDF + Cosine Similarity)")

    def recommend(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        """Return top_n movies most similar to movie_title."""
        matches = self.movies[self.movies['title'].str.lower() == movie_title.lower()]
        if matches.empty:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

        movie_indices = [s[0] for s in sim_scores]
        scores       = [round(s[1], 4) for s in sim_scores]

        result = self.movies.iloc[movie_indices][['title', 'genres', 'year', 'rating']].copy()
        result['similarity_score'] = scores
        result = result.reset_index(drop=True)
        result.index += 1
        return result


# ─────────────────────────────────────────────
# 3. COLLABORATIVE FILTERING (SVD)
# ─────────────────────────────────────────────

class CollaborativeRecommender:
    """User-Item matrix factorization using TruncatedSVD."""

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies  = movies_df.copy()
        self.ratings = ratings_df.copy()
        self.user_movie_matrix = None
        self.predicted_ratings = None
        self._build_model()

    def _build_model(self):
        # Pivot to User × Movie matrix (fill missing with 0)
        self.user_movie_matrix = self.ratings.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        ).fillna(0)

        # SVD decomposition
        n_components = min(5, min(self.user_movie_matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        U   = svd.fit_transform(self.user_movie_matrix)
        Vt  = svd.components_

        self.predicted_ratings = pd.DataFrame(
            np.dot(U, Vt),
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.columns
        )
        print(f"✅ Collaborative Filtering model built (SVD, {n_components} components)")

    def recommend(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """Recommend movies the user hasn't rated yet."""
        if user_id not in self.user_movie_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        already_rated = set(
            self.ratings[self.ratings['user_id'] == user_id]['movie_id']
        )
        user_preds = self.predicted_ratings.loc[user_id]
        unrated = user_preds[~user_preds.index.isin(already_rated)]
        top_movie_ids = unrated.nlargest(top_n).index.tolist()

        result = self.movies[self.movies['movie_id'].isin(top_movie_ids)][
            ['movie_id', 'title', 'genres', 'year', 'rating']
        ].copy()
        pred_scores = {mid: round(unrated[mid], 4) for mid in top_movie_ids}
        result['predicted_rating'] = result['movie_id'].map(pred_scores)
        result = result.sort_values('predicted_rating', ascending=False).drop('movie_id', axis=1)
        result = result.reset_index(drop=True)
        result.index += 1
        return result


# ─────────────────────────────────────────────
# 4. HYBRID RECOMMENDER
# ─────────────────────────────────────────────

class HybridRecommender:
    """Combines Content-Based and Collaborative scores with configurable weights."""

    def __init__(self, content_rec: ContentBasedRecommender,
                 collab_rec: CollaborativeRecommender,
                 movies_df: pd.DataFrame):
        self.content = content_rec
        self.collab  = collab_rec
        self.movies  = movies_df.copy()
        print("✅ Hybrid Recommender ready")

    def recommend(self, user_id: int, movie_title: str,
                  top_n: int = 5, alpha: float = 0.5) -> pd.DataFrame:
        """
        alpha = weight for collaborative score (1-alpha = content weight).
        Merges both ranked lists and scores.
        """
        content_recs = self.content.recommend(movie_title, top_n=10)
        collab_recs  = self.collab.recommend(user_id,      top_n=10)

        # Normalize scores to [0, 1]
        scaler = MinMaxScaler()

        content_recs = content_recs.copy()
        collab_recs  = collab_recs.copy()

        content_recs['norm_score'] = scaler.fit_transform(
            content_recs[['similarity_score']]
        )
        collab_recs['norm_score'] = scaler.fit_transform(
            collab_recs[['predicted_rating']]
        )

        # Merge on title
        merged = pd.merge(
            content_recs[['title', 'genres', 'year', 'rating', 'norm_score']].rename(columns={'norm_score': 'content_score'}),
            collab_recs[['title', 'norm_score']].rename(columns={'norm_score': 'collab_score'}),
            on='title', how='outer'
        ).fillna(0)

        merged['hybrid_score'] = (
            alpha * merged['collab_score'] + (1 - alpha) * merged['content_score']
        ).round(4)

        result = merged.sort_values('hybrid_score', ascending=False).head(top_n)
        result = result.reset_index(drop=True)
        result.index += 1
        return result[['title', 'genres', 'year', 'rating', 'hybrid_score']]


# ─────────────────────────────────────────────
# 5. EVALUATION METRICS
# ─────────────────────────────────────────────

def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top_k = recommended_ids[:k]
    hits  = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / k

def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of relevant items captured in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits  = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / len(relevant_ids)

def evaluate_collaborative(collab_rec: CollaborativeRecommender,
                            ratings_df: pd.DataFrame, k: int = 5) -> dict:
    """Leave-one-out evaluation across all users."""
    precisions, recalls = [], []

    for uid in ratings_df['user_id'].unique():
        user_ratings = ratings_df[ratings_df['user_id'] == uid]
        if len(user_ratings) < 2:
            continue
        # Relevant = movies rated ≥ 4
        relevant = set(user_ratings[user_ratings['rating'] >= 4]['movie_id'])
        if not relevant:
            continue
        try:
            recs = collab_rec.recommend(uid, top_n=k)
            rec_titles = recs['title'].tolist()
            rec_ids = collab_rec.movies[
                collab_rec.movies['title'].isin(rec_titles)
            ]['movie_id'].tolist()
            precisions.append(precision_at_k(rec_ids, relevant, k))
            recalls.append(recall_at_k(rec_ids, relevant, k))
        except Exception:
            continue

    return {
        'Precision@K': round(np.mean(precisions), 4) if precisions else 0,
        'Recall@K':    round(np.mean(recalls),    4) if recalls    else 0,
        'K':           k
    }


# ─────────────────────────────────────────────
# 6. DEMO
# ─────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print('═'*55)

def run_demo():
    print("\n🎬  MOVIE RECOMMENDATION SYSTEM  🎬")
    print("Building models …\n")

    content_rec = ContentBasedRecommender(MOVIES)
    collab_rec  = CollaborativeRecommender(MOVIES, RATINGS)
    hybrid_rec  = HybridRecommender(content_rec, collab_rec, MOVIES)

    # ── Content-Based ──────────────────────────────
    print_section("1. CONTENT-BASED FILTERING")
    query_movie = 'Inception'
    print(f"\n📽  Movies similar to '{query_movie}':\n")
    cb_recs = content_rec.recommend(query_movie, top_n=5)
    print(cb_recs.to_string())

    # ── Collaborative ──────────────────────────────
    print_section("2. COLLABORATIVE FILTERING (SVD)")
    query_user = 3
    user_name  = USERS[USERS['user_id'] == query_user]['name'].values[0]
    print(f"\n👤  Recommendations for User {query_user} ({user_name}):\n")
    cf_recs = collab_rec.recommend(query_user, top_n=5)
    print(cf_recs.to_string())

    # ── Hybrid ─────────────────────────────────────
    print_section("3. HYBRID RECOMMENDER (alpha=0.5)")
    print(f"\n🤝  Hybrid recs for User {query_user} ({user_name}), based on '{query_movie}':\n")
    hy_recs = hybrid_rec.recommend(query_user, query_movie, top_n=5, alpha=0.5)
    print(hy_recs.to_string())

    # ── Evaluation ─────────────────────────────────
    print_section("4. EVALUATION METRICS")
    metrics = evaluate_collaborative(collab_rec, RATINGS, k=5)
    print(f"\n  Collaborative Filtering @ K={metrics['K']}")
    print(f"  Precision@{metrics['K']}: {metrics['Precision@K']}")
    print(f"  Recall@{metrics['K']}:    {metrics['Recall@K']}")

    # ── Dataset Summary ────────────────────────────
    print_section("5. DATASET SUMMARY")
    print(f"\n  📊 Movies   : {len(MOVIES)}")
    print(f"  👥 Users    : {len(USERS)}")
    print(f"  ⭐ Ratings  : {len(RATINGS)}")
    print(f"  🎭 Genres   : {', '.join(sorted(set(' '.join(MOVIES['genres']).split())))}")

    print("\n" + "═"*55)
    print("  ✅ Demo Complete!")
    print("═"*55 + "\n")

    return content_rec, collab_rec, hybrid_rec


if __name__ == '__main__':
    run_demo()
