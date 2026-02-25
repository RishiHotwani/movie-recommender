"""
Optimized MovieLens Recommendation Engine
-----------------------------------------
✓ Lazy TMDB Poster Fetching (fast load)
✓ Unified metadata for ALL recommenders
✓ Zero KeyErrors
✓ Fast Content-Based TF-IDF
✓ SVD Collaborative Filtering
✓ Hybrid blending model
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# 1. TMDB POSTER SYSTEM (Lazy + Cache)
# ============================================================

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE = "https://image.tmdb.org/t/p/w500"

CACHE_FILE = "poster_cache.json"

# Load poster cache
try:
    poster_cache = json.load(open(CACHE_FILE, "r", encoding="utf8")) if os.path.exists(CACHE_FILE) else {}
except:
    poster_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf8") as f:
            json.dump(poster_cache, f, indent=2)
    except:
        pass


# -------------------- Fetch Functions ----------------------

def fetch_tmdb_poster(tmdb_id):
    if pd.isna(tmdb_id) or tmdb_id == 0:
        return None
    try:
        data = requests.get(
            f"{TMDB_BASE}/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}",
            timeout=4
        ).json()
        poster = data.get("poster_path")
        return TMDB_IMAGE + poster if poster else None
    except:
        return None


def imdb_to_tmdb(imdb_id):
    if pd.isna(imdb_id):
        return None
    imdb_str = "tt" + str(int(imdb_id)).zfill(7)

    try:
        data = requests.get(
            f"{TMDB_BASE}/find/{imdb_str}?api_key={TMDB_API_KEY}&external_source=imdb_id",
            timeout=4
        ).json()
        if data.get("movie_results"):
            return data["movie_results"][0]["id"]
    except:
        return None

    return None


def search_tmdb(title, year):
    try:
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year > 0:
            params["year"] = int(year)

        data = requests.get(
            f"{TMDB_BASE}/search/movie",
            params=params,
            timeout=4
        ).json()

        if data.get("results"):
            poster = data["results"][0].get("poster_path")
            return TMDB_IMAGE + poster if poster else None
    except:
        return None

    return None


# -------------------- Master Poster Getter ----------------------

def get_poster(movie_id, title, year, imdbId, tmdbId):
    """Fetch poster lazily and cache it."""
    key = str(movie_id)

    if key in poster_cache:
        return poster_cache[key]

    # Priority 1: TMDB ID
    if tmdbId:
        p = fetch_tmdb_poster(tmdbId)
        if p:
            poster_cache[key] = p
            save_cache()
            return p

    # Priority 2: IMDb → TMDB
    if imdbId:
        new_tmdb = imdb_to_tmdb(imdbId)
        if new_tmdb:
            p = fetch_tmdb_poster(new_tmdb)
            if p:
                poster_cache[key] = p
                save_cache()
                return p

    # Priority 3: Search
    p = search_tmdb(title, year)
    poster_cache[key] = p
    save_cache()
    return p


# ============================================================
# 2. LOAD MOVIELENS DATA
# ============================================================

MOVIES = pd.read_csv("movies.csv")
RATINGS = pd.read_csv("ratings.csv")
LINKS = pd.read_csv("links.csv")[["movieId", "imdbId", "tmdbId"]]

MOVIES = MOVIES.rename(columns={"movieId": "movie_id"})
RATINGS = RATINGS.rename(columns={"userId": "user_id", "movieId": "movie_id"})

MOVIES = MOVIES[MOVIES["genres"] != "(no genres listed)"]

# extract year
MOVIES["year"] = MOVIES["title"].str.extract(r"\((\d{4})\)").fillna(0).astype(int)

# clean title
MOVIES["title"] = MOVIES["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

# clean genres
MOVIES["genres"] = MOVIES["genres"].str.replace("|", " ")

# simple description
MOVIES["description"] = MOVIES["genres"].apply(
    lambda g: f"A {g.replace(' ', ', ')} themed movie"
)

# merge links
LINKS = LINKS.rename(columns={"movieId": "movie_id"})
MOVIES = MOVIES.merge(LINKS, on="movie_id", how="left")

# average rating
rating_mean = RATINGS.groupby("movie_id")["rating"].mean()
MOVIES["rating"] = MOVIES["movie_id"].map(rating_mean).fillna(0)

# ensure poster_url column exists (empty initially)
MOVIES["poster_url"] = None

# USERS table
USERS = pd.DataFrame({"user_id": sorted(RATINGS["user_id"].unique())})
USERS["name"] = USERS["user_id"].apply(lambda x: f"User {x}")


# ============================================================
# 3. CONTENT-BASED RECOMMENDER
# ============================================================

class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df.copy()
        self._build()

    def _build(self):
        self.movies["features"] = (
            self.movies["genres"] + " " + self.movies["description"]
        )
        tfidf = TfidfVectorizer(stop_words="english")
        mat = tfidf.fit_transform(self.movies["features"])
        self.sim = cosine_similarity(mat)

    def recommend(self, title, top_n=5):
        row = self.movies[self.movies["title"].str.lower() == title.lower()]
        if row.empty:
            return pd.DataFrame()

        idx = row.index[0]
        scores = [(i, s) for i, s in enumerate(self.sim[idx]) if i != idx]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        ids = [i for i, _ in scores]
        sims = [float(s) for _, s in scores]

        df = self.movies.iloc[ids][[
            "movie_id", "title", "genres", "year", "rating",
            "imdbId", "tmdbId", "poster_url"
        ]].copy()

        df["similarity_score"] = sims
        df.index = range(1, len(df) + 1)
        return df


# ============================================================
# 4. COLLABORATIVE FILTERING (SVD)
# ============================================================

class CollaborativeRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df.copy()
        self.ratings = ratings_df.copy()
        self._build()

    def _build(self):
        mat = self.ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        n = min(50, min(mat.shape) - 1)
        svd = TruncatedSVD(n_components=n, random_state=42)

        U = svd.fit_transform(mat)
        Vt = svd.components_
        self.pred = pd.DataFrame(np.dot(U, Vt), index=mat.index, columns=mat.columns)

    def recommend(self, user_id, top_n=5):
        rated = set(self.ratings[self.ratings["user_id"] == user_id]["movie_id"])
        preds = self.pred.loc[user_id]

        unrated = preds[~preds.index.isin(rated)]
        top_ids = unrated.nlargest(top_n).index

        df = self.movies[self.movies["movie_id"].isin(top_ids)][[
            "movie_id", "title", "genres", "year", "rating",
            "imdbId", "tmdbId", "poster_url"
        ]].copy()

        df["predicted_rating"] = df["movie_id"].map(preds)
        df = df.sort_values("predicted_rating", ascending=False)

        df.index = range(1, len(df) + 1)
        return df


# ============================================================
# 5. HYBRID RECOMMENDER
# ============================================================

class HybridRecommender:
    def __init__(self, cb, cf, movies_df):
        self.cb = cb
        self.cf = cf
        self.movies = movies_df.copy()

    def recommend(self, user_id, title, top_n=5, alpha=0.5):
        cb_df = self.cb.recommend(title, top_n=20)
        cf_df = self.cf.recommend(user_id, top_n=20)

        if cb_df.empty or cf_df.empty:
            return pd.DataFrame()

        scaler = MinMaxScaler()

        cb_df["content_score"] = scaler.fit_transform(cb_df[["similarity_score"]])
        cf_df["collab_score"] = scaler.fit_transform(cf_df[["predicted_rating"]])

        merged = pd.merge(cb_df, cf_df[["title", "collab_score"]], on="title", how="outer").fillna(0)
        merged["hybrid_score"] = alpha * merged["collab_score"] + (1 - alpha) * merged["content_score"]

        merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)

        merged.index = range(1, len(merged) + 1)
        return merged[[
            "movie_id", "title", "genres", "year", "rating",
            "imdbId", "tmdbId", "poster_url", "hybrid_score"
        ]]


# ============================================================
# 6. METRICS
# ============================================================

def precision_at_k(rec, rel, k):
    return sum(1 for x in rec[:k] if x in rel) / k

def recall_at_k(rec, rel, k):
    return 0 if len(rel) == 0 else sum(1 for x in rec[:k] if x in rel) / len(rel)

def evaluate_collaborative(cf, ratings_df, k=5):
    p, r = [], []

    for uid in ratings_df["user_id"].unique():
        user_r = ratings_df[ratings_df["user_id"] == uid]
        relevant = set(user_r[user_r["rating"] >= 4]["movie_id"])
        if len(relevant) == 0:
            continue

        recs = cf.recommend(uid, top_n=k)
        rec_ids = recs["movie_id"].tolist()

        p.append(precision_at_k(rec_ids, relevant, k))
        r.append(recall_at_k(rec_ids, relevant, k))

    return {
        "Precision@K": round(np.mean(p), 4) if p else 0,
        "Recall@K": round(np.mean(r), 4) if r else 0,
        "K": k
    }