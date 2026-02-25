# 🎬 Movie Recommendation System

A complete Machine Learning project implementing **three recommendation strategies** in pure Python.

---

## 📌 Approaches Implemented

| # | Method | Algorithm | Use Case |
|---|--------|-----------|----------|
| 1 | **Content-Based Filtering** | TF-IDF + Cosine Similarity | "Find movies like *Inception*" |
| 2 | **Collaborative Filtering** | SVD (Matrix Factorization) | "What should *User 3* watch next?" |
| 3 | **Hybrid Recommender** | Weighted blend of 1 + 2 | Best of both worlds |

---

## 🗂 Project Structure

```
movie_recommender/
├── movie_recommender.py   # Main system (all models + demo)
|___app.py
├── README.md              # This file
└── requirements.txt       # Dependencies
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
from movie_recommender import (
    MOVIES, RATINGS, USERS,
    ContentBasedRecommender,
    CollaborativeRecommender,
    HybridRecommender,
    run_demo
)

# Build models
content_rec = ContentBasedRecommender(MOVIES)
collab_rec  = CollaborativeRecommender(MOVIES, RATINGS)
hybrid_rec  = HybridRecommender(content_rec, collab_rec, MOVIES)

# ── Content-Based ──────────────────────────────
recs = content_rec.recommend("Inception", top_n=5)
print(recs)

# ── Collaborative ──────────────────────────────
recs = collab_rec.recommend(user_id=3, top_n=5)
print(recs)

# ── Hybrid ─────────────────────────────────────
recs = hybrid_rec.recommend(user_id=3, movie_title="Inception", top_n=5, alpha=0.5)
print(recs)

# ── Run full demo ──────────────────────────────
run_demo()
```

---

## 🧠 How Each Model Works

### 1. Content-Based Filtering
- Combines each movie's **genres + description** into a feature string
- Applies **TF-IDF vectorization** (Term Frequency–Inverse Document Frequency)
- Measures similarity between movies using **Cosine Similarity**
- Recommends movies with highest similarity to the query movie

### 2. Collaborative Filtering (SVD)
- Builds a **User × Movie** rating matrix
- Applies **Truncated SVD** (Singular Value Decomposition) to factorize it
- Reconstructed matrix captures latent user preferences
- Recommends unrated movies with highest predicted ratings for a user

### 3. Hybrid Recommender
- Normalizes both sets of scores to [0, 1] via MinMaxScaler
- Computes weighted average:  
  `hybrid_score = α × collab_score + (1−α) × content_score`
- Default `alpha = 0.5` (equal weight); tune as needed

---

## 📊 Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision@K** | Hits / K | % of recommended items that are relevant |
| **Recall@K** | Hits / Total Relevant | % of relevant items that were recommended |

Evaluated with **leave-one-out** cross-validation across all users.

---

## 📦 Dataset

Built-in sample dataset with:
- **10000 classic movies** (Action, Drama, Sci-Fi, Animation, Crime…)
- **16 users** with ratings
- **40 ratings** (scale 1–5)

To use your own data, replace the `MOVIES` and `RATINGS` DataFrames with the required schema:

```python
MOVIES  → columns: movie_id, title, genres, description, year, rating
RATINGS → columns: user_id, movie_id, rating
```

---

## 🔧 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | `5` | Number of recommendations to return |
| `alpha` | `0.5` | Collaborative weight in hybrid (0=pure content, 1=pure collab) |
| `n_components` | `5` | SVD latent factors |
| `ngram_range` | `(1,2)` | TF-IDF unigrams + bigrams |

---

## 📚 Dependencies

```
pandas
numpy
scikit-learn
```
