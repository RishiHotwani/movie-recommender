import streamlit as st
import pandas as pd
import numpy as np
import textwrap
import warnings
warnings.filterwarnings("ignore")

st.cache_data.clear()
st.cache_resource.clear()

from movie_recommender import (
    MOVIES, RATINGS, USERS,
    ContentBasedRecommender,
    CollaborativeRecommender,
    HybridRecommender,
    evaluate_collaborative, get_poster
)

# =====================================================================
# PAGE CONFIG (must be FIRST Streamlit command)
# =====================================================================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown(
    """
<style>
.stApp { background-color: #0e1117 !important; }

[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

.movie-card {
    background: #1a1f2e;
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
}

.movie-card:hover {
    border-color: #e50914;
}

.movie-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
}

.movie-meta {
    font-size: 13px;
    color: #a0aec0;
}

.genre-tag {
    display:inline-block;
    background:#2d3561;
    color:#90cdf4;
    padding:2px 10px;
    border-radius:20px;
    font-size:11px;
    margin:2px 5px 2px 0;
}

.score-badge {
    float:right;
    background:#e50914;
    color:white;
    padding:3px 10px;
    border-radius:8px;
    font-size:12px;
    font-weight:700;
}

.rank-badge {
    background:#e50914;
    color:white;
    padding:6px 12px;
    border-radius:8px;
    font-weight:700;
    margin-right:10px;
}

.section-header {
    font-size:25px;
    font-weight:700;
    color:#e50914;
    padding-left:10px;
    border-left:4px solid #e50914;
    margin-top:20px;
    margin-bottom:20px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# SESSION STATE INIT
# =====================================================================
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = RATINGS.copy()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "watched" not in st.session_state:
    st.session_state.watched = []


# =====================================================================
# LOAD MODELS (cached)
# =====================================================================
@st.cache_resource
def load_models():
    cb = ContentBasedRecommender(MOVIES)
    cf = CollaborativeRecommender(MOVIES, RATINGS)
    hy = HybridRecommender(cb, cf, MOVIES)
    return cb, cf, hy

cb, cf, hy = load_models()


# =====================================================================
# MOVIE CARD RENDERER
# =====================================================================
def render_movie(rank, row, score_col=None, score_label=None):
    from movie_recommender import get_poster

    poster = row.get("poster_url")
    if not poster:
        poster = get_poster(
            row["movie_id"], row["title"], row["year"],
            row.get("imdbId"), row.get("tmdbId")
        )

    score_html = ""
    if score_col and score_col in row:
        try:
            s = float(row[score_col])
            score_html = f"<span class='score-badge'>{score_label}: {s:.3f}</span>"
        except:
            pass

    genres = str(row.get("genres", ""))
    tags_html = "".join(
        f"<span class='genre-tag'>{g}</span>" for g in genres.split()
    )

    # ZERO indentation HTML (important)
    html = f"""
<div class="movie-card">
<div style="display:flex; gap:20px;">
<img src="{poster}" width="120" style="border-radius:10px;">

<div style="flex:1;">
<div>
<span class="rank-badge">{rank}</span>
<span class="movie-title">{row.get('title','')}</span>
{score_html}
</div>

<div class="movie-meta">
📅 {int(row.get('year',0)) if row.get('year',0) > 0 else "—"} &nbsp;|&nbsp; ⭐ {round(float(row.get('rating',0)),2)}
</div>

<div style="margin-top:10px;">
{tags_html}
</div>

</div>
</div>
</div>
    """

    st.markdown(html, unsafe_allow_html=True)
# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("🎬 Movie Recommender")

    page = st.radio("Navigation", [
        "🏠 Home", "🔍 Content-Based", "👥 Collaborative",
        "🤝 Hybrid", "🔎 Search", "⭐ Rate Movies",
        "📋 Watchlist", "📊 Analytics"
    ])

    st.markdown("---")
    user_label = st.selectbox("Select User", USERS["name"])
    selected_user = USERS[USERS["name"] == user_label]["user_id"].iloc[0]

    top_n = st.slider("Number of results", 3, 20, 5)


# =====================================================================
# HOME
# =====================================================================
if page == "🏠 Home":
    st.markdown('<div class="section-header">🏠 Dashboard Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", len(MOVIES))
    c2.metric("Users", len(USERS))
    c3.metric("Ratings", len(RATINGS))
    c4.metric("Genres", len(sorted(set(" ".join(MOVIES["genres"]).split()))))

    st.markdown('<div class="section-header">⭐ Top Rated Movies</div>', unsafe_allow_html=True)

    top_df = MOVIES.nlargest(12, "rating")[["title", "rating"]].set_index("title")
    st.bar_chart(top_df)

    st.markdown('<div class="section-header">🎬 Full Movie List</div>', unsafe_allow_html=True)
    st.dataframe(MOVIES[["title", "genres", "year", "rating"]], use_container_width=True)


# =====================================================================
# CONTENT-BASED
# =====================================================================
elif page == "🔍 Content-Based":
    st.markdown('<div class="section-header">🔍 Content-Based Recommendations</div>', unsafe_allow_html=True)

    movie = st.selectbox("Choose a movie:", sorted(MOVIES["title"]))

    if st.button("Get Recommendations"):
        recs = cb.recommend(movie, top_n=top_n)
        for i, row in recs.iterrows():
            render_movie(i, row, "similarity_score", "Similarity")


# =====================================================================
# COLLABORATIVE
# =====================================================================
elif page == "👥 Collaborative":
    st.markdown('<div class="section-header">👥 Collaborative Filtering</div>', unsafe_allow_html=True)

    if st.button("Get Recommendations"):
        recs = cf.recommend(selected_user, top_n=top_n)
        for i, row in recs.iterrows():
            render_movie(i, row, "predicted_rating", "Predicted")


# =====================================================================
# HYBRID
# =====================================================================
elif page == "🤝 Hybrid":
    st.markdown('<div class="section-header">🤝 Hybrid Recommendations</div>', unsafe_allow_html=True)

    movie = st.selectbox("Choose movie:", sorted(MOVIES["title"]))
    alpha = st.slider("Collaborative Weight", 0.0, 1.0, 0.5)

    if st.button("Blend Recommendations"):
        recs = hy.recommend(selected_user, movie, top_n=top_n, alpha=alpha)
        for i, row in recs.iterrows():
            render_movie(i, row, "hybrid_score", "Hybrid")


# =====================================================================
# SEARCH
# =====================================================================
elif page == "🔎 Search":
    st.markdown('<div class="section-header">🔎 Search Movies</div>', unsafe_allow_html=True)

    q = st.text_input("Search by name:")
    filtered = MOVIES.copy()

    if q:
        filtered = filtered[filtered["title"].str.contains(q, case=False)]

    st.dataframe(filtered, use_container_width=True)


# =====================================================================
# RATE MOVIES
# =====================================================================
elif page == "⭐ Rate Movies":
    st.markdown('<div class="section-header">⭐ Rate Movies</div>', unsafe_allow_html=True)

    rated_ids = set(st.session_state.user_ratings[
        st.session_state.user_ratings["user_id"] == selected_user
    ]["movie_id"])

    df = MOVIES[~MOVIES["movie_id"].isin(rated_ids)]

    for _, row in df.head(20).iterrows():
        st.subheader(row["title"])
        rating = st.slider(f"Rate {row['title']}", 1, 5, 3)

        if st.button(f"Submit_{row['movie_id']}"):
            st.session_state.user_ratings = pd.concat([
                st.session_state.user_ratings,
                pd.DataFrame([{
                    "user_id": selected_user,
                    "movie_id": row["movie_id"],
                    "rating": rating
                }])
            ])
            st.success("Rating saved!")
            st.rerun()


# =====================================================================
# WATCHLIST
# =====================================================================
elif page == "📋 Watchlist":
    st.markdown('<div class="section-header">📋 My Watchlist</div>', unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty.")
    else:
        for i, title in enumerate(st.session_state.watchlist):
            row = MOVIES[MOVIES["title"] == title].iloc[0]
            render_movie(i + 1, row)


# =====================================================================
# ANALYTICS
# =====================================================================
elif page == "📊 Analytics":
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

    cf_temp = CollaborativeRecommender(MOVIES, st.session_state.user_ratings)

    for k in [3, 5, 10]:
        m = evaluate_collaborative(cf_temp, st.session_state.user_ratings, k=k)
        st.metric(f"Precision@{k}", m["Precision@K"])
        st.metric(f"Recall@{k}", m["Recall@K"])

    st.markdown("---")
    st.subheader("SVD Explained Variance %")

    from sklearn.decomposition import TruncatedSVD

    um = st.session_state.user_ratings.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)

    comps = min(10, min(um.shape) - 1)
    svd = TruncatedSVD(n_components=comps)
    svd.fit(um)

    var_df = pd.DataFrame({
        "Component": [f"SVD-{i+1}" for i in range(comps)],
        "Variance %": svd.explained_variance_ratio_ * 100
    })

    st.bar_chart(var_df.set_index("Component"))