"""
🎬 Movie Recommendation System — Streamlit UI
==============================================
Run with:  streamlit run app.py
Dependencies: streamlit, pandas, numpy, scikit-learn  (NO plotly needed)
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from movie_recommender import (
    MOVIES, RATINGS, USERS,
    ContentBasedRecommender,
    CollaborativeRecommender,
    HybridRecommender,
    evaluate_collaborative,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #111827 !important;
        padding: 10px;
    }

    .movie-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }

    .movie-card:hover { border-color: #e50914; }

    .movie-title { font-size: 17px; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
    .movie-meta  { font-size: 13px; color: #a0aec0; margin-bottom: 6px; }

    .genre-tag {
        display: inline-block; background: #2d3561; color: #90cdf4;
        border-radius: 20px; padding: 2px 10px; font-size: 11px; margin: 2px 3px 2px 0;
    }

    .score-badge {
        display: inline-block; background: #e50914; color: white;
        border-radius: 8px; padding: 2px 10px;
        font-size: 12px; font-weight: 600; float: right;
    }

    .star-rating { color: #f6c90e; font-size: 14px; }

    .rank-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; background: #e50914; color: white;
        border-radius: 50%; font-weight: 700; font-size: 13px; margin-right: 10px;
    }

    .section-header {
        font-size: 22px; font-weight: 700; color: #e50914;
        margin: 10px 0 16px 0; border-left: 4px solid #e50914; padding-left: 12px;
    }

    .metric-card {
        background: #1a1f2e; border: 1px solid #2d3561;
        border-radius: 10px; padding: 16px; text-align: center;
    }

    .metric-value { font-size: 32px; font-weight: 800; color: #e50914; }
    .metric-label { font-size: 13px; color: #a0aec0; margin-top: 4px; }

    hr { border-color: #2d3561; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = RATINGS.copy()
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'watched' not in st.session_state:
    st.session_state.watched = []


# ─────────────────────────────────────────────
# MODEL CACHING
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    cr = ContentBasedRecommender(MOVIES)
    cf = CollaborativeRecommender(MOVIES, RATINGS)
    hy = HybridRecommender(cr, cf, MOVIES)
    return cr, cf, hy

content_rec, collab_rec, hybrid_rec = load_models()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
GENRE_EMOJI = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Biography':'📖',
    'Comedy':'😂','Crime':'🔫','Drama':'🎭','History':'🏛️',
    'Romance':'❤️','Sci-Fi':'🚀','Thriller':'😱',
}

def genre_tags(genres_str: str) -> str:
    tags = []
    for g in str(genres_str).split():
        emoji = GENRE_EMOJI.get(g, '')
        tags.append(f'<span class="genre-tag">{emoji} {g}</span>')
    return ' '.join(tags)

def star_display(rating) -> str:
    try:
        rating = float(rating)
        full  = int(rating / 2)
        half  = '½' if (rating / 2 - full) >= 0.5 else ''
        empty = 5 - full - (1 if half else 0)
        return f'<span class="star-rating">{"★"*full}{half}{"☆"*empty}</span> {rating}/10'
    except:
        return ''

def render_movie_card(rank: int, row: pd.Series, score_col=None, score_label=None):
    score_html = ""
    if score_col and score_col in row.index:
        val = row[score_col]
        try:
            score_html = f'<span class="score-badge">{score_label}: {float(val):.3f}</span>'
        except:
            pass
    title    = row.get('title', '?')
    genres   = row.get('genres', '')
    year_val = row.get('year', 0)
    year     = int(year_val) if year_val and str(year_val) != 'nan' else '—'
    rating   = row.get('rating', 0)
    desc_arr = MOVIES[MOVIES['title'] == title]['description'].values
    desc_txt = desc_arr[0] if len(desc_arr) > 0 else ''
    st.markdown(f"""
    <div class="movie-card">
        <div>
            <span class="rank-badge">{rank}</span>
            <span class="movie-title">{title}</span>
            {score_html}
        </div>
        <div class="movie-meta" style="margin-top:6px">
            📅 {year} &nbsp;|&nbsp; {star_display(rating)}
        </div>
        <div>{genre_tags(genres)}</div>
        <div style="font-size:12px;color:#718096;margin-top:8px;font-style:italic">{desc_txt}</div>
    </div>
    """, unsafe_allow_html=True)

def progress_bar_chart(df, label_col, value_col, title=""):
    if title:
        st.markdown(f"**{title}**")
    max_val = df[value_col].max()
    for _, row in df.iterrows():
        ratio = float(row[value_col]) / max_val if max_val > 0 else 0
        c1, c2, c3 = st.columns([3, 5, 1])
        with c1:
            st.markdown(f"<div style='font-size:12px;color:#e2e8f0;padding-top:6px;text-align:right'>{row[label_col]}</div>", unsafe_allow_html=True)
        with c2:
            st.progress(ratio)
        with c3:
            st.markdown(f"<div style='font-size:12px;color:#e50914;padding-top:6px'>{row[value_col]:.2f}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Content-Based", "👥 Collaborative", "🤝 Hybrid",
         "🔎 Search & Filter", "⭐ Rate Movies", "📋 My Watchlist", "📊 Analytics"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 👤 Select User")
    user_options = {f"{r['name']} (User {r['user_id']})": r['user_id']
                    for _, r in USERS.iterrows()}
    selected_user_label = st.selectbox("User", list(user_options.keys()), label_visibility="collapsed")
    selected_user_id    = user_options[selected_user_label]
    selected_user_name  = selected_user_label.split(" (")[0]
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_n = st.slider("Results to show", 3, 10, 5)
    st.markdown("---")
    wl_count = len(st.session_state.watchlist)
    st.markdown(f"📋 **Watchlist:** {wl_count} movie{'s' if wl_count!=1 else ''}")


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<h1 style="color:#e50914;text-align:center">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#a0aec0;font-size:16px">Powered by Content-Based · Collaborative Filtering · Hybrid AI</p>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">20</div><div class="metric-label">🎥 Movies</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">8</div><div class="metric-label">👥 Users</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">40</div><div class="metric-label">⭐ Ratings</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">11</div><div class="metric-label">🎭 Genres</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏆 Top Rated Movies</div>', unsafe_allow_html=True)
    top_movies = MOVIES.nlargest(10, 'rating')[['title', 'rating']].reset_index(drop=True)
    progress_bar_chart(top_movies, 'title', 'rating', "IMDb Rating (out of 10)")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">🎭 Genre Distribution</div>', unsafe_allow_html=True)
        genre_counts = {}
        for genres in MOVIES['genres']:
            for g in genres.split():
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_df = pd.Series(genre_counts, name='Movies').sort_values(ascending=False)
        st.bar_chart(genre_df)
    with col_b:
        st.markdown('<div class="section-header">📅 Movies by Decade</div>', unsafe_allow_html=True)
        decade_df = MOVIES.copy()
        decade_df['decade'] = (decade_df['year'] // 10 * 10).astype(str) + 's'
        decade_counts = decade_df.groupby('decade').size().rename('Movies')
        st.bar_chart(decade_counts)

    st.markdown("---")
    st.markdown('<div class="section-header">🎬 All Movies</div>', unsafe_allow_html=True)
    display_df = MOVIES[['title','genres','year','rating']].sort_values('rating', ascending=False).reset_index(drop=True)
    display_df.index += 1
    st.dataframe(display_df, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: CONTENT-BASED
# ─────────────────────────────────────────────
elif page == "🔍 Content-Based":
    st.markdown('<div class="section-header">🔍 Content-Based Recommendations</div>', unsafe_allow_html=True)
    st.markdown("Finds movies with **similar genres and themes** using TF-IDF + Cosine Similarity.")

    movie_choice = st.selectbox("Pick a movie you like:", sorted(MOVIES['title'].tolist()))

    if st.button("🎯 Find Similar Movies", use_container_width=True):
        with st.spinner("Analyzing content…"):
            recs = content_rec.recommend(movie_choice, top_n=top_n)

        st.markdown(f"<br>**Movies similar to _{movie_choice}_:**", unsafe_allow_html=True)
        for i, row in recs.iterrows():
            col_card, col_btn = st.columns([6, 1])
            with col_card:
                render_movie_card(i, row, 'similarity_score', 'Similarity')
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("➕ WL", key=f"wl_cb_{i}", help="Add to Watchlist"):
                    if row['title'] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(row['title'])
                        st.success("Added!")

        st.markdown("---")
        st.markdown("**📊 Similarity Scores**")
        chart_df = recs.set_index('title')['similarity_score'].sort_values()
        st.bar_chart(chart_df)


# ─────────────────────────────────────────────
# PAGE: COLLABORATIVE
# ─────────────────────────────────────────────
elif page == "👥 Collaborative":
    st.markdown('<div class="section-header">👥 Collaborative Filtering</div>', unsafe_allow_html=True)
    st.markdown("Predicts your preferences based on **what similar users enjoyed** (SVD Matrix Factorization).")

    user_rated = st.session_state.user_ratings[
        st.session_state.user_ratings['user_id'] == selected_user_id
    ].merge(MOVIES[['movie_id','title','rating']], on='movie_id')

    with st.expander(f"📽 {selected_user_name}'s rated movies ({len(user_rated)})"):
        for _, r in user_rated.iterrows():
            stars = '⭐' * int(r['rating_x'])
            st.markdown(f"**{r['title']}** — {stars} ({r['rating_x']}/5)")

    if st.button("🤖 Get My Recommendations", use_container_width=True):
        with st.spinner("Running SVD…"):
            cr   = CollaborativeRecommender(MOVIES, st.session_state.user_ratings)
            recs = cr.recommend(selected_user_id, top_n=top_n)

        st.markdown(f"<br>**Picks for {selected_user_name}:**", unsafe_allow_html=True)
        for i, row in recs.iterrows():
            col_card, col_btn = st.columns([6, 1])
            with col_card:
                render_movie_card(i, row, 'predicted_rating', 'Pred. Score')
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("➕ WL", key=f"wl_cf_{i}", help="Add to Watchlist"):
                    if row['title'] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(row['title'])
                        st.success("Added!")

        st.markdown("---")
        st.markdown("**📊 Predicted Ratings**")
        chart_df = recs.set_index('title')['predicted_rating'].sort_values()
        st.bar_chart(chart_df)

    st.markdown("---")
    st.markdown("**📋 User–Movie Rating Matrix**")
    pivot = st.session_state.user_ratings.pivot_table(
        index='user_id', columns='movie_id', values='rating'
    )
    valid_cols = [c for c in pivot.columns if c in MOVIES['movie_id'].values]
    pivot.columns = MOVIES.set_index('movie_id').loc[valid_cols, 'title']
    pivot.index = [
        USERS[USERS['user_id']==uid]['name'].values[0]
        if uid in USERS['user_id'].values else f"User {uid}"
        for uid in pivot.index
    ]
    st.dataframe(pivot.fillna('—'), use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: HYBRID
# ─────────────────────────────────────────────
elif page == "🤝 Hybrid":
    st.markdown('<div class="section-header">🤝 Hybrid Recommender</div>', unsafe_allow_html=True)
    st.markdown("Blends **Content-Based + Collaborative** scores for the best of both worlds.")

    col1, col2 = st.columns(2)
    with col1:
        movie_choice = st.selectbox("Reference movie:", sorted(MOVIES['title'].tolist()))
    with col2:
        alpha = st.slider("⚖️ Collaborative weight (α)", 0.0, 1.0, 0.5, 0.05,
                          help="0 = pure content, 1 = pure collaborative")

    st.info(f"Current blend: **{int((1-alpha)*100)}% Content** + **{int(alpha*100)}% Collaborative**")

    if st.button("🚀 Generate Hybrid Recommendations", use_container_width=True):
        with st.spinner("Blending models…"):
            recs = hybrid_rec.recommend(selected_user_id, movie_choice, top_n=top_n, alpha=alpha)

        st.markdown(f"<br>**Hybrid picks for {selected_user_name} (based on _{movie_choice}_):**", unsafe_allow_html=True)
        for i, row in recs.iterrows():
            col_card, col_btn = st.columns([6, 1])
            with col_card:
                render_movie_card(i, row, 'hybrid_score', 'Hybrid Score')
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("➕ WL", key=f"wl_hy_{i}"):
                    if row['title'] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(row['title'])
                        st.success("Added!")

        st.markdown("---")
        st.markdown("**📊 Hybrid Scores**")
        chart_df = recs.set_index('title')['hybrid_score'].sort_values()
        st.bar_chart(chart_df)


# ─────────────────────────────────────────────
# PAGE: SEARCH & FILTER
# ─────────────────────────────────────────────
elif page == "🔎 Search & Filter":
    st.markdown('<div class="section-header">🔎 Search & Filter Movies</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        search_q = st.text_input("🔍 Search by title or keyword:", placeholder="e.g. space, crime, love…")
    with col2:
        sort_by = st.selectbox("Sort by", ["Rating ↓", "Rating ↑", "Year ↓", "Year ↑", "Title A-Z"])

    all_genres      = sorted(set(' '.join(MOVIES['genres']).split()))
    selected_genres = st.multiselect("Filter by genre:", all_genres)
    year_range      = st.slider("Year range:", int(MOVIES['year'].min()), int(MOVIES['year'].max()), (1970, 2023))
    rating_min      = st.slider("Minimum IMDb rating:", 0.0, 10.0, 7.0, 0.1)

    filtered = MOVIES.copy()
    if search_q:
        mask = (
            filtered['title'].str.contains(search_q, case=False) |
            filtered['description'].str.contains(search_q, case=False) |
            filtered['genres'].str.contains(search_q, case=False)
        )
        filtered = filtered[mask]
    for g in selected_genres:
        filtered = filtered[filtered['genres'].str.contains(g)]
    filtered = filtered[
        (filtered['year'] >= year_range[0]) &
        (filtered['year'] <= year_range[1]) &
        (filtered['rating'] >= rating_min)
    ]

    sort_map = {
        "Rating ↓": ('rating', False), "Rating ↑": ('rating', True),
        "Year ↓":   ('year',   False), "Year ↑":   ('year',   True),
        "Title A-Z":('title',  True),
    }
    s_col, s_asc = sort_map[sort_by]
    filtered = filtered.sort_values(s_col, ascending=s_asc).reset_index(drop=True)

    st.markdown(f"**{len(filtered)} movie{'s' if len(filtered)!=1 else ''} found**")
    st.markdown("---")

    if filtered.empty:
        st.info("No movies match your filters. Try broadening your search.")
    else:
        for i, row in filtered.iterrows():
            col_card, col_btn = st.columns([6, 1])
            with col_card:
                render_movie_card(i + 1, row)
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("➕", key=f"wl_sf_{i}", help="Add to Watchlist"):
                    if row['title'] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(row['title'])
                        st.success("Added!")


# ─────────────────────────────────────────────
# PAGE: RATE MOVIES
# ─────────────────────────────────────────────
elif page == "⭐ Rate Movies":
    st.markdown('<div class="section-header">⭐ Rate Movies</div>', unsafe_allow_html=True)
    st.markdown("Your ratings improve your collaborative recommendations!")

    rated_ids = set(
        st.session_state.user_ratings[
            st.session_state.user_ratings['user_id'] == selected_user_id
        ]['movie_id']
    )
    unrated = MOVIES[~MOVIES['movie_id'].isin(rated_ids)]

    if unrated.empty:
        st.success(f"🎉 {selected_user_name} has rated all movies!")
    else:
        st.markdown(f"**{len(unrated)} unrated movies for {selected_user_name}:**")
        st.markdown("---")
        for _, row in unrated.iterrows():
            c1, c2, c3 = st.columns([4, 2, 1])
            with c1:
                st.markdown(f"**{row['title']}** ({int(row['year'])})")
                st.markdown(f"<small style='color:#a0aec0'>{row['genres']}</small>", unsafe_allow_html=True)
            with c2:
                user_score = st.select_slider(
                    "Rate", options=[1, 2, 3, 4, 5], value=3,
                    key=f"rate_{row['movie_id']}", label_visibility="collapsed"
                )
            with c3:
                if st.button("Submit", key=f"submit_{row['movie_id']}"):
                    new_row = pd.DataFrame([{
                        'user_id':  selected_user_id,
                        'movie_id': row['movie_id'],
                        'rating':   user_score
                    }])
                    st.session_state.user_ratings = pd.concat(
                        [st.session_state.user_ratings, new_row], ignore_index=True
                    )
                    st.success(f"Rated '{row['title']}' → {'⭐'*user_score}")
                    st.rerun()
            st.markdown("---")

    rated_movies = st.session_state.user_ratings[
        st.session_state.user_ratings['user_id'] == selected_user_id
    ].merge(MOVIES[['movie_id','title']], on='movie_id')

    with st.expander(f"📋 {selected_user_name}'s ratings ({len(rated_movies)})"):
        for _, r in rated_movies.iterrows():
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{r['title']}** — {'⭐'*int(r['rating'])} ({r['rating']}/5)")
            with col_b:
                if st.button("🗑️", key=f"del_{r['movie_id']}_{selected_user_id}"):
                    st.session_state.user_ratings = st.session_state.user_ratings[
                        ~((st.session_state.user_ratings['user_id'] == selected_user_id) &
                          (st.session_state.user_ratings['movie_id'] == r['movie_id']))
                    ]
                    st.rerun()


# ─────────────────────────────────────────────
# PAGE: MY WATCHLIST
# ─────────────────────────────────────────────
elif page == "📋 My Watchlist":
    st.markdown('<div class="section-header">📋 My Watchlist</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 To Watch", "✅ Watched"])

    with tab1:
        wl = st.session_state.watchlist
        if not wl:
            st.info("Your watchlist is empty. Add movies from any recommendations page!")
        else:
            st.markdown(f"**{len(wl)} movie{'s' if len(wl)!=1 else ''} to watch:**")
            for i, title in enumerate(wl):
                movie_row = MOVIES[MOVIES['title'] == title]
                if movie_row.empty:
                    continue
                row = movie_row.iloc[0]
                c1, c2, c3 = st.columns([5, 1, 1])
                with c1:
                    render_movie_card(i+1, row)
                with c2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("✅", key=f"mark_{i}", help="Mark as watched"):
                        st.session_state.watchlist.remove(title)
                        if title not in st.session_state.watched:
                            st.session_state.watched.append(title)
                        st.rerun()
                with c3:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"rm_{i}", help="Remove"):
                        st.session_state.watchlist.remove(title)
                        st.rerun()

    with tab2:
        watched = st.session_state.watched
        if not watched:
            st.info("You haven't marked any movies as watched yet.")
        else:
            st.markdown(f"**{len(watched)} movie{'s' if len(watched)!=1 else ''} watched:**")
            for i, title in enumerate(watched):
                movie_row = MOVIES[MOVIES['title'] == title]
                if movie_row.empty:
                    continue
                render_movie_card(i+1, movie_row.iloc[0])


# ─────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown('<div class="section-header">📊 System Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Model Metrics", "👥 User Insights", "🎬 Movie Insights"])

    with tab1:
        st.markdown("#### Model Evaluation (Collaborative Filtering)")
        with st.spinner("Evaluating model…"):
            cr = CollaborativeRecommender(MOVIES, st.session_state.user_ratings)
            for k in [3, 5, 10]:
                m  = evaluate_collaborative(cr, st.session_state.user_ratings, k=k)
                p  = m['Precision@K']
                r  = m['Recall@K']
                f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0
                c1, c2, c3 = st.columns(3)
                with c1: st.metric(f"Precision@{k}", f"{p:.3f}")
                with c2: st.metric(f"Recall@{k}",    f"{r:.3f}")
                with c3: st.metric(f"F1@{k}",         f"{f1:.3f}")
                st.markdown("---")

        st.markdown("#### SVD Explained Variance %")
        from sklearn.decomposition import TruncatedSVD
        um  = st.session_state.user_ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
        n   = min(5, min(um.shape)-1)
        svd = TruncatedSVD(n_components=n, random_state=42)
        svd.fit(um)
        var_df = pd.DataFrame({
            'Component':   [f'SVD-{i+1}' for i in range(n)],
            'Variance %':  (svd.explained_variance_ratio_ * 100).round(2)
        })
        progress_bar_chart(var_df, 'Component', 'Variance %')

    with tab2:
        rpu = st.session_state.user_ratings.merge(USERS, on='user_id')

        st.markdown("#### Ratings per User")
        rpu_count = rpu.groupby('name').size().rename('# Ratings')
        st.bar_chart(rpu_count)

        st.markdown("#### Average Rating per User")
        rpu_avg = rpu.groupby('name')['rating'].mean().rename('Avg Rating').round(2)
        st.bar_chart(rpu_avg)

        st.markdown("#### Rating Distribution (all users)")
        rating_dist = st.session_state.user_ratings['rating'].value_counts().sort_index().rename('Count')
        st.bar_chart(rating_dist)

    with tab3:
        rpm = st.session_state.user_ratings.merge(MOVIES[['movie_id','title']], on='movie_id')

        st.markdown("#### Most Rated Movies")
        rpm_count = rpm.groupby('title').size().rename('# Ratings').sort_values(ascending=False)
        st.bar_chart(rpm_count)

        st.markdown("#### Average User Rating per Movie")
        rpm_avg = rpm.groupby('title')['rating'].mean().rename('Avg Rating').round(2).sort_values(ascending=False)
        st.bar_chart(rpm_avg)

        st.markdown("#### IMDb Rating Distribution")
        bins = pd.cut(MOVIES['rating'], bins=[7, 7.5, 8, 8.5, 9, 9.5, 10], right=False)
        rating_hist = bins.value_counts().sort_index().rename('Movies')
        rating_hist.index = rating_hist.index.astype(str)
        st.bar_chart(rating_hist)

        st.markdown("#### Full Movie Table")
        st.dataframe(
            MOVIES[['title','genres','year','rating']].sort_values('rating', ascending=False).reset_index(drop=True),
            use_container_width=True
        )
