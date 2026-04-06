import os
import re
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Multi-Platform OTT Recommendation System",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
}

.section-title {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.subtitle {
    color: #b3b3b3;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.ott-card {
    background: linear-gradient(180deg, #141414 0%, #1c1c1c 100%);
    border-radius: 18px;
    padding: 12px;
    min-height: 620px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
    border: 1px solid rgba(255,255,255,0.06);
}

.ott-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 10px 30px rgba(229, 9, 20, 0.25);
    border: 1px solid rgba(229, 9, 20, 0.45);
}

.poster-fallback {
    background: #19324d;
    color: #8fd3ff;
    border-radius: 14px;
    height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    text-align: center;
    padding: 16px;
}

.card-title {
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 0.9rem;
    margin-bottom: 0.8rem;
    min-height: 60px;
}

.meta {
    color: #f2f2f2;
    font-size: 1rem;
    margin-bottom: 0.55rem;
}

.desc {
    color: #bfbfbf;
    font-size: 0.98rem;
    line-height: 1.6;
    margin-top: 0.7rem;
    min-height: 95px;
}

.small-card {
    background: linear-gradient(180deg, #171717 0%, #222 100%);
    border-radius: 14px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    min-height: 180px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.small-card:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(229, 9, 20, 0.18);
}

.small-title {
    color: white;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.small-meta {
    color: #d9d9d9;
    font-size: 0.95rem;
    margin-bottom: 0.45rem;
}

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 2rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Multi-Platform OTT Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover similar titles, cross-platform matches, trending picks, and top-rated content.</div>', unsafe_allow_html=True)

# --------------------------------
# SESSION STATE
# --------------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# --------------------------------
# HELPERS
# --------------------------------
def parse_numeric_rating(value):
    """
    Converts ratings like '7.8', '8', 'TV-MA', 'PG-13', '' into a usable numeric score.
    Non-numeric values return 0.
    """
    if pd.isna(value):
        return 0.0

    text = str(value).strip()

    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        try:
            return float(match.group())
        except Exception:
            return 0.0

    return 0.0


# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_and_prepare_data():
    file_map = {
        "Netflix": "netflix_titles.csv",
        "Amazon Prime": "amazon_prime_titles.csv",
        "Disney+": "disney_plus_titles.csv"
    }

    frames = []

    for platform_name, filename in file_map.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df["platform"] = platform_name
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    needed_cols = [
        "title", "director", "cast", "listed_in",
        "description", "country", "rating", "type", "platform"
    ]

    for col in needed_cols:
        if col not in data.columns:
            data[col] = ""

    for col in needed_cols:
        data[col] = data[col].fillna("").astype(str)

    data["title_clean"] = data["title"].str.lower().str.strip()

    data["tags"] = (
        data["director"] + " " +
        data["cast"] + " " +
        data["listed_in"] + " " +
        data["description"] + " " +
        data["country"] + " " +
        data["platform"]
    )

    data["numeric_rating"] = data["rating"].apply(parse_numeric_rating)

    data = data.drop_duplicates(subset=["title_clean", "platform"]).reset_index(drop=True)

    return data


data = load_and_prepare_data()

if data.empty:
    st.error("No dataset files found. Keep CSV files in the same folder as app.py.")
    st.stop()

# --------------------------------
# BUILD TF-IDF VECTORS
# --------------------------------
@st.cache_resource
def build_tfidf(tags_series):
    tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
    vectors = tfidf.fit_transform(tags_series)
    return vectors

tfidf_vectors = build_tfidf(data["tags"])

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.header("Filters")

platform_options = ["All"] + sorted(data["platform"].dropna().unique().tolist())
selected_platform = st.sidebar.selectbox("Select Platform", platform_options)

type_options = ["All"] + sorted(data["type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Select Type", type_options)

top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)
show_watchlist = st.sidebar.checkbox("Show Watchlist")

# --------------------------------
# SEARCH
# --------------------------------
movie_list = sorted(data["title"].dropna().unique().tolist())

search_query = st.text_input("🔍 Search movie or show")

if search_query:
    filtered_movie_list = [m for m in movie_list if search_query.lower() in m.lower()]
else:
    filtered_movie_list = movie_list

if not filtered_movie_list:
    st.warning("No titles found for this search.")
    st.stop()

selected_movie = st.selectbox("Select movie or show", filtered_movie_list)

# --------------------------------
# TMDB DETAILS
# --------------------------------
@st.cache_data(show_spinner=False)
def fetch_details(title, content_type):
    api_key = st.secrets.get("TMDB_API_KEY", None)

    if not api_key:
        return None, "", None

    try:
        clean_title = title.replace(":", "").replace("-", "").strip()

        if content_type == "TV Show":
            search_url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={clean_title}"
        else:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"

        search_response = requests.get(search_url, timeout=8)
        search_data = search_response.json()

        if "results" not in search_data or len(search_data["results"]) == 0:
            alt_title = title.split(":")[0].strip()

            if content_type == "TV Show":
                search_url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={alt_title}"
            else:
                search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={alt_title}"

            search_response = requests.get(search_url, timeout=8)
            search_data = search_response.json()

        if "results" in search_data and len(search_data["results"]) > 0:
            item_data = search_data["results"][0]

            poster_path = item_data.get("poster_path")
            overview = item_data.get("overview", "")
            item_id = item_data.get("id")

            poster_url = None
            if poster_path:
                poster_url = "https://image.tmdb.org/t/p/w500" + poster_path

            trailer_url = None
            if item_id:
                if content_type == "TV Show":
                    video_url = f"https://api.themoviedb.org/3/tv/{item_id}/videos?api_key={api_key}"
                else:
                    video_url = f"https://api.themoviedb.org/3/movie/{item_id}/videos?api_key={api_key}"

                video_response = requests.get(video_url, timeout=8)
                video_data = video_response.json()

                if "results" in video_data:
                    for video in video_data["results"]:
                        if video.get("site") == "YouTube" and video.get("type") in ["Trailer", "Teaser"]:
                            trailer_key = video.get("key")
                            trailer_url = f"https://www.youtube.com/watch?v={trailer_key}"
                            break

            return poster_url, overview, trailer_url

        return None, "", None

    except Exception:
        return None, "", None


# --------------------------------
# RECOMMENDATION FUNCTIONS
# --------------------------------
def recommend(title, top_n=5, platform="All", content_type="All"):
    title_clean = title.lower().strip()

    matching_rows = data[data["title_clean"] == title_clean]
    if matching_rows.empty:
        return []

    idx = matching_rows.index[0]

    sim_scores_array = cosine_similarity(tfidf_vectors[idx], tfidf_vectors).flatten()
    sim_scores = list(enumerate(sim_scores_array))

    recommendations = []
    seen = set()

    # Hybrid score = 0.75 * similarity + 0.25 * normalized rating
    scored_items = []
    max_rating = max(data["numeric_rating"].max(), 1)

    for i, sim_score in sim_scores:
        row = data.iloc[i]
        normalized_rating = row["numeric_rating"] / max_rating
        hybrid_score = (sim_score * 0.75) + (normalized_rating * 0.25)
        scored_items.append((i, hybrid_score, sim_score))

    scored_items = sorted(scored_items, key=lambda x: x[1], reverse=True)

    for i, hybrid_score, sim_score in scored_items[1:]:
        row = data.iloc[i]
        unique_key = (row["title"], row["platform"])

        if unique_key in seen:
            continue

        if platform != "All" and row["platform"] != platform:
            continue

        if content_type != "All" and row["type"] != content_type:
            continue

        recommendations.append({
            "Title": row["title"],
            "Type": row["type"],
            "Rating": row["rating"],
            "Platform": row["platform"],
            "Genre": row["listed_in"],
            "Score": round(hybrid_score * 100, 1)
        })

        seen.add(unique_key)

        if len(recommendations) == top_n:
            break

    return recommendations


def cross_platform_recommend(title, top_n=5):
    title_clean = title.lower().strip()

    matching_rows = data[data["title_clean"] == title_clean]
    if matching_rows.empty:
        return []

    idx = matching_rows.index[0]
    selected_platform_name = matching_rows.iloc[0]["platform"]

    sim_scores_array = cosine_similarity(tfidf_vectors[idx], tfidf_vectors).flatten()
    sim_scores = list(enumerate(sim_scores_array))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    results = []
    seen = set()

    for i, score in sim_scores[1:]:
        row = data.iloc[i]
        unique_key = (row["title"], row["platform"])

        if unique_key in seen:
            continue

        if row["platform"] == selected_platform_name:
            continue

        results.append({
            "Title": row["title"],
            "Platform": row["platform"],
            "Type": row["type"],
            "Rating": row["rating"],
            "Score": round(score * 100, 1)
        })

        seen.add(unique_key)

        if len(results) == top_n:
            break

    return results


def get_top_trending(n=10):
    trending = data.copy()
    trending = trending.sort_values(by=["numeric_rating", "title"], ascending=[False, True])
    return trending.head(n)


def get_best_by_rating(content_type="Movie", n=10):
    best = data.copy()
    if content_type != "All":
        best = best[best["type"] == content_type]
    best = best.sort_values(by=["numeric_rating", "title"], ascending=[False, True])
    return best.head(n)


# --------------------------------
# WATCHLIST
# --------------------------------
def add_to_watchlist(item):
    key = (item["Title"], item["Platform"])
    existing = [(x["Title"], x["Platform"]) for x in st.session_state.watchlist]

    if key not in existing:
        st.session_state.watchlist.append(item)


# --------------------------------
# TOP TRENDING
# --------------------------------
st.markdown('<div class="section-title">🔥 Top Trending</div>', unsafe_allow_html=True)
trending_items = get_top_trending(5)
trend_cols = st.columns(5)

for idx, (_, row) in enumerate(trending_items.iterrows()):
    with trend_cols[idx % 5]:
        poster, overview, trailer_url = fetch_details(row["title"], row["type"])
        st.markdown('<div class="small-card">', unsafe_allow_html=True)
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown('<div class="poster-fallback">No poster available</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="small-title">{row["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-meta">⭐ {row["rating"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-meta">📺 {row["platform"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-meta">🎞️ {row["type"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------
# BEST BY RATING
# --------------------------------
st.markdown('<div class="section-title">🏆 Best by Rating</div>', unsafe_allow_html=True)
best_items = get_best_by_rating("Movie", 5)
best_cols = st.columns(5)

for idx, (_, row) in enumerate(best_items.iterrows()):
    with best_cols[idx % 5]:
        poster, overview, trailer_url = fetch_details(row["title"], row["type"])
        st.markdown('<div class="small-card">', unsafe_allow_html=True)
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown('<div class="poster-fallback">No poster available</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="small-title">{row["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-meta">⭐ {row["rating"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-meta">📺 {row["platform"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-meta">🎞️ {row["type"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------
# MOST SIMILAR ACROSS PLATFORMS
# --------------------------------
st.markdown('<div class="section-title">🌍 Most Similar Across Platforms</div>', unsafe_allow_html=True)
cross_results = cross_platform_recommend(selected_movie, top_n=5)

if cross_results:
    cross_cols = st.columns(5)
    for idx, item in enumerate(cross_results):
        with cross_cols[idx % 5]:
            poster, overview, trailer_url = fetch_details(item["Title"], item["Type"])
            st.markdown('<div class="small-card">', unsafe_allow_html=True)

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown('<div class="poster-fallback">No poster available</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="small-title">{item["Title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-meta">📺 {item["Platform"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-meta">🎞️ {item["Type"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-meta">⭐ {item["Rating"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-meta">🔥 {item["Score"]}% similar</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No cross-platform matches found.")

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------
# MAIN RECOMMENDER
# --------------------------------
recommend_btn = st.button("Recommend")

if recommend_btn:
    results = recommend(
        selected_movie,
        top_n=top_n,
        platform=selected_platform,
        content_type=selected_type
    )

    st.markdown('<div class="section-title">🎯 Recommended Titles</div>', unsafe_allow_html=True)

    if results:
        cols = st.columns(5)

        for idx, item in enumerate(results):
            with cols[idx % 5]:
                poster, overview, trailer_url = fetch_details(item["Title"], item["Type"])
                short_overview = overview[:140] + "..." if overview else "No description available"

                st.markdown('<div class="ott-card">', unsafe_allow_html=True)

                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown('<div class="poster-fallback">No poster available</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="card-title">{item["Title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">⭐ {item["Rating"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">🎞️ {item["Type"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">📺 {item["Platform"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">🔥 Match: {item["Score"]}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="desc">{short_overview}</div>', unsafe_allow_html=True)

                if trailer_url:
                    st.markdown(f"[▶ Watch Trailer]({trailer_url})")
                else:
                    st.write("Trailer not available")

                if st.button("➕ Add to Watchlist", key=f"watchlist_{idx}_{item['Title']}_{item['Platform']}"):
                    add_to_watchlist(item)

                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for the selected filters.")

# --------------------------------
# WATCHLIST DISPLAY
# --------------------------------
if show_watchlist:
    st.markdown('<div class="section-title">📌 My Watchlist</div>', unsafe_allow_html=True)

    if st.session_state.watchlist:
        watchlist_df = pd.DataFrame(st.session_state.watchlist)
        st.dataframe(watchlist_df, use_container_width=True)
    else:
        st.info("Watchlist is empty.")