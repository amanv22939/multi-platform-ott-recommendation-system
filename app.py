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

.subtitle {
    color: #b3b3b3;
    font-size: 1rem;
    margin-bottom: 1.5rem;
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
    margin-bottom: 10px;
}

.card-title {
    color: white;
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 0.9rem;
    margin-bottom: 0.8rem;
    min-height: 64px;
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Multi-Platform OTT Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Search movie/show and get smart recommendations across platforms.</div>',
    unsafe_allow_html=True
)

# --------------------------------
# SESSION STATE
# --------------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# --------------------------------
# HELPERS
# --------------------------------
def parse_numeric_rating(value):
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


def parse_genres(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return text


def normalize_df(df, source_name, platform_label):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    out = pd.DataFrame()

    # TITLE
    possible_title_cols = ["title", "movie_title", "name", "original_title"]
    title_col = next((c for c in possible_title_cols if c in df.columns), None)
    out["title"] = df[title_col].astype(str) if title_col else ""

    # DIRECTOR
    possible_director_cols = ["director", "directors"]
    director_col = next((c for c in possible_director_cols if c in df.columns), None)
    out["director"] = df[director_col].astype(str) if director_col else ""

    # CAST
    possible_cast_cols = ["cast", "actors", "stars", "crew"]
    cast_col = next((c for c in possible_cast_cols if c in df.columns), None)
    out["cast"] = df[cast_col].astype(str) if cast_col else ""

    # GENRE
    if "listed_in" in df.columns:
        out["listed_in"] = df["listed_in"].astype(str)
    elif "genres" in df.columns:
        out["listed_in"] = df["genres"].astype(str)
    elif "genre" in df.columns:
        out["listed_in"] = df["genre"].astype(str)
    else:
        out["listed_in"] = ""

    # DESCRIPTION
    possible_desc_cols = ["description", "overview", "plot", "summary"]
    desc_col = next((c for c in possible_desc_cols if c in df.columns), None)
    out["description"] = df[desc_col].astype(str) if desc_col else ""

    # COUNTRY
    possible_country_cols = ["country", "production_countries", "origin_country"]
    country_col = next((c for c in possible_country_cols if c in df.columns), None)
    out["country"] = df[country_col].astype(str) if country_col else ""

    # RATING
    possible_rating_cols = ["rating", "vote_average", "imdb_score", "score"]
    rating_col = next((c for c in possible_rating_cols if c in df.columns), None)
    out["rating"] = df[rating_col].astype(str) if rating_col else ""

    # TYPE
    if "type" in df.columns:
        out["type"] = df["type"].astype(str)
    elif "media_type" in df.columns:
        out["type"] = df["media_type"].astype(str)
    else:
        out["type"] = "Movie"

    # PLATFORM
    out["platform"] = platform_label

    standard_cols = ["title", "director", "cast", "listed_in", "description", "country", "rating", "type", "platform"]
    for col in standard_cols:
        if col not in out.columns:
            out[col] = ""

    for col in standard_cols:
        out[col] = out[col].fillna("").astype(str)

    out["title"] = out["title"].str.strip()
    out = out[out["title"] != ""].copy()

    return out

# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_and_prepare_data():
    file_map = {
        "netflix_titles.csv": "Netflix",
        "amazon_prime_titles.csv": "Amazon Prime",
        "disney_plus_titles.csv": "Disney+",
        "indian_movies.csv": "Indian Library",
        "tmdb_5000_movies.csv": "Global Library",
        "movies_updated.csv": "Updated Library"
    }

    frames = []

    for filename, platform_label in file_map.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                cleaned = normalize_df(df, filename, platform_label)
                frames.append(cleaned)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")

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
# BUILD TF-IDF
# --------------------------------
@st.cache_resource
def build_tfidf(tags_series):
    tfidf = TfidfVectorizer(stop_words="english", max_features=6000)
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
        clean_title = title.split(":")[0].strip()

        if content_type.lower() == "tv show":
            search_url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={clean_title}"
        else:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"

        search_response = requests.get(search_url, timeout=8)
        search_data = search_response.json()

        if "results" not in search_data or len(search_data["results"]) == 0:
            alt_title = title.replace(":", "").replace("-", "").strip()

            if content_type.lower() == "tv show":
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
                if content_type.lower() == "tv show":
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
# RECOMMENDATION FUNCTION
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
    seen_titles = set()
    seen_keys = set()

    max_rating = max(data["numeric_rating"].max(), 1)
    scored_items = []

    for i, sim_score in sim_scores:
        row = data.iloc[i]
        normalized_rating = row["numeric_rating"] / max_rating
        hybrid_score = (sim_score * 0.75) + (normalized_rating * 0.25)
        scored_items.append((i, hybrid_score, sim_score))

    scored_items = sorted(scored_items, key=lambda x: x[1], reverse=True)

    for i, hybrid_score, sim_score in scored_items[1:]:
        row = data.iloc[i]
        unique_key = (row["title"], row["platform"])

        if sim_score < 0.10:
            continue

        if row["title"].strip().lower() == title_clean:
            continue

        if row["title"] in seen_titles:
            continue

        if unique_key in seen_keys:
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

        seen_titles.add(row["title"])
        seen_keys.add(unique_key)

        if len(recommendations) == top_n:
            break

    return recommendations

# --------------------------------
# WATCHLIST
# --------------------------------
def add_to_watchlist(item):
    key = (item["Title"], item["Platform"])
    existing = [(x["Title"], x["Platform"]) for x in st.session_state.watchlist]

    if key not in existing:
        st.session_state.watchlist.append(item)

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

    st.subheader("🎯 Recommended Titles")

    if results:
        cols = st.columns(5)

        for idx, item in enumerate(results):
            with cols[idx % 5]:
                poster, overview, trailer_url = fetch_details(item["Title"], item["Type"])
                short_overview = overview[:140] + "..." if overview else "No description available"

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
    else:
        st.warning("No recommendations found for the selected filters.")

# --------------------------------
# WATCHLIST DISPLAY
# --------------------------------
if show_watchlist:
    st.subheader("📌 My Watchlist")

    if st.session_state.watchlist:
        watchlist_df = pd.DataFrame(st.session_state.watchlist)
        st.dataframe(watchlist_df, use_container_width=True)
    else:
        st.info("Watchlist is empty.")