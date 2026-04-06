import os
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

st.title("🎬 Multi-Platform OTT Recommendation System")

# --------------------------------
# SESSION STATE
# --------------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

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
        data[col] = data[col].fillna("")

    data["title_clean"] = data["title"].str.lower().str.strip()

    data["tags"] = (
        data["director"] + " " +
        data["cast"] + " " +
        data["listed_in"] + " " +
        data["description"] + " " +
        data["country"] + " " +
        data["platform"]
    )

    data = data.drop_duplicates(subset=["title_clean", "platform"]).reset_index(drop=True)

    return data

data = load_and_prepare_data()

if data.empty:
    st.error("No dataset files found. Keep CSV files in the same folder as app.py.")
    st.stop()

# --------------------------------
# COMPUTE SIMILARITY
# --------------------------------
@st.cache_resource
def compute_similarity(tags_series):
    tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
    vectors = tfidf.fit_transform(tags_series)
    return cosine_similarity(vectors)

similarity = compute_similarity(data["tags"])

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
# FETCH MOVIE DETAILS FROM TMDB
# --------------------------------
@st.cache_data(show_spinner=False)
def fetch_details(title, content_type):
    api_key = st.secrets["TMDB_API_KEY"]

    try:
        clean_title = title.replace(":", "").replace("-", "").strip()

        if content_type == "TV Show":
            search_url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={clean_title}"
        else:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"

        search_response = requests.get(search_url, timeout=5)
        search_data = search_response.json()

        if "results" not in search_data or len(search_data["results"]) == 0:
            alt_title = title.split(":")[0].strip()

            if content_type == "TV Show":
                search_url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={alt_title}"
            else:
                search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={alt_title}"

            search_response = requests.get(search_url, timeout=5)
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

                video_response = requests.get(video_url, timeout=5)
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

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    seen = set()

    for i, score in sim_scores[1:]:
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
            "Score": round(score * 100, 1)
        })

        seen.add(unique_key)

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
# MAIN BUTTON
# --------------------------------
recommend_btn = st.button("Recommend")

if recommend_btn:
    results = recommend(
        selected_movie,
        top_n=top_n,
        platform=selected_platform,
        content_type=selected_type
    )

    st.subheader("Recommended Titles")

    if results:
        cols = st.columns(5)

        for idx, item in enumerate(results):
            with cols[idx % 5]:
                poster, overview, trailer_url = fetch_details(item["Title"], item["Type"])
                short_overview = overview[:120] + "..." if overview else "No description available"

                if poster:
                    st.image(poster)
                else:
                    st.info("No poster available")

                st.markdown(f"**{item['Title']}**")
                st.write(f"⭐ {item['Rating']}")
                st.write(f"🎞️ {item['Type']}")
                st.write(f"📺 {item['Platform']}")
                st.write(f"🔥 Match: {item['Score']}%")
                st.caption(short_overview)

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