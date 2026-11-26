
import streamlit as st
import pandas as pd
import ast
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favourite movie!")
st.divider()


# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/movies_metadata.csv",
        low_memory=False,
        on_bad_lines="skip"
    )

    required_columns = ["title", "genres", "overview"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing from the CSV."
            )

    df = df[required_columns].copy()

    df["title"] = df["title"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("[]").astype(str)

    def extract_genres(value):
        try:
            genre_list = ast.literal_eval(value)
            if isinstance(genre_list, list):
                return " ".join(
                    item.get("name", "")
                    for item in genre_list
                    if isinstance(item, dict)
                )
        except (ValueError, SyntaxError):
            pass

        return re.sub(r"[^a-zA-Z ]", " ", value)

    df["genres"] = df["genres"].apply(extract_genres)

    df = df[df["title"].str.strip() != ""]
    df = df.drop_duplicates(subset=["title"])
    df = df.reset_index(drop=True)

    df["combined_features"] = (
        df["title"] + " " +
        df["genres"] + " " +
        df["overview"]
    )

    return df


# ---------------- CREATE RECOMMENDATION MODEL ----------------
@st.cache_resource
def build_model(text_data):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )

    feature_matrix = vectorizer.fit_transform(text_data)

    return feature_matrix


# ---------------- MAIN APPLICATION ----------------
try:
    with st.spinner("Loading movie dataset..."):
        movies = load_data()

    if len(movies) < 2:
        st.error("The dataset does not contain enough movies.")
        st.stop()

    with st.spinner("Building recommendation model..."):
        feature_matrix = build_model(
            movies["combined_features"].tolist()
        )

    st.success("Movie Recommendation System is ready!")

    # Dataset overview
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Movies", f"{len(movies):,}")

    with col2:
        st.metric("Dataset Columns", "3")

    with col3:
        st.metric("Recommendation Method", "TF-IDF")

    st.divider()

    # Movie selection
    st.subheader("🍿 Find Similar Movies")

    selected_index = st.selectbox(
        "Choose your favourite movie:",
        options=movies.index.tolist(),
        format_func=lambda i: movies.loc[i, "title"]
    )

    number = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("🎬 Recommend Movies", type="primary"):

        # Calculate similarity for the selected movie only
        similarity_scores = linear_kernel(
            feature_matrix[selected_index],
            feature_matrix
        ).flatten()

        # Sort movies by similarity
        similar_indices = similarity_scores.argsort()[::-1]

        st.subheader("✨ Recommended Movies")

        count = 0

        for movie_index in similar_indices:

            if movie_index == selected_index:
                continue

            if similarity_scores[movie_index] <= 0:
                continue

            movie = movies.iloc[movie_index]

            with st.container(border=True):
                st.markdown(f"### 🎥 {movie['title']}")

                st.write(
                    "**Genres:**",
                    movie["genres"] if movie["genres"] else "Not available"
                )

                overview = movie["overview"]

                if overview:
                    st.write(overview)
                else:
                    st.write("Movie description not available.")

                st.caption(
                    f"Similarity Score: "
                    f"{similarity_scores[movie_index] * 100:.2f}%"
                )

            count += 1

            if count >= number:
                break

        if count == 0:
            st.warning("No similar movies were found.")

except FileNotFoundError:
    st.error(
        "Dataset not found! Make sure movies_metadata.csv "
        "is inside the data folder."
    )

except Exception as error:
    st.error(f"An error occurred: {error}")
    st.info(
        "Check that your CSV file is correct and contains "
        "title, genres and overview columns."
    )
