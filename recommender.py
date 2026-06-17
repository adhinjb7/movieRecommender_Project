import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATA
# ==========================================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")

# ==========================================
# DATASET OVERVIEW
# ==========================================

print("\n=== MOVIES DATA ===")
print(movies.head())

print("\n=== RATINGS DATA ===")
print(ratings.head())

print("\n=== TAGS DATA ===")
print(tags.head())

print("\n=== DATASET SHAPES ===")
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Tags:", tags.shape)

# ==========================================
# GENRE ANALYSIS
# ==========================================

genre_counts = {}

for genres in movies["genres"]:

    genre_list = genres.split("|")

    for genre in genre_list:

        if genre not in genre_counts:
            genre_counts[genre] = 1
        else:
            genre_counts[genre] += 1

print("\n=== TOP 10 GENRES ===")

sorted_genres = sorted(
    genre_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for genre, count in sorted_genres[:10]:
    print(f"{genre}: {count}")

# ==========================================
# AVERAGE MOVIE RATINGS
# ==========================================

average_ratings = ratings.groupby(
    "movieId"
)["rating"].mean()

average_ratings = average_ratings.reset_index()

average_ratings.rename(
    columns={"rating": "avg_rating"},
    inplace=True
)

movies_with_ratings = movies.merge(
    average_ratings,
    on="movieId",
    how="left"
)

# ==========================================
# PROCESS TAGS
# ==========================================

movie_tags = tags.groupby(
    "movieId"
)["tag"].apply(
    lambda x: " ".join(
        str(tag)
        for tag in x
    )
)

movie_tags = movie_tags.reset_index()

movie_tags.rename(
    columns={"tag": "combined_tags"},
    inplace=True
)

movies = movies.merge(
    movie_tags,
    on="movieId",
    how="left"
)

movies["combined_tags"] = (
    movies["combined_tags"]
    .fillna("")
)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

movies["genres_clean"] = (
    movies["genres"]
    .str.replace(
        "|",
        " ",
        regex=False
    )
)

movies["features"] = (
    movies["genres_clean"]
    + " "
    + movies["combined_tags"]
)

# ==========================================
# TF-IDF
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["features"]
)

# ==========================================
# COSINE SIMILARITY
# ==========================================

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

# ==========================================
# MOVIE LOOKUP TABLE
# ==========================================

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
)

# ==========================================
# TOP MOVIES BY GENRE
# ==========================================

def recommend_top_genre(
    genre,
    n=10
):

    filtered = movies_with_ratings[
        movies_with_ratings["genres"]
        .str.contains(
            genre,
            case=False,
            na=False
        )
    ]

    filtered = filtered.sort_values(
        by="avg_rating",
        ascending=False
    )

    return filtered[
        ["title", "genres", "avg_rating"]
    ].head(n)

# ==========================================
# CONTENT-BASED RECOMMENDER
# ==========================================

def recommend_similar_movies(
    movie_title,
    n=10
):

    if movie_title not in movie_indices:
        return None

    idx = movie_indices[movie_title]

    sim_scores = list(
        enumerate(
            cosine_sim[idx]
        )
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:n+1]

    movie_ids = [
        score[0]
        for score in sim_scores
    ]

    similarity_values = [
        round(score[1], 3)
        for score in sim_scores
    ]

    recommendations = movies.iloc[
        movie_ids
    ][[
        "title",
        "genres"
    ]].copy()

    recommendations[
        "similarity"
    ] = similarity_values

    return recommendations

# ==========================================
# TESTS
# ==========================================

print("\n=== TOP SCI-FI MOVIES ===")
print(
    recommend_top_genre(
        "Sci-Fi"
    )
)

print("\n=== MOVIES SIMILAR TO TOY STORY ===")
print(
    recommend_similar_movies(
        "Toy Story (1995)"
    )
)

# ==========================================
# INTERACTIVE MODE
# ==========================================

while True:

    movie_name = input(
        "\nEnter a movie title (or quit): "
    )

    if movie_name.lower() == "quit":
        break

    recommendations = (
        recommend_similar_movies(
            movie_name
        )
    )

    if recommendations is None:
        print("Movie not found.")
    else:
        print(
            recommendations.to_string(
                index=False
            )
        )