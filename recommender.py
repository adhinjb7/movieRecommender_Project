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
# AVERAGE RATINGS
# ==========================================

average_ratings = (
    ratings.groupby("movieId")["rating"]
    .mean()
    .reset_index()
)

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

movie_tags = (
    tags.groupby("movieId")["tag"]
    .apply(
        lambda x:
        " ".join(str(tag) for tag in x)
    )
    .reset_index()
)

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
# CONTENT-BASED RECOMMENDER
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["features"]
)

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
)

# ==========================================
# COLLABORATIVE FILTERING
# ==========================================

movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

movie_similarity = (
    movie_matrix.corr()
)

# ==========================================
# LOOKUP TABLES
# ==========================================

title_to_movieid = pd.Series(
    movies["movieId"].values,
    index=movies["title"]
)

movieid_to_title = pd.Series(
    movies["title"].values,
    index=movies["movieId"]
)

# ==========================================
# TOP GENRE RECOMMENDER
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
        [
            "title",
            "genres",
            "avg_rating"
        ]
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

    sim_scores = sim_scores[
        1:n+1
    ]

    movie_ids = [
        score[0]
        for score in sim_scores
    ]

    similarity_values = [
        round(score[1], 3)
        for score in sim_scores
    ]

    recommendations = (
        movies.iloc[movie_ids]
        [
            [
                "title",
                "genres"
            ]
        ]
        .copy()
    )

    recommendations[
        "similarity"
    ] = similarity_values

    return recommendations

# ==========================================
# COLLABORATIVE RECOMMENDER
# ==========================================

def recommend_collaborative(
    movie_title,
    n=10
):

    if movie_title not in (
        title_to_movieid.index
    ):
        return None

    movie_id = (
        title_to_movieid[
            movie_title
        ]
    )

    if movie_id not in (
        movie_similarity.columns
    ):
        return None

    similar_movies = (
        movie_similarity[
            movie_id
        ]
        .sort_values(
            ascending=False
        )
    )

    similar_movies = (
        similar_movies.iloc[
            1:n+1
        ]
    )

    results = []

    for movie_id, score in (
        similar_movies.items()
    ):

        results.append(
            {
                "title":
                movieid_to_title[
                    movie_id
                ],

                "correlation":
                round(
                    score,
                    3
                )
            }
        )

    return pd.DataFrame(
        results
    )

# ==========================================
# TEST EXAMPLES
# ==========================================

print(
    "\n=== CONTENT BASED ==="
)

print(
    recommend_similar_movies(
        "Toy Story (1995)"
    )
)

print(
    "\n=== COLLABORATIVE ==="
)

print(
    recommend_collaborative(
        "Toy Story (1995)"
    )
)

# ==========================================
# INTERACTIVE MENU
# ==========================================

while True:

    print(
        "\nMovie Recommendation System"
    )

    print(
        "1. Content-Based"
    )

    print(
        "2. Collaborative"
    )

    print(
        "3. Top Genre Movies"
    )

    print(
        "4. Quit"
    )

    choice = input(
        "\nSelect option: "
    )

    if choice == "1":

        movie = input(
            "Movie title: "
        )

        result = (
            recommend_similar_movies(
                movie
            )
        )

        print(result)

    elif choice == "2":

        movie = input(
            "Movie title: "
        )

        result = (
            recommend_collaborative(
                movie
            )
        )

        print(result)

    elif choice == "3":

        genre = input(
            "Genre: "
        )

        result = (
            recommend_top_genre(
                genre
            )
        )

        print(result)

    elif choice == "4":

        print(
            "Goodbye!"
        )

        break

    else:

        print(
            "Invalid option."
        )