import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


class ContentRecommender:

    def __init__(
        self,
        movies,
        tags
    ):

        self.movies = (
            movies.copy()
        )

        self.tags = (
            tags.copy()
        )

        self._prepare_features()

    def _prepare_features(
        self
    ):

        movie_tags = (
            self.tags.groupby(
                "movieId"
            )["tag"]
            .apply(
                lambda x:
                " ".join(
                    str(tag)
                    for tag in x
                )
            )
            .reset_index()
        )

        movie_tags.rename(
            columns={
                "tag":
                "combined_tags"
            },
            inplace=True
        )

        self.movies = (
            self.movies.merge(
                movie_tags,
                on="movieId",
                how="left"
            )
        )

        self.movies[
            "combined_tags"
        ] = (
            self.movies[
                "combined_tags"
            ].fillna("")
        )

        self.movies[
            "genres_clean"
        ] = (
            self.movies[
                "genres"
            ]
            .str.replace(
                "|",
                " ",
                regex=False
            )
        )

        self.movies[
            "features"
        ] = (
            self.movies[
                "genres_clean"
            ]
            + " "
            +
            self.movies[
                "combined_tags"
            ]
        )

        self.vectorizer = (
            TfidfVectorizer(
                stop_words="english"
            )
        )

        self.tfidf_matrix = (
            self.vectorizer
            .fit_transform(
                self.movies[
                    "features"
                ]
            )
        )

        self.cosine_sim = (
            cosine_similarity(
                self.tfidf_matrix,
                self.tfidf_matrix
            )
        )

        self.movie_indices = (
            pd.Series(
                self.movies.index,
                index=self.movies[
                    "title"
                ]
            )
        )

    def recommend(
        self,
        movie_title,
        n=10
    ):

        if (
            movie_title
            not in
            self.movie_indices
        ):
            return None

        idx = (
            self.movie_indices[
                movie_title
            ]
        )

        sim_scores = list(
            enumerate(
                self.cosine_sim[
                    idx
                ]
            )
        )

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        sim_scores = (
            sim_scores[
                1:n+1
            ]
        )

        movie_ids = [
            score[0]
            for score
            in sim_scores
        ]

        similarity_scores = [
            round(
                score[1],
                3
            )
            for score
            in sim_scores
        ]

        recommendations = (
            self.movies.iloc[
                movie_ids
            ][
                [
                    "title",
                    "genres"
                ]
            ]
            .copy()
        )

        recommendations[
            "similarity"
        ] = (
            similarity_scores
        )

        return (
            recommendations
        )