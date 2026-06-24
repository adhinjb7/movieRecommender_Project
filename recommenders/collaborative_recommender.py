import pandas as pd


class CollaborativeRecommender:

    def __init__(
        self,
        movies,
        ratings
    ):

        self.movies = (
            movies.copy()
        )

        self.ratings = (
            ratings.copy()
        )

        self._build_model()

    def _build_model(
        self
    ):

        self.movie_matrix = (
            self.ratings.pivot_table(
                index="userId",
                columns="movieId",
                values="rating"
            )
        )

        self.movie_similarity = (
            self.movie_matrix.corr()
        )

        self.title_to_movieid = (
            pd.Series(
                self.movies["movieId"].values,
                index=self.movies["title"]
            )
        )

        self.movieid_to_title = (
            pd.Series(
                self.movies["title"].values,
                index=self.movies["movieId"]
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
            self.title_to_movieid.index
        ):
            return None

        movie_id = (
            self.title_to_movieid[
                movie_title
            ]
        )

        if (
            movie_id
            not in
            self.movie_similarity.columns
        ):
            return None

        similar_movies = (
            self.movie_similarity[
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

        for (
            similar_movie_id,
            correlation
        ) in (
            similar_movies.items()
        ):

            if pd.isna(
                correlation
            ):
                continue

            results.append(
                {
                    "title":
                    self.movieid_to_title[
                        similar_movie_id
                    ],

                    "correlation":
                    round(
                        correlation,
                        3
                    )
                }
            )

        return pd.DataFrame(
            results
        )

    def recommend_for_user(
        self,
        user_id,
        n=10
    ):

        user_ratings = (
            self.ratings[
                self.ratings[
                    "userId"
                ]
                ==
                user_id
            ]
        )

        if len(
            user_ratings
        ) == 0:

            return None

        liked_movies = (
            user_ratings[
                user_ratings[
                    "rating"
                ]
                >= 4.0
            ]
        )

        recommendation_scores = {}

        for movie_id in (
            liked_movies[
                "movieId"
            ]
        ):

            if (
                movie_id
                not in
                self.movie_similarity.columns
            ):
                continue

            similar_movies = (
                self.movie_similarity[
                    movie_id
                ]
            )

            for (
                similar_movie_id,
                score
            ) in (
                similar_movies.items()
            ):

                if pd.isna(
                    score
                ):
                    continue

                if (
                    similar_movie_id
                    in
                    recommendation_scores
                ):

                    recommendation_scores[
                        similar_movie_id
                    ] += score

                else:

                    recommendation_scores[
                        similar_movie_id
                    ] = score

        already_seen = set(
            user_ratings[
                "movieId"
            ]
        )

        recommendations = []

        for (
            movie_id,
            score
        ) in (
            recommendation_scores.items()
        ):

            if (
                movie_id
                in
                already_seen
            ):
                continue

            recommendations.append(
                {
                    "title":
                    self.movieid_to_title[
                        movie_id
                    ],

                    "score":
                    round(
                        score,
                        3
                    )
                }
            )

        recommendations = sorted(
            recommendations,
            key=lambda x:
            x["score"],
            reverse=True
        )

        return pd.DataFrame(
            recommendations[:n]
        )