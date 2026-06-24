import pandas as pd


class HybridRecommender:

    def __init__(
        self,
        content_recommender,
        collaborative_recommender
    ):

        self.content = (
            content_recommender
        )

        self.collaborative = (
            collaborative_recommender
        )

    def recommend(
        self,
        movie_title,
        n=10,
        content_weight=0.5,
        collaborative_weight=0.5
    ):

        content_results = (
            self.content.recommend(
                movie_title,
                n=n * 2
            )
        )

        collaborative_results = (
            self.collaborative.recommend(
                movie_title,
                n=n * 2
            )
        )

        if (
            content_results is None
            and
            collaborative_results is None
        ):
            return None

        combined_scores = {}

        if (
            content_results
            is not None
        ):

            for _, row in (
                content_results.iterrows()
            ):

                title = row["title"]

                score = (
                    row["similarity"]
                    *
                    content_weight
                )

                combined_scores[
                    title
                ] = score

        if (
            collaborative_results
            is not None
        ):

            for _, row in (
                collaborative_results.iterrows()
            ):

                title = row["title"]

                score = (
                    row["correlation"]
                    *
                    collaborative_weight
                )

                if (
                    title
                    in
                    combined_scores
                ):

                    combined_scores[
                        title
                    ] += score

                else:

                    combined_scores[
                        title
                    ] = score

        recommendations = []

        for (
            title,
            score
        ) in (
            combined_scores.items()
        ):

            recommendations.append(
                {
                    "title": title,
                    "hybrid_score":
                    round(
                        score,
                        3
                    )
                }
            )

        recommendations = sorted(
            recommendations,
            key=lambda x:
            x["hybrid_score"],
            reverse=True
        )

        return pd.DataFrame(
            recommendations[:n]
        )