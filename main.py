from utils.data_loader import (
    load_data
)

from recommenders.content_recommender import (
    ContentRecommender
)

from recommenders.collaborative_recommender import (
    CollaborativeRecommender
)

from recommenders.hybrid_recommender import (
    HybridRecommender
)


def print_menu():

    print(
        "\n=============================="
    )

    print(
        "MOVIE RECOMMENDATION SYSTEM"
    )

    print(
        "=============================="
    )

    print(
        "1. Content-Based Recommendations"
    )

    print(
        "2. Collaborative Recommendations"
    )

    print(
        "3. User Recommendations"
    )

    print(
        "4. Hybrid Recommendations"
    )

    print(
        "5. Exit"
    )


def main():

    print(
        "\nLoading datasets..."
    )

    movies, ratings, tags = (
        load_data()
    )

    print(
        "Datasets loaded."
    )

    print(
        "\nBuilding recommenders..."
    )

    content = (
        ContentRecommender(
            movies,
            tags
        )
    )

    collaborative = (
        CollaborativeRecommender(
            movies,
            ratings
        )
    )

    hybrid = (
        HybridRecommender(
            content,
            collaborative
        )
    )

    print(
        "Recommenders ready."
    )

    while True:

        print_menu()

        choice = input(
            "\nSelect option: "
        )

        if choice == "1":

            movie_title = input(
                "\nMovie title: "
            )

            results = (
                content.recommend(
                    movie_title
                )
            )

            if results is None:

                print(
                    "\nMovie not found."
                )

            else:

                print(
                    "\nCONTENT-BASED RESULTS"
                )

                print(
                    results.to_string(
                        index=False
                    )
                )

        elif choice == "2":

            movie_title = input(
                "\nMovie title: "
            )

            results = (
                collaborative.recommend(
                    movie_title
                )
            )

            if results is None:

                print(
                    "\nMovie not found."
                )

            else:

                print(
                    "\nCOLLABORATIVE RESULTS"
                )

                print(
                    results.to_string(
                        index=False
                    )
                )

        elif choice == "3":

            try:

                user_id = int(
                    input(
                        "\nUser ID: "
                    )
                )

            except ValueError:

                print(
                    "\nInvalid user ID."
                )

                continue

            results = (
                collaborative
                .recommend_for_user(
                    user_id
                )
            )

            if results is None:

                print(
                    "\nUser not found."
                )

            else:

                print(
                    "\nUSER RECOMMENDATIONS"
                )

                print(
                    results.to_string(
                        index=False
                    )
                )

        elif choice == "4":

            movie_title = input(
                "\nMovie title: "
            )

            results = (
                hybrid.recommend(
                    movie_title
                )
            )

            if results is None:

                print(
                    "\nMovie not found."
                )

            else:

                print(
                    "\nHYBRID RESULTS"
                )

                print(
                    results.to_string(
                        index=False
                    )
                )

        elif choice == "5":

            print(
                "\nGoodbye!"
            )

            break

        else:

            print(
                "\nInvalid option."
            )


if __name__ == "__main__":

    main()