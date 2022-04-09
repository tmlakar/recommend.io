import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, pairwise

def collaborativeFiltering():
    # # load item data
    item = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1",
                       names=["movie_id", "movie_title", "release_date", "video_release_date",
                              "imbd_url", "unknown", "action", "adventure", "animation",
                              "childrens", "comedy", "crime", "documentary", "drama", "fantasy",
                              "film_noir", "horror", "musical", "mystery", "romance",
                              "sci-fi", "thriller", "war", "western"])

    # load ratings data
    rating = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    # peak at dataframe
    item.head()

    # we only need the movie_id and movie_title
    movies = item.loc[:, :"movie_title"].copy()
    movies.head()

    print(movies.movie_title)

    # peak at rating data
    rating.head()

    # dropping timestamp
    rating.drop("timestamp", axis=1, inplace=True)
    print(rating.head())

    # creating n x m matrix where n is user_id and m is item_id
    user_ratings = pd.pivot_table(rating, index="user_id", columns="item_id", values="rating").fillna(0)

    # user and item counts
    n_users = len(user_ratings.index)
    n_items = len(user_ratings.columns)

    print(f"Users: {n_users}\nItems: {n_items}")
    print(user_ratings)

    train, test = train_test_split(data=user_ratings.to_numpy(), n_users=n_users, n_items=n_items)

    user_similarity = pairwise.cosine_similarity(train + 1e-9)
    item_similarity = pairwise.cosine_similarity(train.T + 1e-9)

    print(user_similarity.shape, item_similarity.shape)

def train_test_split(data: np.array, n_users: int, n_items: int):
        # create a empty array of shape n x m for test
        test = np.zeros((n_users, n_items))
        train = data.copy()

        # for each user, we generate a random sample of 5 from movies they've watched
        for user in range(n_users):
            random_sample = np.random.choice(data[user, :].nonzero()[0],
                                             size=5,
                                             replace=False)
            # set the train to zero to represent no rating and the test will be the original rating
            train[user, random_sample] = 0.
            test[user, random_sample] = data[user, random_sample]

        return train, test



