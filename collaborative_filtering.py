import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, pairwise
from sklearn.metrics import mean_absolute_error

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
    users = pd.read_csv("ml-100k/u.user", sep="|", encoding="latin-1", names=["user_id", "age", "gender", "occupation", "zip_code"])

    # print(item, rating, users)

    """
        ITEMS
    """
    # peak at dataframe for items
    print("ITEMS")
    # print(item.head())
    # print()
    # we only need the movie_id and movie_title
    movies = item.loc[:, :"movie_title"].copy()

    print(movies.movie_title)
    print()

    """
        RATINGS
    """

    # peak at rating data
    print("RATINGS")
    # print(rating.head())


    # dropping timestamp
    rating.drop("timestamp", axis=1, inplace=True)
    print(rating.head())
    print()

    """
        N X M matrix where n is user_id and m is item_id, 
        so we have all the data about the rating of n user for m item.
    """

    # creating n x m matrix where n is user_id and m is item_id
    user_ratings = pd.pivot_table(rating, index="user_id", columns="item_id", values="rating").fillna(0)
    user_ratings_transpose = user_ratings.T
    user_ratings_transpose.to_csv('data/ratings.csv', sep='\t', encoding='utf-8')


    # user and item counts
    n_users = len(user_ratings.index)
    n_items = len(user_ratings.columns)
    n_ratings = len(rating.index)


    print(f"Number of users: {n_users}\nNumber of items: {n_items}")
    print(user_ratings.head())

    """
        Sparseness of the matrix.
    """

    sparsity = (1 - n_ratings/(n_items*n_users))*100;
    print(sparsity)

    """
        In this dataset, every user has rated at least 20 movies which results in a sparsity of 93.7%. 
        That means that 6.3% of the user-item ratings have a value.
    """


    train, test = train_test_split(data=user_ratings.to_numpy(), n_users=n_users, n_items=n_items)

    """
        Cosine distance to obtain similarity matrix.
    """

    user_similarity = pairwise.cosine_similarity(train + 1e-9)
    item_similarity = pairwise.cosine_similarity(train.T + 1e-9)

    # print(user_similarity.shape, item_similarity.shape)
    print()
    print("USER SIMILARITY MATRIX")
    print()
    print(user_similarity)
    print()

    """
        User based collaborative filtering.
    """

    # predict user ratings not included in data
    user_preds = np.dot(user_similarity, train) / np.array([np.abs(user_similarity).sum(axis=1)]).T

    # get the nonzero elements
    nonzero_test = test[test.nonzero()]
    nonzero_user_preds = user_preds[test.nonzero()]

    user_rating_preds = mean_squared_error(nonzero_test, nonzero_user_preds)
    print(f"UBCF Mean Squared Error: {user_rating_preds}")
    user_rating_preds1 = mean_absolute_error(nonzero_test, nonzero_user_preds)
    print(f"UBCF Mean Absolute Error: {user_rating_preds1}")

def train_test_split(data: np.array, n_users: int, n_items: int):
        # create a empty array of shape n x m for test
        test = np.zeros((n_users, n_items))
        train = data.copy()


        dataSamples = open('samples.txt', 'w')

        # for each user, we generate a random sample of 10 from movies they've watched
        for user in range(n_users):
            print(user, file=dataSamples)
            random_sample = np.random.choice(data[user, :].nonzero()[0],
                                             size=10,
                                             replace=False)
            print(random_sample, file= dataSamples)
            print("\n", file=dataSamples)
            ratings1 = pd.DataFrame(data)
            # ratings1.to_csv('filtering/ratings.csv', sep='\t', encoding='utf-8')

            # set the train to zero to represent no rating and the test will be the original rating
            train[user, random_sample] = 0.

            print("TRAIN")
            print(train, file=dataSamples)
            print("\n", file=dataSamples)
            # train1 = pd.DataFrame(train)
            # train1.to_csv('filtering/train.csv', sep='\t', encoding='utf-8')

            test[user, random_sample] = data[user, random_sample]
            print("TEST")
            print(test, file=dataSamples)
            print("\n", file=dataSamples)
            # test1 = pd.DataFrame(test)
            # test1.to_csv('filtering/test.csv', sep='\t', encoding='utf-8')




        dataSamples.close()
        return train, test



