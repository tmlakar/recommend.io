
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import warnings

def dataProcess():
    sourceFile = open('results.txt', 'w')
    """
    Getting the data from the movieLens 100k data set and formatting it.
    """

    rating_df = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    item_df = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1",
                          names=["movie_id", "movie_title", "release_date", "video_release_date",
                                 "imbd_url", "unknown", "action", "adventure", "animation",
                                 "childrens", "comedy", "crime", "documentary", "drama", "fantasy",
                                 "film_noir", "horror", "musical", "mystery", "romance",
                                 "sci-fi", "thriller", "war", "western"])

    user_df = pd.read_csv("ml-100k/u.user", sep="|", encoding="latin-1", names=["user_id", "age", "gender",
                                                                                "occupation", "zip_code"])

    print(rating_df, file=sourceFile)
    print(item_df, file=sourceFile)
    print(user_df, file=sourceFile)

    # checking unique users
    print(f"# of Unique Users: {rating_df['user_id'].nunique()}")

    # checking number of items
    print(f"# of items: {rating_df['item_id'].nunique()}")

    # convert timestamp column to time stamp
    rating_df["timestamp"] = rating_df.timestamp.apply(lambda x: datetime.fromtimestamp(x / 1e3))

    # check if change has been applied
    print(rating_df.info())
    rating_df.head()

    # peak at items_df
    print(item_df.info())
    item_df.head()

    # drop empty column
    item_df.drop("video_release_date", axis=1, inplace=True)

    # convert non-null values to datetime in release_date
    item_df["release_date"] = item_df[item_df.release_date.notna()]["release_date"].apply(
        lambda x: datetime.strptime(x, "%d-%b-%Y"))

    # check if change is applied
    print(item_df.info(), item_df.shape)
    item_df.head()

    full_df = pd.merge(user_df, rating_df, how="left", on="user_id")
    full_df = pd.merge(full_df, item_df, how="left", right_on="movie_id", left_on="item_id")
    full_df.head()
    print(full_df, file=sourceFile)

    """
    Plotting some data.
    """

    # return number of rows associated to each title
    top_ten_movies = full_df.groupby("movie_title").size().sort_values(ascending=False)[:5]

    # plot the counts
    # plt.figure(figsize=(12, 5))
    # plt.barh(y= top_ten_movies.index,
    #          width= top_ten_movies.values)
    # plt.title("10 Most Rated Movies in the Data", fontsize=16)
    # plt.ylabel("Moive", fontsize=14)
    # plt.xlabel("Count", fontsize=14)
    # plt.show()

    genres = ["unknown", "action", "adventure", "animation", "childrens", "comedy", "crime",
              "documentary", "drama", "fantasy", "film_noir", "horror", "musical",
              "mystery", "romance", "sci-fi", "thriller", "war", "western"]

    full_df[full_df.movie_title == "Star Wars (1977)"][genres].iloc[0].sort_values(ascending=False)

    # the least rated movies
    least_10_movies = full_df.groupby("movie_title").size().sort_values(ascending=False)[-10:]
    print(least_10_movies)

    movies_rated = rating_df.groupby("user_id").size().sort_values(ascending=False)
    print(f"Max movies rated by one user: {max(movies_rated)}\nMin movies rated by one user: {min(movies_rated)}")

    """
    Number of male and female raters + popular genres among genders.
    """

    # count the number of male and female raters
    gender_counts = user_df.gender.value_counts()

    # plot the counts
    plt.figure(figsize=(12, 5))
    plt.bar(x=gender_counts.index[0], height=gender_counts.values[0], color="blue")
    plt.bar(x=gender_counts.index[1], height=gender_counts.values[1], color="orange")
    plt.title("Number of Male and Female Participants", fontsize=16)
    plt.xlabel("Gender", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.show()

    full_df[genres + ["gender"]].groupby("gender").sum().T.plot(kind="barh", figsize=(12, 5), color=["orange", "blue"])
    plt.xlabel("Counts", fontsize=14)
    plt.ylabel("Genre", fontsize=14)
    plt.title("Popular Genres Among Genders", fontsize=16)
    plt.show()

    """
    Genre popularity
    """

    # get the genre names in the dataframe and their counts
    label = item_df.loc[:, "unknown":].sum().index
    label_counts = item_df.loc[:, "unknown":].sum().values

    # plot a bar chart
    plt.figure(figsize=(12, 5))
    plt.barh(y=label, width=label_counts)
    plt.title("Genre Popularity", fontsize=16)
    plt.ylabel("Genres", fontsize=14)
    plt.xlabel("Counts", fontsize=14)

    plt.show()

    """
    Store data - occupation
    """
    # creating the index and values variables for occupation
    occ_label = user_df.occupation.value_counts().index
    occ_label_counts = user_df.occupation.value_counts().values

    # plot horizontal bar chart
    plt.figure(figsize=(12, 5))
    plt.barh(y=occ_label, width=occ_label_counts)
    plt.title("Most common User Occupations", fontsize=16)
    plt.show()

    # creating a empty df to store data
    df_temp = pd.DataFrame(columns=["occupation", "avg_rating"])

    # loop through all the occupations
    for idx, occ in enumerate(occ_label):
        df_temp.loc[idx, "occupation"] = occ
        df_temp.loc[idx, "avg_rating"] = round(full_df[full_df["occupation"] == occ]["rating"].mean(), 2)

    # sort from highest to lowest
    df_temp = df_temp.sort_values("avg_rating", ascending=False).reset_index(drop=True)
    print(df_temp)

    sourceFile.close()