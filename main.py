import pandas as pd

print("WELCOME TO THE COLLABORATIVE RECOMMENDATION SYSTEM")

print("........................................................")

"""
Getting the data from the movieLens 100k data set and formatting it.
"""

rating_df = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

item_df = pd.read_csv("ml-100k/u.item", sep="|",encoding="latin-1",
                      names=["movie_id", "movie_title", "release_date", "video_release_date",
                             "imbd_url", "unknown", "action", "adventure", "animation",
                             "childrens", "comedy", "crime", "documentary", "drama", "fantasy",
                             "film_noir", "horror", "musical", "mystery", "romance",
                             "sci-fi", "thriller", "war", "western"])

user_df = pd.read_csv("ml-100k/u.user", sep="|", encoding="latin-1", names=["user_id", "age", "gender",
                                                                            "occupation", "zip_code"])

print(rating_df)
print(item_df)
print(user_df)

"""
Predicting the rating or preference that a user would give to an item
"""



print("........................................................")