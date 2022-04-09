import pandas as pd

metadata = pd.read_csv('movieLens-sm/movies.csv', low_memory=False)

print(metadata.head(3))
print("THIS IS A COLLABORATIVE RECOMMENDATION SYSTEM")