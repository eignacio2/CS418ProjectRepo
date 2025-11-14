import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('data/raw/dataset.csv')

# Group by genre and compute average popularity
genre_popularity = (
    df.groupby("track_genre")["popularity"]
    .mean()
    .sort_values()
)

# Create plot
plt.figure(figsize=(14, 8))
plt.bar(genre_popularity.index, genre_popularity.values)

plt.title("Average Popularity by Genre", fontsize=16)
plt.xlabel("Genre", fontsize=14)
plt.ylabel("Average Popularity", fontsize=14)

# Rotate labels so they don't overlap
plt.xticks(rotation=90, fontsize=8)

# Make layout fit better
plt.tight_layout()

# Show plot
plt.show()
