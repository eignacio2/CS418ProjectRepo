import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/mnt/data/spotify_tracks.csv')

# Convert release date to datetime
df['album_release_date'] = pd.to_datetime(df['album_release_date'], errors='coerce')
df = df.dropna(subset=['album_release_date'])

# Extract month
df['month'] = df['album_release_date'].dt.month

# Define summer months
df['is_summer'] = df['month'].isin([6, 7, 8])

# High-energy artists selected earlier
high_energy_artists = [
    "Martin Garrix", "Major Lazer", "Dua Lipa",
    "Doja Cat", "Bad Bunny", "KAROL G", "Travis Scott"
]
df_he = df[df['artist_name'].isin(high_energy_artists)].copy()

# Prepare data
summer_popularity = df_he[df_he['is_summer'] == True]['popularity']
nonsummer_popularity = df_he[df_he['is_summer'] == False]['popularity']

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot(
    [summer_popularity, nonsummer_popularity],
    labels=["Summer Releases", "Non-Summer Releases"]
)

plt.ylabel("Track Popularity")
plt.title("Popularity of High-Energy Artists in Summer vs Non-Summer Months")

plt.tight_layout()
plt.show()
