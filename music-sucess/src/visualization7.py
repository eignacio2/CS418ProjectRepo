import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("spotify_tracks.csv")

# Convert date column
df['added_at'] = pd.to_datetime(df['added_at'], errors='coerce')
df['month'] = df['added_at'].dt.month

# Compute Top 25% threshold
hit_threshold = df['track_popularity'].quantile(0.75)

# Create hit indicator
df['is_hit'] = df['track_popularity'] >= hit_threshold

# Group by month and calculate % hits
monthly_hit_rate = df.groupby('month')['is_hit'].mean().reset_index()
monthly_hit_rate['hit_rate'] = monthly_hit_rate['is_hit'] * 100

# Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(monthly_hit_rate['month'], monthly_hit_rate['hit_rate'])
plt.title("Monthly Hit Rate (Top 25% of Popularity)")
plt.xlabel("Month")
plt.ylabel("Hit Rate (%)")
plt.xticks(range(1, 13))