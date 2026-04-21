import pandas as pd

# URL to the raw data on GitHub
data_url = "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"

# Load the dataset
# The 24.csv usually has columns like 'Rank', 'Puzzles'
df = pd.read_csv(data_url)

# Sample 50 problems randomly
# random_state ensures your results are reproducible if you run the script again
sampled_df = df.sample(n=50, random_state=42)

# Save your sample to a local file for your experiment
sampled_df.to_csv("my_game_of_24_sample.csv", index=False)

print(f"Successfully sampled {len(sampled_df)} problems.")
print(sampled_df.head())