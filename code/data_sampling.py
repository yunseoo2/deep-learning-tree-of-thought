import pandas as pd

data_url = "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"

df = pd.read_csv(data_url)

sampled_df = df.sample(n=50, random_state=42)
sampled_df.to_csv("my_game_of_24_sample.csv", index=False)

print(f"Successfully sampled {len(sampled_df)} problems.")
print(sampled_df.head())