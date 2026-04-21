import pandas as pd

data_url = "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"

df = pd.read_csv(data_url)

sampled_df = df.sample(n=50, random_state=42)
sampled_df.to_csv("test_data.csv", index=False)
print(sampled_df.head())