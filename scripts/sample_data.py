import pandas as pd


# Reading the original training file
df=pd.read_json("data/original_train.jsonl", lines=True)

# Reduction of the training sample for computational efficiency
df_0 = df[df['label'] == 0].sample(n=500, random_state=42)
df_1 = df[df['label'] == 1].sample(n=500, random_state=42)

# Combining the results
restricted_training_df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Generate Training CSV File with 1000 datapoints
restricted_training_df.to_csv("data/training_sample.csv")


