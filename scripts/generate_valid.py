from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/processed/dataset_processed.csv")

train, valid = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv("data/processed/train.csv", index=False)
valid.to_csv("data/processed/valid.csv", index=False)