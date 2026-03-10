import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("hf://datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json")

# split 85% train 10% test 5% eval
train_df, temp_df = train_test_split(df, test_size=0.15, random_state=42)
test_df, eval_df = train_test_split(temp_df, test_size=1/3, random_state=42)

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Eval: {len(eval_df)}")
