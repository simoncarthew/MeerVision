# IMPORTS
import pandas as pd

# READ IN TRAINING CSV
train_df = pd.read_csv('train.csv')

# SET STD TRAINING PARAMETERS
batch = 32
epochs = 30
lr = 0.01
augment = False
percval = 0.5

# LOOP OVER ALL TRAINING INSTANCES
for index, row in train_df.iterrows():
    print(f"Index: {index}")
    print(f"Row data:\n{row}")
    # Access individual columns like row['column_name']