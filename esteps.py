import pandas as pd

# Load one file manually — no header, tab-separated
df = pd.read_csv("als4.ts", sep="\t", header=None)
print(df.shape)   # how many rows (time windows) and columns (signals)?
print(df.head())  # what does the data look like?