import pandas as pd

# Load the CSV data
df = pd.read_csv("output/validation_results.csv", sep=";")

# Drop columns that are not needed
df = df.drop(columns=["TP", "FP", "FN"])

# Group by temperature and calculate mean metrics
grouped_by_temperature = df.groupby("temperature").mean(numeric_only=True)

print("=== Averages grouped by temperature ===")
print(grouped_by_temperature)

print("\n")

# Group by technique and calculate mean metrics
grouped_by_technique = df.groupby("technique").mean(numeric_only=True)

print("=== Averages grouped by technique ===")
print(grouped_by_technique)
