
# INTERN CIRCLE - TASK 2
# Exploratory Data Analysis on Titanic Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

# 2. DataFrame operations
print("\nFirst 5 rows:")
display(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
display(df.describe(include="all").T)

# 3. Missing value analysis
print("\nMissing values per column:")
display(df.isnull().sum().to_frame("Missing Values"))

print("\nMissing value percentage:")
display(
    (df.isnull().mean() * 100)
    .round(2)
    .to_frame("Missing Percentage")
)

# 4. Duplicate analysis
print("\nDuplicate rows before cleaning:", df.duplicated().sum())

# Preserve the original data for reference
cleaned_df = df.copy()

# Remove exact duplicate rows
cleaned_df = cleaned_df.drop_duplicates().copy()

print("Duplicate rows after cleaning:", cleaned_df.duplicated().sum())

# 5. Handle missing values
cleaned_df["age"] = cleaned_df["age"].fillna(
    cleaned_df["age"].median()
)

cleaned_df["embarked"] = cleaned_df["embarked"].fillna(
    cleaned_df["embarked"].mode()[0]
)

# Deck has substantial missingness; remove the column
cleaned_df = cleaned_df.drop(columns=["deck"])

print("\nMissing values after cleaning:")
display(cleaned_df.isnull().sum().to_frame("Remaining Missing Values"))

# 6. Survival analysis
print("\nSurvival counts (0 = Did not survive, 1 = Survived):")
display(cleaned_df["survived"].value_counts().sort_index())

print("\nOverall survival percentage:")
print(round(cleaned_df["survived"].mean() * 100, 2), "%")

# 7. Grouping and aggregation by gender
print("\nSurvival by gender:")
gender_analysis = cleaned_df.groupby("sex")["survived"].agg(
    ["count", "mean"]
)
gender_analysis.columns = ["Passenger Count", "Survival Rate"]
gender_analysis["Survival Percentage"] = (
    gender_analysis["Survival Rate"] * 100
).round(2)
display(gender_analysis)

# 8. Grouping and aggregation by passenger class
print("\nSurvival by passenger class:")
class_analysis = cleaned_df.groupby("pclass")["survived"].agg(
    ["count", "mean"]
)
class_analysis.columns = ["Passenger Count", "Survival Rate"]
class_analysis["Survival Percentage"] = (
    class_analysis["Survival Rate"] * 100
).round(2)
display(class_analysis)

# 9. Numerical analysis by passenger class
print("\nAge and fare statistics by class:")
display(
    cleaned_df.groupby("pclass")[["age", "fare"]]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

# 10. Correlation between numerical variables
print("\nNumerical correlation matrix:")
display(
    cleaned_df.select_dtypes(include="number")
    .corr()
    .round(2)
)

# 11. Visualize survival counts using Pandas and Matplotlib
survival_counts = cleaned_df["survived"].value_counts().sort_index()

plt.figure(figsize=(7, 4))
plt.bar(
    ["Did Not Survive", "Survived"],
    survival_counts.values
)
plt.title("Titanic Passenger Survival Counts")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# 12. Visualize survival by gender
gender_analysis["Survival Rate"].plot(
    kind="bar",
    figsize=(7, 4),
    title="Survival Rate by Gender",
    ylabel="Survival Rate",
    xlabel="Gender",
    ylim=(0, 1)
)
plt.tight_layout()
plt.show()

# 13. Visualize survival by passenger class
class_analysis["Survival Rate"].plot(
    kind="bar",
    figsize=(7, 4),
    title="Survival Rate by Passenger Class",
    ylabel="Survival Rate",
    xlabel="Passenger Class",
    ylim=(0, 1)
)
plt.tight_layout()
plt.show()

# 14. Save cleaned dataset
cleaned_df.to_csv("titanic_cleaned.csv", index=False)

print("\nEDA COMPLETED SUCCESSFULLY")
print("Cleaned dataset saved as titanic_cleaned.csv")
