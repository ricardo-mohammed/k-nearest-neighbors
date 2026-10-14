
"""Exploratory data analysis helpers for the Palmer Penguins dataset."""

import seaborn as sns
from IPython.display import display

def inspect_penguins(df):
    """Inspect the Palmer Penguins dataset."""

    print("=== Dataset Shape ===")
    print(df.shape)

    print("\n=== First 5 Rows ===")
    display(df.head())

    print("\n=== Dataset Information ===")
    df.info()

    print("\n=== Missing Values ===")
    display(df.isnull().sum().to_frame("Missing Values"))

    print("\n=== Numerical Statistics ===")
    display(df.describe())

    print("\n=== Species Distribution ===")
    display(df["species"].value_counts().to_frame("Count"))

    print("\n=== Duplicate Rows ===")
    print(df.duplicated().sum())
    

def run_eda():
    """Load the Palmer Penguins dataset and perform EDA."""
    df = sns.load_dataset("penguins")
    inspect_penguins(df)