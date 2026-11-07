
"""
GR-2 Student Performance Prediction - Dataset Generator
Generates a synthetic dataset of student academic metrics and performance labels.
Suitable for BCA Machine Learning Lab practicals and projects.
"""

import os
import random
import numpy as np
import pandas as pd


def generate_student_dataset(total_students: int = 200, output_path: str = "data/student_performance.csv") -> pd.DataFrame:
    """
    Generates a synthetic student performance dataset with academic attributes
    and classifies students into Good, Average, or Poor performance tiers.

    Parameters:
        total_students (int): Number of student records to generate (default: 200).
        output_path (str): File path where the CSV dataset will be saved.

    Returns:
        pd.DataFrame: Generated student performance DataFrame.
    """
    # Set seed for reproducible random data generation
    np.random.seed(42)
    random.seed(42)

    # Generate synthetic student academic records
    data = {
        "Student_ID": list(range(1, total_students + 1)),
        "Attendance": np.random.randint(40, 101, total_students),
        "Previous_Marks": np.random.randint(30, 101, total_students),
        "Study_Hours": np.round(
            np.random.uniform(1, 8, total_students), 1
        ),
        "Assignment_Marks": np.random.randint(30, 101, total_students),
        "Participation": np.random.randint(1, 11, total_students)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate overall performance score using weighted academic factors
    df["Performance_Score"] = (
        df["Attendance"] * 0.20
        + df["Previous_Marks"] * 0.30
        + df["Study_Hours"] * 5
        + df["Assignment_Marks"] * 0.25
        + df["Participation"] * 1.5
    )

    # Classify student performance based on calculated score
    def classify_performance(score: float) -> str:
        if score >= 88:
            return "Good"
        elif score >= 55:
            return "Average"
        else:
            return "Poor"

    df["Performance"] = df["Performance_Score"].apply(classify_performance)

    # Create data folder if not present and save dataset
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        print("Dataset created successfully!")
        print(f"Total students: {len(df)}")
        print("\nFirst 5 records:")
        print(df.head())
        print("\nPerformance distribution:")
        print(df["Performance"].value_counts())
    except Exception as error:
        print(f"Error while saving dataset: {error}")
        raise

    return df


if __name__ == "__main__":
    generate_student_dataset()
