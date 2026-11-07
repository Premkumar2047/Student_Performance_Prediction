"""
GR-2 Student Performance Prediction Project
===========================================
A Machine Learning project suitable for BCA Machine Learning syllabus.
Demonstrates:
1. Object-Oriented Programming (OOP) in Python (Student Class)
2. Data Preprocessing & Cleaning using Pandas
3. Feature Statistical Profiling using NumPy
4. Exploratory Data Analysis (EDA) visualizations using Matplotlib
5. Supervised Learning with Decision Tree & Random Forest Classifiers
6. Overfitting Analysis (Train vs Test Accuracy comparison)
7. Confusion Matrix & Classification Report metrics
8. Feature Importance analysis
9. Validated interactive single-student performance prediction with confidence score
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# STEP 1: STUDENT CLASS (OOP Concept)
# ==========================================

class Student:
    """
    Represents an individual student with academic performance attributes.
    Demonstrates Object-Oriented Programming (OOP) in Python.
    """

    def __init__(
        self,
        attendance: float,
        previous_marks: float,
        study_hours: float,
        assignment_marks: float,
        participation: float
    ):
        self.attendance = attendance
        self.previous_marks = previous_marks
        self.study_hours = study_hours
        self.assignment_marks = assignment_marks
        self.participation = participation

    def get_student_data(self) -> pd.DataFrame:
        """
        Converts the student attributes into a single-row Pandas DataFrame
        with column names matching the trained model's feature names.
        """
        return pd.DataFrame([{
            "Attendance": self.attendance,
            "Previous_Marks": self.previous_marks,
            "Study_Hours": self.study_hours,
            "Assignment_Marks": self.assignment_marks,
            "Participation": self.participation
        }])


# ==========================================
# STEP 2: LOAD AND CLEAN DATA
# ==========================================

def load_and_clean_data(
    input_file_path: str = "data/student_performance.csv",
    output_cleaned_path: str = "data/cleaned_student_performance.csv"
) -> pd.DataFrame:
    """
    Loads raw dataset from CSV, performs validation, removes duplicates,
    imputes missing values, and saves the cleaned dataset.

    Parameters:
        input_file_path (str): Path to raw CSV file.
        output_cleaned_path (str): Destination path for cleaned dataset.

    Returns:
        pd.DataFrame: Cleaned student performance dataset.
    """
    print("\n" + "=" * 50)
    print("STEP 1: LOADING & DATA PREPROCESSING")
    print("=" * 50)

    # Validate file existence with descriptive exception handling
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"\n[ERROR]: File '{input_file_path}' not found!")
        print("Please run 'python generate_dataset.py' first to create the dataset.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"\n[ERROR]: File '{input_file_path}' is empty!")
        sys.exit(1)
    except Exception as error:
        print(f"\n[ERROR]: Failed to read dataset: {error}")
        sys.exit(1)

    # Validate required columns
    expected_columns = [
        "Attendance",
        "Previous_Marks",
        "Study_Hours",
        "Assignment_Marks",
        "Participation",
        "Performance"
    ]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"\n[ERROR]: Missing required columns in dataset: {missing_cols}")
        sys.exit(1)

    # Remove duplicate records
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate record(s).")

    # Handle missing values: median for numeric, mode for categorical
    for column in df.select_dtypes(include=["number"]).columns:
        if df[column].isnull().any():
            median_val = df[column].median()
            df[column] = df[column].fillna(median_val)

    for column in df.select_dtypes(include=["object"]).columns:
        if df[column].isnull().any():
            mode_val = df[column].mode()[0]
            df[column] = df[column].fillna(mode_val)

    # Ensure output directory exists and save cleaned dataset
    os.makedirs(os.path.dirname(output_cleaned_path) or "data", exist_ok=True)
    df.to_csv(output_cleaned_path, index=False)

    print("Cleaned dataset updated successfully!")
    print(f"Cleaned records saved to: '{output_cleaned_path}'")
    print("\nPerformance Distribution:")
    print(df["Performance"].value_counts())

    return df


# ==========================================
# STEP 3: MEANINGFUL NUMPY FEATURE STATISTICS
# ==========================================

def display_numpy_feature_statistics(df: pd.DataFrame, feature_names: list) -> None:
    """
    Uses NumPy arrays and vector operations to compute
    Mean, Standard Deviation, Median, Min, and Max for all academic features.

    Parameters:
        df (pd.DataFrame): Dataset containing features.
        feature_names (list): List of feature column names.
    """
    print("\n" + "=" * 50)
    print("STEP 2: NUMPY FEATURE STATISTICAL ANALYSIS")
    print("=" * 50)

    print(f"{'Feature':<20} {'Mean':<10} {'Std Dev':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)

    for feature in feature_names:
        # Convert feature column to a 1D NumPy array
        feature_array = np.asarray(df[feature], dtype=np.float64)

        mean_val = np.mean(feature_array)
        std_val = np.std(feature_array)
        median_val = np.median(feature_array)
        min_val = np.min(feature_array)
        max_val = np.max(feature_array)

        print(
            f"{feature:<20} {mean_val:<10.2f} {std_val:<10.2f} "
            f"{median_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f}"
        )


# ==========================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

def perform_eda(df: pd.DataFrame) -> None:
    """
    Generates and saves exploratory data analysis charts:
    1. Performance class distribution bar chart.
    2. Attendance vs Performance Score scatter plot.
    3. Study Hours vs Performance Score scatter plot.
    4. Correlation heatmap of academic metrics (excluding arbitrary ID).

    Parameters:
        df (pd.DataFrame): Student performance DataFrame.
    """
    print("\n" + "=" * 50)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50)

    os.makedirs("data", exist_ok=True)

    # 1. Performance distribution bar chart
    performance_order = ["Poor", "Average", "Good"]
    performance_counts = (
        df["Performance"]
        .value_counts()
        .reindex(performance_order, fill_value=0)
    )

    plt.figure(figsize=(8, 5))
    performance_counts.plot(kind="bar", color=["#e74c3c", "#f39c12", "#2ecc71"])
    plt.title("Student Performance Distribution")
    plt.xlabel("Performance Tier")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("data/performance_distribution.png")
    plt.close()

    # 2. Attendance vs Performance Score scatter plot
    if "Performance_Score" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df["Attendance"], df["Performance_Score"], color="#2980b9", alpha=0.7)
        plt.title("Attendance vs Performance Score")
        plt.xlabel("Attendance (%)")
        plt.ylabel("Performance Score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("data/attendance_vs_performance.png")
        plt.close()

        # 3. Study Hours vs Performance Score scatter plot
        plt.figure(figsize=(8, 5))
        plt.scatter(df["Study_Hours"], df["Performance_Score"], color="#8e44ad", alpha=0.7)
        plt.title("Study Hours vs Performance Score")
        plt.xlabel("Study Hours per Day")
        plt.ylabel("Performance Score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("data/study_hours_vs_performance.png")
        plt.close()

    # 4. Correlation Heatmap
    # Exclude Student_ID to prevent false correlation with the student ID number
    numeric_df = df.select_dtypes(include=["number"])
    if "Student_ID" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["Student_ID"])

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(9, 6))
    plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(
        np.arange(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=30,
        ha="right"
    )
    plt.yticks(
        np.arange(len(correlation_matrix.columns)),
        correlation_matrix.columns
    )
    plt.title("Academic Features Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("data/correlation_heatmap.png")
    plt.close()

    print("All EDA graphs updated successfully in 'data/' directory!")


# ==========================================
# STEP 5: PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data(df: pd.DataFrame):
    """
    Extracts feature matrix (X) and target vector (y).
    Validates class balance using NumPy.

    Parameters:
        df (pd.DataFrame): DataFrame containing cleaned student records.

    Returns:
        tuple: (X, y, features)
    """
    features = [
        "Attendance",
        "Previous_Marks",
        "Study_Hours",
        "Assignment_Marks",
        "Participation"
    ]

    X = df[features]
    y = df["Performance"]

    # Meaningful NumPy usage: inspect unique classes and sample counts
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\nDataset Class Balance:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class '{cls}': {count} students ({count / len(y) * 100:.1f}%)")

    return X, y, features


# ==========================================
# STEP 6: TRAIN MODELS
# ==========================================

def train_models(X: pd.DataFrame, y: pd.Series):
    """
    Performs 80-20 stratified train-test split and trains both
    Decision Tree and Random Forest classifiers.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target performance labels.

    Returns:
        tuple: (dt_model, rf_model, X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 50)
    print("STEP 4: MODEL TRAINING")
    print("=" * 50)

    # 80-20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training Samples: {len(X_train)} (80%)")
    print(f"Testing Samples:  {len(X_test)} (20%)")

    # Initialize classifiers with fixed random state for reproducibility
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit models on training data
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    print("Decision Tree and Random Forest models trained successfully.")

    return dt_model, rf_model, X_train, X_test, y_train, y_test


# ==========================================
# STEP 7: EVALUATE MODELS & OVERFITTING CHECK
# ==========================================

def evaluate_models(
    dt_model: DecisionTreeClassifier,
    rf_model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    """
    Evaluates both models on both Training and Testing datasets to demonstrate
    performance, overfitting analysis, classification reports, and confusion matrices.
    """
    print("\n" + "=" * 50)
    print("STEP 5: MODEL EVALUATION & OVERFITTING ANALYSIS")
    print("=" * 50)

    # Predict on both Train and Test sets
    dt_train_preds = dt_model.predict(X_train)
    dt_test_preds = dt_model.predict(X_test)

    rf_train_preds = rf_model.predict(X_train)
    rf_test_preds = rf_model.predict(X_test)

    # Calculate accuracies
    dt_train_acc = accuracy_score(y_train, dt_train_preds)
    dt_test_acc = accuracy_score(y_test, dt_test_preds)

    rf_train_acc = accuracy_score(y_train, rf_train_preds)
    rf_test_acc = accuracy_score(y_test, rf_test_preds)

    class_labels = ["Poor", "Average", "Good"]

    # 1. Decision Tree Evaluation
    print("\n" + "-" * 40)
    print("1. DECISION TREE CLASSIFIER")
    print("-" * 40)
    print(f"Training Accuracy : {dt_train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {dt_test_acc * 100:.2f}%")
    if dt_train_acc > dt_test_acc:
        print(f"Note: Decision Tree shows signs of overfitting (Train > Test gap: {(dt_train_acc - dt_test_acc) * 100:.2f}%).")

    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, dt_test_preds, labels=class_labels, zero_division=0))

    dt_cm = confusion_matrix(y_test, dt_test_preds, labels=class_labels)
    print("Decision Tree Confusion Matrix (Poor, Average, Good):")
    print(dt_cm)

    # 2. Random Forest Evaluation
    print("\n" + "-" * 40)
    print("2. RANDOM FOREST CLASSIFIER")
    print("-" * 40)
    print(f"Training Accuracy : {rf_train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {rf_test_acc * 100:.2f}%")

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_test_preds, labels=class_labels, zero_division=0))

    rf_cm = confusion_matrix(y_test, rf_test_preds, labels=class_labels)
    print("Random Forest Confusion Matrix (Poor, Average, Good):")
    print(rf_cm)

    # 3. Model Comparison Summary
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Model':<25} {'Train Accuracy':<18} {'Test Accuracy':<15}")
    print("-" * 58)
    print(f"{'Decision Tree':<25} {f'{dt_train_acc * 100:.2f}%':<18} {f'{dt_test_acc * 100:.2f}%':<15}")
    print(f"{'Random Forest':<25} {f'{rf_train_acc * 100:.2f}%':<18} {f'{rf_test_acc * 100:.2f}%':<15}")
    print("-" * 58)

    if rf_test_acc >= dt_test_acc:
        print("Conclusion: Random Forest performed better or equal in test generalization.")
    else:
        print("Conclusion: Decision Tree performed better on this test set.")


# ==========================================
# STEP 8: FEATURE IMPORTANCE
# ==========================================

def show_feature_importance(rf_model: RandomForestClassifier, feature_names: list) -> None:
    """
    Computes and plots feature importances from the Random Forest model.
    Saves the visualization to data/feature_importance.png.

    Parameters:
        rf_model (RandomForestClassifier): Fitted Random Forest model.
        feature_names (list): List of feature column names.
    """
    print("\n" + "=" * 50)
    print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)

    # Extract feature importance using Random Forest
    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance Rankings:")
    print(feature_importance_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"],
        color="#34495e"
    )
    plt.title("Feature Importance in Student Performance (Random Forest)")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("data/feature_importance.png")
    plt.close()

    print("\nFeature importance graph created successfully at 'data/feature_importance.png'!")


# ==========================================
# STEP 9: INPUT VALIDATION & PREDICTION
# ==========================================

def get_valid_numeric_input(prompt: str, minimum: float, maximum: float) -> float:
    """
    Prompts the user for numeric input with range validation and robust error handling.

    Parameters:
        prompt (str): Text prompt displayed to the user.
        minimum (float): Minimum allowed value (inclusive).
        maximum (float): Maximum allowed value (inclusive).

    Returns:
        float: Validated numeric value.
    """
    while True:
        try:
            raw_input_str = input(prompt).strip()
            value = float(raw_input_str)

            if not np.isfinite(value):
                print("Invalid input! Please enter a finite number.")
                continue

            if minimum <= value <= maximum:
                return value

            print(f"Out of range! Please enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Invalid input! Please enter a valid numeric value.")
        except (KeyboardInterrupt, EOFError):
            print("\nInput cancelled by user.")
            sys.exit(0)


def predict_student_performance(rf_model: RandomForestClassifier, feature_names: list) -> None:
    """
    Collects student academic details from user, performs prediction using
    Random Forest, and calculates prediction confidence using NumPy.

    Parameters:
        rf_model (RandomForestClassifier): Trained Random Forest model.
        feature_names (list): Expected feature column names.
    """
    print("\n" + "=" * 50)
    print("   STUDENT PERFORMANCE PREDICTION (INTERACTIVE)   ")
    print("=" * 50)

    attendance = get_valid_numeric_input("Enter Attendance (% [0 - 100]): ", 0.0, 100.0)
    previous_marks = get_valid_numeric_input("Enter Previous Marks [0 - 100]: ", 0.0, 100.0)
    study_hours = get_valid_numeric_input("Enter Study Hours per day [0 - 24]: ", 0.0, 24.0)
    assignment_marks = get_valid_numeric_input("Enter Assignment Marks [0 - 100]: ", 0.0, 100.0)
    participation = get_valid_numeric_input("Enter Participation score [1 - 10]: ", 1.0, 10.0)

    # Use Student class instance to structure input data
    student_obj = Student(
        attendance=attendance,
        previous_marks=previous_marks,
        study_hours=study_hours,
        assignment_marks=assignment_marks,
        participation=participation
    )

    new_student_df = student_obj.get_student_data()[feature_names]

    # Model prediction
    predicted_class = rf_model.predict(new_student_df)[0]

    # Meaningful NumPy usage: extract class probabilities and model confidence score
    class_probabilities = rf_model.predict_proba(new_student_df)[0]
    confidence_percentage = np.max(class_probabilities) * 100.0

    print("\n" + "-" * 40)
    print(f"PREDICTED PERFORMANCE : {predicted_class}")
    print(f"CONFIDENCE SCORE      : {confidence_percentage:.2f}%")
    print("-" * 40)

    print("Class Probabilities:")
    for class_name, probability in zip(rf_model.classes_, class_probabilities):
        print(f"  {class_name:<10}: {probability * 100:.2f}%")


# ==========================================
# STEP 10: MAIN EXECUTION FUNCTION
# ==========================================

def main():
    """
    Main function to execute the complete Student Performance Prediction pipeline.
    """
    print("=" * 60)
    print("   GR-2 STUDENT PERFORMANCE PREDICTION SYSTEM   ")
    print("   BCA Machine Learning Practical Project       ")
    print("=" * 60)

    # 1. Load and clean dataset
    df = load_and_clean_data(
        input_file_path="data/student_performance.csv",
        output_cleaned_path="data/cleaned_student_performance.csv"
    )

    # 2. Features and target preparation
    X, y, feature_names = prepare_data(df)

    # 3. Meaningful NumPy feature statistics
    display_numpy_feature_statistics(df, feature_names)

    # 4. Exploratory Data Analysis (EDA)
    perform_eda(df)

    # 5. Train models
    dt_model, rf_model, X_train, X_test, y_train, y_test = train_models(X, y)

    # 6. Evaluate models (Train vs Test, Overfitting, Confusion Matrix)
    evaluate_models(dt_model, rf_model, X_train, X_test, y_train, y_test)

    # 7. Show and save feature importance
    show_feature_importance(rf_model, feature_names)

    # 8. Interactive Student Performance Prediction
    predict_student_performance(rf_model, feature_names)

    print("\nProject execution completed successfully!")


if __name__ == "__main__":
    main()