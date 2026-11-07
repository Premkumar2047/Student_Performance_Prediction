# GR-2 Student Performance Prediction System 🎓📊

A Machine Learning project developed for the **BCA Machine Learning Curriculum**. This system predicts a student's academic performance tier (**Good**, **Average**, or **Poor**) based on key academic factors using supervised classification models.

---

## 📌 Project Overview
Academic institutions track multiple performance metrics to identify students who may need extra academic support. This project demonstrates an end-to-end Machine Learning workflow:
1. **Data Preprocessing & Cleaning**: Handling missing values, removing duplicate entries, and data validation using Pandas.
2. **NumPy Feature Profiling**: Calculating summary statistics (Mean, Standard Deviation, Median, Min, Max) across academic metrics.
3. **Exploratory Data Analysis (EDA)**: Visualizing distributions, scatter relationships, and correlation matrices using Matplotlib.
4. **Supervised Model Training**: Training **Decision Tree** and **Random Forest** classifiers using 80/20 stratified train-test splits.
5. **Overfitting Analysis & Model Evaluation**: Comparing training vs testing accuracy, precision, recall, F1-score, and confusion matrices.
6. **Feature Importance Analysis**: Identifying which factors impact student performance the most.
7. **Interactive Prediction CLI**: Predicting a student's performance tier along with model confidence percentage.

---

## 📂 Project Directory Structure

```text
Student_Performance_Prediction/
│
├── data/
│   ├── student_performance.csv             # Generated raw dataset (200 records)
│   ├── cleaned_student_performance.csv     # Cleaned dataset after preprocessing
│   ├── performance_distribution.png        # Bar chart: class distribution
│   ├── attendance_vs_performance.png       # Scatter plot: attendance vs score
│   ├── study_hours_vs_performance.png      # Scatter plot: study hours vs score
│   ├── correlation_heatmap.png             # Correlation heatmap of features
│   └── feature_importance.png              # Random Forest feature ranking chart
│
├── generate_dataset.py                     # Synthetic dataset generation script
├── main.py                                 # Main ML pipeline and interactive predictor
├── requirements.txt                        # Required Python packages
├── .gitignore                              # Git exclusion file
└── README.md                               # Project documentation
```

---

## 📊 Dataset Schema

The dataset contains **200 synthetic student records** with the following academic features:

| Column Name | Data Type | Description |
|---|---|---|
| `Student_ID` | Integer | Unique identifier for each student (excluded from ML modeling) |
| `Attendance` | Integer | Class attendance percentage (40% to 100%) |
| `Previous_Marks` | Integer | Marks obtained in previous semester exams (30 to 100) |
| `Study_Hours` | Float | Self-study hours per day (1.0 to 8.0 hours) |
| `Assignment_Marks` | Integer | Continuous evaluation / internal marks (30 to 100) |
| `Participation` | Integer | Active classroom engagement rating (1 to 10 scale) |
| `Performance_Score` | Float | Weighted aggregate score |
| `Performance` | Categorical | Target class: **Good** (≥ 88), **Average** (≥ 55), **Poor** (< 55) |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Student_Performance_Prediction.git
cd Student_Performance_Prediction
```

### 2. Create and Activate Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Generate Dataset
```bash
python generate_dataset.py
```
*Generates 200 student records and saves to `data/student_performance.csv`.*

### Step 2: Run Main Machine Learning Pipeline
```bash
python main.py
```
*Executes data cleaning, NumPy statistical profiling, EDA visualizations, model training, evaluation, and starts the interactive CLI prediction.*

---

## 📈 Machine Learning Results & Comparison

### Model Comparison Summary

| Model | Training Accuracy | Testing Accuracy | Overfitting Observation |
|---|---|---|---|
| **Decision Tree Classifier** | 100.00% | 75.00% | Suffers from overfitting (25% generalization gap) |
| **Random Forest Classifier** | 100.00% | 87.50% | Significantly higher test generalization |

### Feature Importance Rankings (Random Forest)
1. **Study Hours per day**: ~41.1% (Highest predictor of student success)
2. **Previous Marks**: ~19.4%
3. **Assignment Marks**: ~15.8%
4. **Attendance**: ~13.5%
5. **Class Participation**: ~10.3%

---

## 💻 Interactive Prediction Example

```text
==================================================
   STUDENT PERFORMANCE PREDICTION (INTERACTIVE)   
==================================================
Enter Attendance (% [0 - 100]): 85
Enter Previous Marks [0 - 100]: 80
Enter Study Hours per day [0 - 24]: 5
Enter Assignment Marks [0 - 100]: 90
Enter Participation score [1 - 10]: 8

----------------------------------------
PREDICTED PERFORMANCE : Good
CONFIDENCE SCORE      : 79.00%
----------------------------------------
Class Probabilities:
  Average   : 21.00%
  Good      : 79.00%
  Poor      : 0.00%
```

---

## 🛠️ Technologies Used
- **Python 3.x**
- **NumPy**: Vectorized mathematics, feature profiling, confidence score calculation.
- **Pandas**: Data structures, data cleaning, CSV export/import.
- **Matplotlib**: Statistical charts, scatter plots, correlation heatmaps.
- **Scikit-Learn**: Decision Tree & Random Forest models, train-test splitting, metrics.

---

## 📝 Author & Acknowledgements
- **Course**: Bachelor of Computer Applications (BCA) - Machine Learning Lab
- Developed as an academic practical project demonstrating Supervised Classification.
