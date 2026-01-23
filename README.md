# Mental Health & Lifestyle Analysis Pipeline

## Project Description
This project implements a robust data science pipeline to analyze the relationship between lifestyle choices (diet, sleep patterns, social interaction) and mental wellbeing (stress, happiness, diagnosed conditions). 

The analysis moves beyond simple correlations, utilizing **One-Way ANOVA**, **Planned Contrasts (A priori)**, and **Spearman Rank Correlations** to test specific theoretical hypotheses.

### Main Objectives
1.  **Data Integrity:** Implement a "Hybrid Cleaning Strategy" to handle missing data without biasing the sample.
2.  **Hypothesis Testing:** Statistically validate specific claims regarding diet quality, sleep duration, and social behavior.
3.  **Visualization:** Provide a comprehensive visual report of distributions, correlations, and cleaning impacts.

### Research Hypotheses
1.  **Diet & Happiness:** Plant-based diets (Vegan/Vegetarian) are associated with higher happiness scores compared to other diets (Junk, Balanced, Keto).
2.  **Sleep & Stress:** Higher stress levels are negatively correlated with average sleep hours.
3.  **Mental Health & Socializing:** Individuals with diagnosed mental health conditions exhibit significantly different social interaction levels compared to healthy individuals.

---

## Folder & Module Structure

```text
├── main.py                   # Entry point: Orchestrates the entire analysis pipeline
├── requirements.txt          # List of external Python dependencies
├── Mental_Health_Lifestyle_Dataset.csv  # Raw Dataset
├── README.md                 # Project Documentation
├── src/                      # Source Code Modules
│   ├── __init__.py
│   ├── utils.py              # Logger configuration & file handling
│   ├── data_cleaning.py      # Logic for Hybrid Cleaning, Outliers, and Duplicates
│   ├── exploration.py        # EDA logic (Data info, descriptive stats)
│   ├── correlation.py        # Statistical correlation tests (Pearson/Spearman)
│   ├── ANOVA.py              # Logic for One-Way ANOVA and Planned Contrasts (OLS)
│   └── visualization.py      # Generation of plots (Boxplots, Heatmaps, Pies)
└── tests/                    # Unit Tests
    ├── test_data_cleaning.py # Tests for imputation and outlier removal
    ├── test_correlation.py   # Tests for correlation logic and edge cases
    ├── test_ANOVA.py         # Tests for OLS models and contrast weights
    └── test_exploration.py   # Tests for data summary functions


This structure separates statistical logic, research questions, and execution flow, ensuring clarity and maintainability.

Key Stages & Methodology

The analysis is orchestrated by main.py and follows these stages:

1. Data Import & Cleaning

Missing numerical values handled via mean imputation.

Rows with missing categorical variables removed.

Outliers detected using Z-score thresholds (|Z| > 3).

Duplicate records removed to preserve data integrity.

2. Exploratory Data Analysis (EDA)

Descriptive statistics and group summaries.

Distribution visualization using histograms and boxplots.

Preliminary checks informing the choice of statistical tests.

3. Statistical Modeling

One-Way ANOVA:
Used to test group differences for diet type and mental health condition.

Planned Contrasts:
A priori contrasts implemented to test theory-driven comparisons (e.g., plant-based diets vs. all others).

Spearman Correlation:
Used for ordinal–continuous relationships (stress level vs. sleep hours).

4. Visualization

Boxplots for group comparisons.

Correlation heatmaps for exploratory insight.

Visual outputs support interpretation but do not replace statistical testing.

Key Definitions & Parameters

Significance Level (α): 0.05 for all hypothesis tests.

Planned Contrast:
A statistical comparison defined before analysis to test a specific theoretical hypothesis.

Social Interaction Score:
A numeric scale representing frequency and quality of social engagement.

Hybrid Cleaning Strategy:
A combination of imputation (numeric data) and row removal (categorical data) to balance bias and sample size.

Data Description

Dataset Name: Mental Health and Lifestyle Habits (2019–2024)

Size: ~3,000 rows, 12 variables

Source: Kaggle
https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-lifestyle-habits-2019-2024/data

Key Variables

Diet Type

Sleep Hours

Stress Level

Happiness Score

Mental Health Condition

Social Interaction Score

How to Run the Project
1. Clone the Repository
git clone https://github.com/orsobovich/mental-health-lifestyle-analysis.git
cd mental-health-lifestyle-analysis

2. Create and Activate Virtual Environment
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Analysis
python main.py


This will:

Execute all analyses

Print statistical results to the console

Display generated plots

5. Run Tests (Optional)
pytest

References

Pandas & NumPy — data manipulation and numerical operations

SciPy — statistical tests and correlations

Statsmodels — OLS regression, ANOVA, and planned contrasts

Field, A. (2018). Discovering Statistics Using Python. Sage Publications.