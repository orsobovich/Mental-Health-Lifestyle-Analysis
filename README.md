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
```

---

## Key Stages & Methodology

The analysis is orchestrated by `main.py` and follows these stages:

### 1. Data Import & Cleaning
- Missing numerical values handled via mean imputation.
- Rows with missing categorical variables removed.
- Outliers detected using Z-score thresholds (|Z| > 3).
- Duplicate records removed to preserve data integrity.

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics and group summaries.
- Distribution visualization using histograms and boxplots.
- Preliminary checks informing the choice of statistical tests.

### 3. Statistical Modeling
- **One-Way ANOVA:** Used to detect group differences across categorical variables.
- **Planned Contrasts:** A priori contrasts defined to test theory-driven hypotheses with increased statistical power.
- **Spearman Correlation:** Applied to ordinal–continuous relationships (e.g., stress level vs. sleep hours).

### 4. Visualization
- Comparative boxplots for group-level differences.
- Correlation heatmaps for exploratory insights.
- Visual outputs support interpretation but do not replace statistical testing.

---

## Data Description

- **Dataset Name:** Mental Health and Lifestyle Habits (2019–2024)
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-lifestyle-habits-2019-2024/data
- **Size:** ~3,000 rows, 12 variables

### Key Variables
- Diet Type
- Happiness Score
- Sleep Hours
- Stress Level
- Mental Health Condition
- Social Interaction Score

---

## How to Run the Project

### 1. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
python main.py
```

---

## Expected Output

When you run `python main.py`, the pipeline generates the following outputs:

### Console Outputs
1. **Data Cleaning Summary:**
   - Number of missing values imputed
   - Number of outliers detected and removed
   - Number of duplicate rows removed
   - Final dataset dimensions

2. **Exploratory Data Analysis:**
   - Dataset shape and data types
   - Descriptive statistics (mean, median, std, min, max) for numerical variables
   - Group-level summaries by categorical variables

3. **Statistical Test Results:**
   - **ANOVA Results:** F-statistic, p-value, and interpretation for each hypothesis
   - **Planned Contrasts:** t-statistics, p-values, and effect sizes for theory-driven comparisons
   - **Spearman Correlations:** Correlation coefficients and significance levels


### 4. Output & Results

Upon execution, the pipeline generates a comprehensive report comprising **13 interactive figures**, organized into four analytical phases. 

Uniquely, this pipeline renders statistical tables (ANOVA results, Contrast Weights) directly as images for easier reporting.

#### Phase 1: Data Integrity & Exploration
* **Figure 1: Hybrid Cleaning Report** - Visualizes the missing values status and the data retention rate after the hybrid cleaning process.
* **Figure 2: Categorical Distributions** - Pie charts displaying demographic and lifestyle breakdowns (Diet, Gender, Country, etc.).
* **Figure 3: Numerical Distributions** - Histograms with Kernel Density Estimation (KDE) overlays for variables like Age, Sleep, and Work Hours.

#### Phase 2: Correlation Analysis
* **Figure 4: Correlation Heatmap** - A generic heatmap displaying Spearman correlations between all numerical variables.
* **Figure 5: Linear Regression Plot** - Visualizes the specific relationship between Happiness Score and Social Interaction Score (with confidence intervals).

#### Phase 3: Hypothesis 1 (Diet & Happiness)
This phase combines visual evidence with generated statistical tables:
* **Figure 6: Diet vs. Happiness Boxplots** - Distribution of happiness scores across diet types, overlaid with raw data points (stripplot).
* **Figure 7:** Descriptive Statistics Table (Mean/Std Happiness by Diet).
* **Figure 8:** ANOVA Results Table (F-statistic & p-value).
* **Figure 9:** Contrast Weights Table (Showing the specific weights used for the "Vegetarian/Vegan vs. Others" hypothesis).

#### Phase 4: Hypothesis 3 (Mental Health & Socializing)
Analysis of social interaction scores across mental health conditions:
* **Figure 10: Social Interaction Boxplots** - Comparison between healthy individuals ('None') and those with diagnosed conditions.
* **Figure 11:** Descriptive Statistics Table (Mean/Std Social Score by Condition).
* **Figure 12:** ANOVA Results Table.
* **Figure 13:** Contrast Weights Table (Showing weights for "Healthy vs. Diagnosed").

### Example Output Snippet
```
=== DATA CLEANING REPORT ===
Missing values imputed: 127
Outliers removed: 43
Duplicate rows removed: 8
Final dataset: 2,822 rows × 12 columns

=== HYPOTHESIS 1: Diet & Happiness ===
ANOVA F-statistic: 12.45, p-value: 0.0001
Planned Contrast (Plant-based vs Others):
  t = 3.21, p = 0.0013, Cohen's d = 0.42
Result: Plant-based diets significantly associated with higher happiness (p < 0.05)

=== HYPOTHESIS 2: Sleep & Stress ===
Spearman correlation: ρ = -0.58, p < 0.0001
Result: Strong negative correlation confirmed

=== HYPOTHESIS 3: Mental Health & Social Interaction ===
ANOVA F-statistic: 8.92, p-value: 0.0028
Result: Significant difference in social interaction by mental health status
```

---

## References
- Dataset: Soundankar, A. (2024). Mental Health and Lifestyle Habits (2019-2024). *Kaggle*. https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-lifestyle-habits-2019-2024/data