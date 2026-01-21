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
