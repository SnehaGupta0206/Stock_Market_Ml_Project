# Forecasting Indian Stock Market Indices using Machine Learning

An end-to-end data science project that studies how two major NSE indices, **NIFTY 50** and **NIFTY MIDCAP**, behaved from 2013 to 2024, tests which technical signals carry real information, and builds classification models to predict whether the next trading day closes **UP or DOWN**.

The project is framed as a case study: a research firm wants to replace its opinion-driven "Daily Market Outlook" with evidence-backed views, and asks a data-science intern to *explain how the indices behave, prove which signals work, and say where tomorrow's close is heading.*

---

## Table of Contents

- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Repository Structure](#repository-structure)
- [Workflow](#workflow)
- [Model Results](#model-results)
- [Installation and Usage](#installation-and-usage)
- [Limitations and Future Work](#limitations-and-future-work)
- [Tech Stack](#tech-stack)
- [Acknowledgements](#acknowledgements)

---

## Key Findings

| # | Objective | Finding |
|---|-----------|---------|
| 1 | Wealth creation | Rs.100 invested in 2013 grew by about **+593% in MIDCAP** versus **+293% in NIFTY 50** (roughly 6.9x vs 3.9x). Mid-caps built the most wealth. |
| 2 | Risk | MIDCAP is about **32% riskier** on a daily basis (daily volatility 1.37% vs 1.04%). The worst single day was **20 March 2020** (COVID crash): MIDCAP fell 14.9%. |
| 3 | Volume | Correlation between previous-day volume and the next return is essentially zero (MIDCAP 0.05, NIFTY 50 -0.005). Heavy volume alone does **not** predict the next move. |
| 4 | Indicators | Price-level indicators (SMA, EMA, SAR) track the closing price almost perfectly (correlation about 1.0) but only because they are built from price. Momentum indicators (RSI, Stochastic, PSY) have near-zero correlation with the price level and are better suited to predicting direction. |
| 5 | Prediction | The best model, **Logistic Regression, reached 61.3% test accuracy**, only marginally above the 60.9% obtained by always predicting UP. Daily direction is genuinely hard to predict. |

> **Honest takeaway:** the models found little reliable edge over a naive "always UP" guess. See [Model Results](#model-results) for the detail.

---

## Dataset

- **Source:** *Forecasting of Indian Stock Market Indices*, Mendeley Data, [doi:10.17632/swrw7d78ss.1](https://data.mendeley.com/datasets/swrw7d78ss/1), licensed **CC BY 4.0**.
- **Coverage:** daily data, 2013 to 2024, for NIFTY 50, NIFTY BANK and NIFTY MIDCAP.
- **Columns (20 per day):** previous-day Open, High, Low, Close and Volume, plus 14 technical indicators (SMA_6, EMA_12, EMA_26, MACD, MACD_Signal, RSI_6, RSV_9, PSY_13, K_9, D_3, OBV, SAR, CMF). The target is today's close, `Close_(t)`.

### Data quality decisions

- **NIFTY BANK was excluded.** Its file failed a basic validation check: the daily High was lower than the daily Low in 2,204 of 2,265 rows (about 97%), which indicates a source error. The analysis therefore covers **NIFTY 50 and NIFTY MIDCAP**.
- Rows with missing values appeared only at the start of each series (indicators need a warm-up period). These **19 rows per index were dropped** rather than filled with artificial values.
- A stray space in the `Open _(t-1)` column name was fixed, and text dates were converted to proper datetimes.

---

## Project Objectives

1. **Wealth creation:** which index rewarded investors most over 2013 to 2024?
2. **Risk personality:** which index is riskier and when does risk spike?
3. **Volume:** does heavy trading volume signal the next price move?
4. **Indicators:** which technical indicators carry real signal and which are noise or redundant?
5. **Forecasting:** how accurately can the next day's direction be predicted, and with which model?

The full case study is in [`case_study_and_objectives.txt`](case_study_and_objectives.txt).

---

## Repository Structure

```
Stock_Market_Ml_Project/
├── Dashboard/                       # Dashboard files
├── data/
│   ├── raw/                         # Original Excel files from Mendeley Data
│   └── processed/                   # Cleaned and model-ready CSV files
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_classification_models.ipynb
├── case_study_and_objectives.txt
├── requirements.txt
└── README.md
```

---

## Workflow

### 1. Data cleaning (`01_data_cleaning.ipynb`)
Loads the three raw Excel files, standardises column names and date formats, validates High and Low values, removes the faulty NIFTY BANK file, drops warm-up rows, and saves clean files:
`nifty50_clean.csv`, `midcap_clean.csv` and the combined `all_indices_clean.csv` (5,904 rows).

### 2. Exploratory data analysis (`02_EDA.ipynb`)
Answers Objectives 1 to 4 with visuals and statistics: price trends, growth of Rs.100 invested, daily-return distributions, 30-day rolling volatility, volume versus return, UP/DOWN day share, and a correlation heatmap of the indicators.

### 3. Feature engineering (`03_feature_engineering.ipynb`)
Builds six new features, all using only information available *before* the day being predicted (no data leakage):

| Feature | Meaning |
|---------|---------|
| `Prev_Return_%` | Previous day's return (momentum) |
| `Prev_Range_%` | Previous day's High-Low range as a % of close |
| `Volatility_30d` | 30-day rolling standard deviation of past returns |
| `MA_Distance_%` | Distance of previous close from its 6-day average |
| `MACD_Hist` | MACD minus its signal line |
| `DayOfWeek` | Calendar day (0 = Monday) |

It also creates the target `Direction` (1 = closed UP, 0 = closed DOWN; 56% of days are UP), drops 60 further warm-up rows, and saves `model_ready.csv` (5,844 rows x 28 columns).

### 4. Classification models (`04_classification_models.ipynb`)
Selects 15 momentum and volatility features, performs a **chronological** 85/15 train-test split (`shuffle=False`, so models learn from the past and are tested on the most recent data), scales features with `StandardScaler`, and trains five classifiers.

---

## Model Results

Training set: 4,967 rows. Test set: 877 rows (534 UP days, 343 DOWN days).

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | Logistic Regression | **61.3%** |
| 2 | SVM (linear kernel) | 60.9% |
| 3 | Naive Bayes | 58.3% |
| 4 | Random Forest (100 trees) | 54.7% |
| 5 | Decision Tree | 53.9% |

**How to read these numbers**

- In the test period, 60.9% of days (534 of 877) were UP. A model that **always predicts UP** therefore scores 60.9%. Logistic Regression beats this by only 0.4 percentage points, and the linear SVM matches it exactly because it predicted UP for every day.
- Logistic Regression identified UP days well (recall 0.89) but caught only 18% of DOWN days (recall 0.18).
- The Decision Tree scored 100% on training data against 53.9% on test data, a clear case of overfitting. Random Forest also performed below the naive baseline.
- Conclusion: with daily technical indicators alone, next-day direction is very hard to forecast. The results are best read as evidence of how little signal these inputs contain.

---

## Installation and Usage

```bash
# 1. Clone the repository
git clone https://github.com/SnehaGupta0206/Stock_Market_Ml_Project.git
cd Stock_Market_Ml_Project

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and run the notebooks in order
jupyter notebook notebooks/
```

Run the notebooks in order (**01 to 04**), because each one reads the file produced by the previous one. Place the original Excel files in `data/raw/` before starting:

- `Features NIFTY 50_ALL1.xlsx`
- `Features NIFTY BANK_ALL.xlsx`
- `Features NIFTY MIDCAP_ALL.xlsx`

---

## Limitations and Future Work

- **Only direction is modelled.** Next-day closing *price* (regression) is not yet forecast.
- **Accuracy alone is misleading** on an imbalanced target. Future work should report precision, recall, F1 and ROC-AUC, and compare every model against the always-UP baseline.
- **Class imbalance:** try `class_weight="balanced"`, threshold tuning, or resampling so DOWN days are detected.
- **Validation:** replace the single split with walk-forward (time-series) cross-validation and hyperparameter tuning.
- **Both indices are pooled** in one model; separate per-index models could behave differently.
- **NIFTY BANK is missing** because of the source data fault; sourcing a corrected file would complete the three-index comparison.
- **Richer inputs:** macro indicators, global market cues, news sentiment, and gradient-boosting or sequence models (XGBoost, LSTM).
- **No transaction costs or trading strategy** have been backtested.

> This project is for educational purposes only and is **not financial advice**.

---

## Tech Stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, openpyxl, Jupyter.

---

## Acknowledgements

Dataset: *Forecasting of Indian Stock Market Indices*, Mendeley Data, doi:10.17632/swrw7d78ss.1, CC BY 4.0.

---

## Author

**Sneha Gupta** · [GitHub](https://github.com/SnehaGupta0206)