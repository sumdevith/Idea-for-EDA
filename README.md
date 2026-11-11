# EDA Playbook: A Step-by-Step Guide for Data Analysts

A practical cheat sheet for **Exploratory Data Analysis (EDA)**.
Use it as a checklist when you start a new dataset, or as a source of ideas for "what should I check next?"

> **Credit:** These are my personal study notes, rewritten in my own words, from the course
> *Exploratory Data Analysis & Unsupervised Learning* (Institute of Technology of Cambodia,
> Department of Applied Mathematics and Statistics). The Titanic and Gapminder examples come from that course.
> The extra code in `eda_toolkit.py` is my own.

---

## Contents

1. [What is EDA and why do it?](#1-what-is-eda-and-why-do-it)
2. [The EDA workflow at a glance](#2-the-eda-workflow-at-a-glance)
3. [Step 1: Know your data (sources, format, types)](#3-step-1-know-your-data)
4. [Step 2: Check data quality](#4-step-2-check-data-quality)
5. [Step 3: Univariate analysis](#5-step-3-univariate-analysis-one-column-at-a-time)
6. [Step 4: Preprocessing](#6-step-4-preprocessing)
7. [Step 5: Bivariate analysis](#7-step-5-bivariate-analysis-two-columns-at-a-time)
8. [Step 6: Show more than two variables](#8-step-6-show-more-than-two-variables-in-one-chart)
9. [Visualization checklist](#9-visualization-checklist)
10. [Final checklist before modeling](#10-final-checklist-before-modeling)
11. [Quick start with the code](#11-quick-start-with-the-code)
12. [Further reading](#12-further-reading)

---

## 1. What is EDA and why do it?

**GIGO: Garbage In, Garbage Out.** If the data is bad, the result is bad, even with a very good model.

EDA means getting curious about your data. It helps you to:

- **Inspect** the data (size, types, missing values)
- **Detect** problems (outliers, duplicates, wrong values)
- **Uncover** patterns and relationships
- **Find** new questions to investigate
- **Prepare** for modeling (check assumptions, select features, choose methods)

Typical problems in real data: missing values, outliers, and inconsistent entries (`"Male"`, `"male"`, `"M"`).

**Descriptive vs inferential statistics**

| Descriptive (used in EDA) | Inferential |
|---|---|
| Summarizes the sample you have | Generalizes to the population |
| Tables, graphs, mean/median/spread | Probability, hypothesis tests, prediction |

---

## 2. The EDA workflow at a glance

```
Problem → Data → EDA → Model → Validation & Conclusion
                 │
   ┌─────────────┼──────────────────────────────────────┐
   1. Know data  2. Data quality  3. Univariate  4. Preprocess  5. Bivariate  6. Multi-variable
```

| Step | Main question | Main tools |
|---|---|---|
| 1. Know your data | Where does it come from? What type is each column? | `df.info()`, `df.dtypes` |
| 2. Data quality | Can I trust this data? | 6 quality dimensions |
| 3. Univariate | What does each column look like alone? | Frequency tables, `describe()`, boxplot, histogram |
| 4. Preprocess | How do I fix the problems? | Drop, impute, cap, transform, encode |
| 5. Bivariate | Which columns are related? | Correlation, η², χ², Cramér's V |
| 6. Multi-variable | Can one chart tell the whole story? | Color, shape, size, facets |

---

## 3. Step 1: Know your data

### Data sources

| | Primary | Secondary |
|---|---|---|
| Meaning | Collected by you, for a specific purpose | Already collected by others |
| Examples | Surveys, interviews, observations, experiments | Government reports, Kaggle, books, NGO/industry reports |
| Typical risk | Small sample, bias in collection | Inconsistent formats, **missing values** (the most common problem) |

### Data format

- **Structured:** tables with rows and columns (Excel, CSV, SQL).
- **Unstructured:** free-form (text, images, audio, video, web pages).

### Data types: choose the right statistics and chart

```
Data
├── Qualitative (categorical)
│   ├── Nominal  : no order        (gender, blood type, country)
│   └── Ordinal  : has an order    (education level, rating 1-5, Low/Medium/High)
└── Quantitative (numerical)
    ├── Discrete   : counts        (number of children, visits)
    └── Continuous : any value     (height, weight, temperature, fare)
```

**Test:** can you do `+`, `−`, and meaningful comparison?
`Age 50 > 30` and `25 − 20 = 5` make sense → quantitative.
`"S" + "C"` does not → qualitative.

> ⚠️ Numbers are not always quantitative. `Survived` (0/1) and `Pclass` (1/2/3) are **categories** stored as numbers. Convert them (`astype("category")`).

---

## 4. Step 2: Check data quality

Six dimensions. Check all of them before analysis.

| Dimension | Question | Example problem | Usual fix |
|---|---|---|---|
| **Timeliness** | Is it up to date for my purpose? | Temperatures from the 1960s used to forecast tomorrow | Update the data |
| **Uniqueness** | Any accidental duplicates? | Same patient recorded twice; same row in train and test set (biased model) | Drop duplicates |
| **Validity** | Are values inside the valid range? | Height = 0, negative cholesterol | Remove / treat as missing (needs domain knowledge) |
| **Consistency** | Same format everywhere? | `15/03/2004` vs `03/15/2004`; `Male` vs `M`; cm vs m in one column | Convert / transform |
| **Accuracy** | Does it reflect what it should measure? | Survey filled with confusion or jokes. **Hardest to detect.** | Remove if proven wrong |
| **Completeness** | Any missing values? | `NaN`, `NA`, `null`, or **disguised** as `0`, `-1`, `-999` | Impute or drop (see Step 4) |

**Pro tip:** always look for *missing values in disguise*. A child's height of `0` is not a height, it is a missing value.

---

## 5. Step 3: Univariate analysis (one column at a time)

Goal: **describe, understand, and detect problems** in each variable separately.

### 5.1 Qualitative columns

| Statistic | Meaning |
|---|---|
| Absolute frequency | Count of each category |
| Relative frequency | Proportion / percentage of each category |
| Mode | Category with the highest count |

**Charts:** bar plot / count plot (best default), pie chart (only for few categories).

### 5.2 Quantitative columns

**Center**
- **Mean:** average, **sensitive to outliers**.
- **Median (Q2):** middle value, robust.
- **Quick check:** if `mean ≫ median` → right-skewed data or extreme large values (e.g. Titanic `Fare`: mean 32.2, median 14.45).

**Quantiles**
- Q1 = 25th percentile, Q2 = median = 50th, Q3 = 75th.
- Sort the data. If `n` is odd, the median is the middle term. If `n` is even, it is the average of the two middle terms. Q1 and Q3 are the medians of the lower and upper halves.

**Spread**
- Variance and standard deviation: how far points are from the mean.
- IQR = Q3 − Q1: the spread of the middle 50% of the data (robust).

**`describe()` reading guide**

| Look at | To find |
|---|---|
| mean vs median (50%) | Skewness or extreme values |
| std | How spread out the data is |
| min / max | **Invalid values** (age 0.42? fare 0? max 512?) |
| 25% / 75% | Where most data sits |

**Charts**

| Chart | Shows | Notes |
|---|---|---|
| **Boxplot** | Quartiles, range, outliers | Box = Q1 to Q3, line = median, whiskers reach `Q1 − 1.5·IQR` and `Q3 + 1.5·IQR`, points beyond = outliers |
| **Histogram** | Shape, skewness, gaps | Bar height = count in each bin. Try different bin widths |
| **KDE plot** | Smooth version of the histogram | Easier to see main ranges and patterns |

**Tip:** for strongly skewed positive data (fare, income, population), look at the **log scale** as well.

---

## 6. Step 4: Preprocessing

### 6.1 Missing values: first find the type

| Type | Meaning | Example | Drop affects other columns? |
|---|---|---|---|
| **MCAR** (Missing Completely At Random) | Missingness is pure chance, unrelated to anything | System glitch | No |
| **MAR** (Missing At Random) | Related to *other observed columns*, not to its own value | Height missing more often for one gender | **Yes** |
| **MNAR** (Missing Not At Random) | Related to the *missing value itself* | Very high or very low incomes not reported | Yes. Needs domain knowledge |

**How to test quickly:** drop the rows with NA, then compare the distribution of the *other* columns before and after.
Almost no change → maybe MCAR. Big change → MAR/MNAR.

> ⚠️ MCAR does not mean the *variable* is unrelated to other columns. Only the *missingness* is random.

### 6.2 Rules of thumb (percentage of NA in a column)

| % NA | Suggested action |
|---|---|
| < 5% | Drop the rows |
| 5–10% | Can drop if the sample is large, but be careful about the missing type |
| 10–20% | Better to impute, according to the type |
| 20–30% | Remove the column if it is not important |
| > 30% | Remove the column |

### 6.3 Strategy by type

| Type | What to do |
|---|---|
| MCAR | < 5%: drop. Otherwise simple imputation (median/mode) |
| MAR | Do **not** drop if important or many. Use correlations → **KNN** or **MICE (IterativeImputer)** |
| MNAR | Trickiest. Use expertise, flag the missingness (add an `is_missing` column), run a sensitivity analysis |

**Simple imputation:** use the **median** when outliers exist (the mean is pulled by them).
Note: it creates a tall spike at the median in the histogram. That is a side effect to be aware of.
**Model-based imputation:** pick predictors by looking at correlations first (e.g. Titanic `Age` is more related to `Fare` and family size than to categorical columns).

### 6.4 Drop rows vs drop columns

- Few NA spread over many columns → drop **rows**.
- One column with most values missing (e.g. Titanic `Cabin`: 327 missing out of 418 in the test set) → drop the **column**.
- Handle the issues from smallest to biggest.

### 6.5 Outliers, leverage points, influential points

| Term | Definition | Detect with |
|---|---|---|
| **Outlier** | Extreme **y** (far from the regression line) | IQR rule, z-score, residuals (> 3 std) |
| **High leverage point** | Extreme **x** (far from the mean of predictors) | Hat values, Mahalanobis distance |
| **Influential point** | Outlier + high leverage → changes the model a lot | Cook's distance, DFFITS |

For a **normal** distribution about 99.3% of values fall within `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`, so about 0.7% outside are flagged as outliers.

Not every outlier is a problem. **Influential points** are the ones that really change the analysis.

**Handling options**
1. **Check first:** is it a data-entry error or a real value? Real values should be kept or treated carefully.
2. **Capping (winsorizing):** limit extreme values to a threshold.
3. **Trimming:** remove them completely.
4. **Transform:** z-score `(x − mean)/std`, min-max `(x − min)/(max − min)`, `log(x)` or `√x` (positive data only).
5. **Robust methods:** robust regression (e.g. Huber loss), rank-based statistics.

> There is no absolute solution. It depends on the analysis.

### 6.6 Encoding categorical variables

**One-hot encoding** turns each category into a 0/1 column. It is needed for parametric models such as linear regression.
`['Asia','Europe','Africa']` → columns `Africa, Americas, Asia, Europe, Oceania` with a single `1` per row.

---

## 7. Step 5: Bivariate analysis (two columns at a time)

First decide the **case** by the types of the two columns:

| Case | Indicator | Chart |
|---|---|---|
| **Quantitative vs Quantitative** | Covariance, **Pearson**, **Spearman** | Scatter plot, correlation heatmap |
| **Quantitative vs Qualitative** | **η² (eta-squared)** | Box plot / violin plot per group, conditional histogram/KDE |
| **Qualitative vs Qualitative** | **χ² test**, **Cramér's V** | Stacked / grouped bar plot, treemap / mosaic plot |
| Time series | Trend over time | Line plot (quantitative), evolution bar plot (qualitative) |

### 7.1 Quantitative vs Quantitative

- **Covariance:** sign shows direction (positive = move together, negative = move opposite). The size depends on units, so it is hard to compare.
- **Pearson correlation `r`** (between −1 and 1):
  - `r → 1`: positive linear relationship; `r → −1`: negative; `r ≈ 0`: no *linear* trend (or noise).
  - Helps pick useful inputs for linear models and find **redundant** inputs (very high correlation between two predictors).
  - ⚠️ **Correlation is not causation.**
- **Spearman rank correlation:** Pearson computed on ranks.

| | Pearson | Spearman |
|---|---|---|
| Measures | Linear relationship | Monotonic relationship (always up or always down) |
| Data type | Numerical | Ordinal or numerical (e.g. Likert scale) |
| Outliers | Sensitive | Less sensitive |
| Type | Parametric | Non-parametric |

**Tip:** if Spearman is much larger than Pearson (e.g. GDP vs life expectancy: 0.86 vs 0.68), the relation is **monotonic but not linear**, or outliers hurt Pearson. Try a log scale on the scatter plot.

### 7.2 Quantitative vs Qualitative: η² (eta-squared)

Idea: if the category is related to the number, the group averages should be **far apart** and the values inside each group **not too spread out**.

```
η² = BSS / TSS          (between-group sum of squares / total sum of squares)
0 → no relation       1 → strong relation
```

Example from the course: continent explains life expectancy (η² ≈ 0.64) better than GDP per capita (η² ≈ 0.42).
It is the basis of ANOVA. Like Pearson, it is **sensitive to outliers**.
Always confirm with side-by-side **box/violin plots**.

### 7.3 Qualitative vs Qualitative: χ² and Cramér's V

1. Build the **contingency table** (counts for each pair of categories).
2. Compare **observed** counts with **expected** counts if the two variables were independent.
3. **χ² test:** H₀ = independent. A small p-value → reject H₀ → the variables are associated.
4. **Cramér's V** (0 to 1) measures the *strength*:

```
V = sqrt( χ² / ( n · min(r − 1, k − 1) ) )
```

The plain V is biased upward. Use the **bias-corrected** version (`cramers_v(..., bias_correction=True)`).

> With a very large sample, χ² gives a tiny p-value even for a weak association. Always report **Cramér's V** too.

### Quick reference: "Which indicator for which pair?"

| Column A | Column B | Use |
|---|---|---|
| Quantitative | Quantitative | Pearson (linear) / Spearman (monotonic, outliers, ordinal) |
| Ordinal | Ordinal / Quantitative | Spearman |
| Quantitative | Nominal | η² |
| Nominal | Nominal | χ² + Cramér's V |

---

## 8. Step 6: Show more than two variables in one chart

| Extra variable | Use | Example |
|---|---|---|
| Qualitative | **Color** (discrete) or **symbol/shape** | Color = continent, symbol = GDP category |
| Quantitative | **Color gradient** or **size** (bubble chart) | Size = population |
| Time | Animation or facets | Evolution by year |

Do not combine too many colors and symbols. The chart becomes hard to read.

**Time series:** use a line plot for quantitative values over time. For categories over time, use evolution (stacked) bar plots.

---

## 9. Visualization checklist

- 🎯 **Purpose:** distribution, relationship, or comparison?
- 📊 **Right chart:** bar, line, box, heatmap, scatter?
- 👥 **Audience:** experts, general public, decision-makers?
- 🎨 **Clean and consistent:** title, axis labels, legend, readable font size
- 🧠 **Honest:** no distortion; use log scale or transformation when needed, and say so
- ⚙️ **Interactive** when it helps (tooltips, zoom, filters)
- ♿ **Accessible:** color-blind friendly colors, good contrast, alt text

---

## 10. Final checklist before modeling

- [ ] Know the data source and what each column means
- [ ] Column types are correct (numbers that are really categories converted)
- [ ] Duplicates checked and handled
- [ ] Invalid values and disguised missing values found
- [ ] Formats are consistent (dates, units, category spelling)
- [ ] Missing values: % per column, type guessed (MCAR/MAR/MNAR), strategy chosen and written down
- [ ] Outliers examined: error or real? Influential points checked
- [ ] Every variable viewed alone (table + chart)
- [ ] Relationships checked for all three cases (Q–Q, Q–C, C–C)
- [ ] Redundant (highly correlated) predictors identified
- [ ] Categorical variables encoded if the model needs it
- [ ] Every cleaning decision documented, so someone else can reproduce it

---

## 11. Quick start with the code

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

```python
import pandas as pd
import eda_toolkit as eda

df = pd.read_csv("your_data.csv")

# Step 1-2: inspect and quality
print(eda.overview(df))
print(eda.missing_report(df))
print(eda.disguised_missing(df, ["height", "weight"]))   # 0, -1, -999 ...

# Step 3: univariate
quant, qual = eda.split_columns(df)
print(eda.summarize_quant(df, quant))
print(eda.summarize_qual(df, qual[0]))

# Step 4: preprocess
df = eda.impute_median(df, ["age"])          # or eda.impute_knn / eda.impute_iterative
df["fare_capped"] = eda.cap_outliers(df["fare"])

# Step 5: bivariate (all three cases at once)
tables = eda.bivariate_summary(df, quant, qual)
print(tables["spearman"], tables["eta_squared"], tables["cramers_v"], sep="\n\n")
```

| Function | Purpose |
|---|---|
| `overview`, `split_columns` | Shape, dtypes, NA, duplicates; quantitative/qualitative split |
| `missing_report`, `disguised_missing`, `missing_vs_other_columns`, `missing_indicator_test` | Completeness and missing-type checks |
| `summarize_quant`, `summarize_qual`, `iqr_outliers` | Univariate summaries |
| `cap_outliers`, `zscore`, `minmax`, `one_hot` | Transformations |
| `impute_median`, `impute_knn`, `impute_iterative` | Imputation |
| `corr_matrix`, `corr_gap`, `eta_squared`, `chi2_test`, `cramers_v`, `bivariate_summary` | Bivariate analysis |

Run `python eda_toolkit.py` for a small demo on synthetic data.

---

## 12. Further reading

- *Python for Data Analysis*, Wes McKinney (3rd edition): pandas, NumPy, Jupyter
- *Applied Missing Data Analysis*, Craig K. Enders: deep dive on MCAR / MAR / MNAR
- *Fundamentals of Data Visualization*, Claus O. Wilke: https://clauswilke.com/dataviz/
- The Examples Book, EDA with Python: https://the-examples-book.com/tools/python/eda

---

## License

Notes and code in this repository: MIT License (add a `LICENSE` file).
Course materials belong to their original authors, and this repo contains only my own summary.
