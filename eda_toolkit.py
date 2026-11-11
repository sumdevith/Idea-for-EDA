"""
eda_toolkit.py
==============
Small, reusable helpers for Exploratory Data Analysis (EDA).

Follows the workflow in README.md:
    1. Inspect  ->  2. Data quality  ->  3. Univariate  ->  4. Preprocess
    5. Bivariate  ->  6. Pre-modeling checks

Requires: pandas, numpy, scipy  (scikit-learn only for KNN / Iterative imputation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1. INSPECT
# ---------------------------------------------------------------------------

def overview(df: pd.DataFrame) -> pd.DataFrame:
    """One-table overview: dtype, missing count/percent, unique values."""
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Duplicated rows: {df.duplicated().sum()}")
    return out


def split_columns(df: pd.DataFrame, max_unique_as_cat: int = 10):
    """
    Split columns into quantitative and qualitative lists.
    A numeric column with few unique values (e.g. 0/1, class 1/2/3)
    is treated as qualitative. Check the result by hand!
    """
    quant, qual = [], []
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric and df[col].nunique() > max_unique_as_cat:
            quant.append(col)
        else:
            qual.append(col)
    return quant, qual


# ---------------------------------------------------------------------------
# 2. DATA QUALITY
# ---------------------------------------------------------------------------

def disguised_missing(df: pd.DataFrame, cols, suspects=(0, -1, -999, "", "NA", "N/A", "null")):
    """Count suspicious placeholder values that may be missing values in disguise."""
    rows = []
    for col in cols:
        for s in suspects:
            n = (df[col] == s).sum()
            if n > 0:
                rows.append({"column": col, "value": s, "count": int(n),
                             "pct": round(n / len(df) * 100, 2)})
    return pd.DataFrame(rows)


def missing_action(pct: float) -> str:
    """Rule of thumb from the course for what to do with a column's % of NA."""
    if pct == 0:
        return "nothing to do"
    if pct < 5:
        return "drop rows"
    if pct < 10:
        return "drop rows if sample is large; check missing type first"
    if pct < 20:
        return "impute (choose method by missing type)"
    if pct <= 30:
        return "remove column if not important, else impute carefully"
    return "remove column"


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing % per column + suggested action (only columns with NA)."""
    pct = (df.isna().mean() * 100).round(2)
    rep = pd.DataFrame({"n_missing": df.isna().sum(), "pct_missing": pct})
    rep["suggested_action"] = rep["pct_missing"].apply(missing_action)
    return rep[rep["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def missing_vs_other_columns(df: pd.DataFrame, col: str, other_cat_cols) -> dict:
    """
    Compare category proportions BEFORE vs AFTER dropping rows with NA in `col`.
    Big change in a column  ->  missingness is related to it (maybe MAR).
    Almost no change        ->  could be MCAR.
    """
    result = {}
    kept = df.dropna(subset=[col])
    for c in other_cat_cols:
        before = df[c].value_counts(normalize=True)
        after = kept[c].value_counts(normalize=True)
        result[c] = pd.concat([before, after], axis=1, keys=["before", "after"]).round(3)
    return result


def missing_indicator_test(df: pd.DataFrame, col: str, other_cat_col: str) -> float:
    """
    Chi-square test: is 'col is missing' independent of another categorical column?
    Small p-value (< 0.05) -> missingness depends on that column (MAR, not MCAR).
    """
    table = pd.crosstab(df[col].isna(), df[other_cat_col])
    return stats.chi2_contingency(table)[1]


# ---------------------------------------------------------------------------
# 3. UNIVARIATE
# ---------------------------------------------------------------------------

def summarize_qual(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Absolute + relative frequency. The first row is the mode."""
    counts = df[col].value_counts(dropna=False)
    return pd.DataFrame({"count": counts, "proportion": (counts / len(df)).round(3)})


def summarize_quant(df: pd.DataFrame, cols) -> pd.DataFrame:
    """describe() + skew hint (mean vs median) + IQR outlier count."""
    desc = df[cols].describe().T
    desc["median_gap_%"] = ((desc["mean"] - desc["50%"]) / desc["50%"].replace(0, np.nan) * 100).round(1)
    desc["n_outliers_IQR"] = [len(iqr_outliers(df[c])) for c in cols]
    return desc.round(3)


def iqr_fences(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    s = s.dropna()
    low, high = iqr_fences(s, k)
    return s[(s < low) | (s > high)]


# ---------------------------------------------------------------------------
# 4. PREPROCESS
# ---------------------------------------------------------------------------

def cap_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Capping (winsorizing): limit extreme values to the IQR fences."""
    low, high = iqr_fences(s, k)
    return s.clip(lower=low, upper=high)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def one_hot(df: pd.DataFrame, cols, drop_first: bool = False) -> pd.DataFrame:
    """One-hot encode categorical columns (needed for linear models, etc.)."""
    return pd.get_dummies(df, columns=list(cols), drop_first=drop_first, dtype=float)


def impute_median(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Simple imputation. Median is safer than mean when outliers exist."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(out[c].median())
    return out


def impute_knn(df: pd.DataFrame, cols, n_neighbors: int = 5) -> pd.DataFrame:
    """Model-based imputation with K-nearest neighbours (good for MAR)."""
    from sklearn.impute import KNNImputer
    out = df.copy()
    out[cols] = KNNImputer(n_neighbors=n_neighbors).fit_transform(out[cols])
    return out


def impute_iterative(df: pd.DataFrame, cols, random_state: int = 0) -> pd.DataFrame:
    """Model-based imputation, each column is predicted from the others (MICE-style)."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    out = df.copy()
    out[cols] = IterativeImputer(random_state=random_state).fit_transform(out[cols])
    return out


# ---------------------------------------------------------------------------
# 5. BIVARIATE
# ---------------------------------------------------------------------------

def corr_matrix(df: pd.DataFrame, cols, method: str = "pearson") -> pd.DataFrame:
    """
    Correlation matrix.
    method = 'pearson'  -> linear relation, sensitive to outliers
    method = 'spearman' -> monotonic relation, robust, OK for ordinal data
    """
    return df[cols].corr(method=method).round(3)


def corr_gap(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Spearman minus Pearson (absolute). A big gap suggests a non-linear but
    monotonic relation, or outliers affecting Pearson.
    """
    return (df[cols].corr("spearman") - df[cols].corr("pearson")).abs().round(3)


def eta_squared(df: pd.DataFrame, quant: str, cat: str) -> float:
    """
    Eta-squared = BSS / TSS, between 0 and 1.
    0 -> no relation between the groups and the quantitative column,
    1 -> strong relation. Sensitive to outliers.
    """
    d = df[[quant, cat]].dropna()
    grand_mean = d[quant].mean()
    tss = ((d[quant] - grand_mean) ** 2).sum()
    g = d.groupby(cat)[quant]
    bss = (g.count() * (g.mean() - grand_mean) ** 2).sum()
    return float(bss / tss) if tss > 0 else np.nan


def chi2_test(df: pd.DataFrame, a: str, b: str) -> dict:
    """Chi-square test of independence for two categorical columns."""
    table = pd.crosstab(df[a], df[b])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": chi2, "p_value": p, "dof": dof,
            "min_expected_count": expected.min(),
            "note": "expected counts should be >= 5 for a reliable test"}


def cramers_v(df: pd.DataFrame, a: str, b: str, bias_correction: bool = True) -> float:
    """
    Cramer's V, between 0 (no association) and 1 (strong association).
    The plain version overestimates, so bias correction is ON by default.
    """
    table = pd.crosstab(df[a], df[b])
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = chi2 / n
    if not bias_correction:
        return float(np.sqrt(phi2 / min(k - 1, r - 1)))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / min(kc - 1, rc - 1)))


def bivariate_summary(df: pd.DataFrame, quant_cols, qual_cols) -> dict:
    """Run all three cases at once. Returns a dict of tables."""
    res = {}
    if len(quant_cols) >= 2:
        res["pearson"] = corr_matrix(df, quant_cols, "pearson")
        res["spearman"] = corr_matrix(df, quant_cols, "spearman")
    if quant_cols and qual_cols:
        res["eta_squared"] = pd.DataFrame(
            {q: {c: round(eta_squared(df, q, c), 3) for c in qual_cols} for q in quant_cols})
    if len(qual_cols) >= 2:
        res["cramers_v"] = pd.DataFrame(
            {b: {a: round(cramers_v(df, a, b), 3) for a in qual_cols} for b in qual_cols})
    return res


if __name__ == "__main__":
    # tiny self-test with synthetic data
    rng = np.random.default_rng(0)
    n = 300
    demo = pd.DataFrame({
        "group": rng.choice(["A", "B", "C"], n),
        "x": rng.normal(50, 10, n),
    })
    demo["y"] = demo["x"] * 0.5 + (demo["group"] == "A") * 8 + rng.normal(0, 5, n)
    demo["label"] = np.where(demo["y"] > demo["y"].median(), "high", "low")
    demo.loc[rng.choice(n, 20, replace=False), "x"] = np.nan

    print(overview(demo), "\n")
    print(missing_report(demo), "\n")
    for name, table in bivariate_summary(demo.dropna(), ["x", "y"], ["group", "label"]).items():
        print(f"--- {name} ---\n{table}\n")
