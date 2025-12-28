"""
Core statistical analysis functions for the Reversal Curse research.

This module provides all statistical tests and analyses required for:
- Computing reversal gaps and asymmetry scores
- Mixed-effects modeling
- Effect size calculations
- Bootstrap confidence intervals
- Power analysis
- Equivalence testing

All functions are designed to produce publication-ready statistics
following APA guidelines.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Suppress convergence warnings in production
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class StatisticalResult:
    """Container for statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_lower: float
    ci_upper: float
    df: Optional[float] = None
    n: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }
        if self.df is not None:
            result["df"] = self.df
        if self.n is not None:
            result["n"] = self.n
        if self.additional_info:
            result.update(self.additional_info)
        return result

    def to_apa(self) -> str:
        """Format result in APA style."""
        if self.df is not None:
            if isinstance(self.df, tuple):
                df_str = f"({self.df[0]}, {self.df[1]})"
            else:
                df_str = f"({self.df:.0f})"
        else:
            df_str = ""

        if self.p_value < 0.001:
            p_str = "p < .001"
        else:
            p_str = f"p = {self.p_value:.3f}"

        if "F" in self.test_name:
            return f"F{df_str} = {self.statistic:.2f}, {p_str}, {self.effect_size_name} = {self.effect_size:.3f}"
        elif "t" in self.test_name:
            return f"t{df_str} = {self.statistic:.2f}, {p_str}, {self.effect_size_name} = {self.effect_size:.3f}"
        elif "z" in self.test_name.lower():
            return f"z = {self.statistic:.2f}, {p_str}, {self.effect_size_name} = {self.effect_size:.3f}"
        else:
            return f"{self.test_name}: statistic = {self.statistic:.2f}, {p_str}, {self.effect_size_name} = {self.effect_size:.3f}"


@dataclass
class ReversalGapResult:
    """Container for reversal gap analysis results."""

    forward_accuracy: float
    reverse_accuracy: float
    gap: float
    gap_ci_lower: float
    gap_ci_upper: float
    cohens_h: float
    n_observations: int
    statistical_test: StatisticalResult


def calculate_reversal_gap(
    forward_correct: np.ndarray,
    reverse_correct: np.ndarray,
    paired: bool = False,
    bootstrap_n: int = 10000,
    alpha: float = 0.05,
) -> ReversalGapResult:
    """
    Calculate the reversal gap between forward and reverse accuracy.

    Parameters
    ----------
    forward_correct : np.ndarray
        Binary array of correct (1) / incorrect (0) for forward direction
    reverse_correct : np.ndarray
        Binary array of correct (1) / incorrect (0) for reverse direction
    paired : bool, default=False
        Whether the observations are paired (same items/participants)
    bootstrap_n : int, default=10000
        Number of bootstrap iterations for confidence intervals
    alpha : float, default=0.05
        Significance level for confidence intervals

    Returns
    -------
    ReversalGapResult
        Complete analysis results including gap, CI, and statistical test
    """
    forward_correct = np.asarray(forward_correct)
    reverse_correct = np.asarray(reverse_correct)

    forward_acc = np.mean(forward_correct)
    reverse_acc = np.mean(reverse_correct)
    gap = forward_acc - reverse_acc

    # Bootstrap confidence interval for the gap
    n_forward = len(forward_correct)
    n_reverse = len(reverse_correct)

    bootstrap_gaps = np.zeros(bootstrap_n)
    rng = np.random.default_rng(42)

    for i in range(bootstrap_n):
        if paired:
            idx = rng.choice(len(forward_correct), size=len(forward_correct), replace=True)
            boot_forward = forward_correct[idx]
            boot_reverse = reverse_correct[idx]
        else:
            boot_forward = rng.choice(forward_correct, size=n_forward, replace=True)
            boot_reverse = rng.choice(reverse_correct, size=n_reverse, replace=True)

        bootstrap_gaps[i] = np.mean(boot_forward) - np.mean(boot_reverse)

    ci_lower = np.percentile(bootstrap_gaps, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_gaps, 100 * (1 - alpha / 2))

    # Effect size (Cohen's h for proportions)
    cohens_h = compute_cohens_h(forward_acc, reverse_acc)

    # Statistical test
    if paired:
        # McNemar's test for paired proportions
        stat_result = mcnemar_test(forward_correct, reverse_correct)
    else:
        # Two-proportion z-test
        stat_result = two_proportion_test(
            np.sum(forward_correct), n_forward,
            np.sum(reverse_correct), n_reverse
        )

    return ReversalGapResult(
        forward_accuracy=forward_acc,
        reverse_accuracy=reverse_acc,
        gap=gap,
        gap_ci_lower=ci_lower,
        gap_ci_upper=ci_upper,
        cohens_h=cohens_h,
        n_observations=n_forward + n_reverse,
        statistical_test=stat_result,
    )


def compute_cohens_h(p1: float, p2: float) -> float:
    """
    Compute Cohen's h effect size for comparing two proportions.

    Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Parameters
    ----------
    p1 : float
        First proportion (0 to 1)
    p2 : float
        Second proportion (0 to 1)

    Returns
    -------
    float
        Cohen's h effect size

    Notes
    -----
    Conventions (Cohen, 1988):
        - Small: h = 0.2
        - Medium: h = 0.5
        - Large: h = 0.8
    """
    # Clamp to avoid numerical issues
    p1 = np.clip(p1, 0.001, 0.999)
    p2 = np.clip(p2, 0.001, 0.999)

    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))

    return phi1 - phi2


def compute_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True,
    paired: bool = False
) -> float:
    """
    Compute Cohen's d effect size for comparing two means.

    Parameters
    ----------
    group1 : np.ndarray
        First group data (treatment/experimental group)
    group2 : np.ndarray
        Second group data (control/comparison group)
    pooled : bool, default=True
        Whether to use pooled standard deviation (recommended for between-subjects)
    paired : bool, default=False
        Whether the data are paired (within-subjects). If True, computes
        Cohen's d_z using the SD of the differences.

    Returns
    -------
    float
        Cohen's d effect size

    Notes
    -----
    Conventions (Cohen, 1988):
        - Small: d = 0.2
        - Medium: d = 0.5
        - Large: d = 0.8

    For non-pooled between-subjects designs, uses the average of both SDs
    rather than just the control SD (more robust when groups are equal).
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    mean1, mean2 = float(np.mean(group1)), float(np.mean(group2))
    n1, n2 = len(group1), len(group2)

    if paired:
        # For paired/within-subjects designs, use SD of differences
        if n1 != n2:
            raise ValueError("Paired Cohen's d requires equal-length arrays")
        differences = group1 - group2
        std_diff = float(np.std(differences, ddof=1))
        if std_diff == 0:
            return 0.0 if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)
        return float(np.mean(differences)) / std_diff

    # Between-subjects designs
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))

    if pooled:
        # Pooled SD (assumes homogeneity of variance)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
        if pooled_std == 0:
            return 0.0 if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)
        return (mean1 - mean2) / pooled_std
    else:
        # Non-pooled: use average of SDs (more robust than using only one group)
        std1 = np.sqrt(var1)
        std2 = np.sqrt(var2)
        avg_std = (std1 + std2) / 2
        if avg_std == 0:
            return 0.0 if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)
        return (mean1 - mean2) / avg_std


def two_proportion_test(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    alternative: str = "two-sided"
) -> StatisticalResult:
    """
    Two-proportion z-test.

    Parameters
    ----------
    count1 : int
        Number of successes in group 1
    nobs1 : int
        Total observations in group 1
    count2 : int
        Number of successes in group 2
    nobs2 : int
        Total observations in group 2
    alternative : str
        'two-sided', 'larger', or 'smaller'

    Returns
    -------
    StatisticalResult
        Test results including z-statistic, p-value, and effect size
    """
    count = np.array([count1, count2])
    nobs = np.array([nobs1, nobs2])

    z_stat, p_value = proportions_ztest(count, nobs, alternative=alternative)

    p1, p2 = count1 / nobs1, count2 / nobs2
    cohens_h = compute_cohens_h(p1, p2)

    # Confidence interval for difference in proportions
    se = np.sqrt(p1 * (1 - p1) / nobs1 + p2 * (1 - p2) / nobs2)
    ci_lower = (p1 - p2) - 1.96 * se
    ci_upper = (p1 - p2) + 1.96 * se

    return StatisticalResult(
        test_name="Two-proportion z-test",
        statistic=z_stat,
        p_value=p_value,
        effect_size=cohens_h,
        effect_size_name="Cohen's h",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=nobs1 + nobs2,
    )


def mcnemar_test(
    condition1: np.ndarray,
    condition2: np.ndarray
) -> StatisticalResult:
    """
    McNemar's test for paired binary data.

    Parameters
    ----------
    condition1 : np.ndarray
        Binary outcomes for condition 1
    condition2 : np.ndarray
        Binary outcomes for condition 2 (same subjects)

    Returns
    -------
    StatisticalResult
        Test results
    """
    condition1 = np.asarray(condition1)
    condition2 = np.asarray(condition2)

    # Create contingency table
    # b = condition1 success, condition2 failure
    # c = condition1 failure, condition2 success
    b = np.sum((condition1 == 1) & (condition2 == 0))
    c = np.sum((condition1 == 0) & (condition2 == 1))

    # McNemar's test with continuity correction
    if b + c == 0:
        chi2 = 0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = stats.chi2.sf(chi2, df=1)

    # Odds ratio as effect size
    if c == 0:
        odds_ratio = np.inf if b > 0 else 1.0
    else:
        odds_ratio = b / c

    # Cohen's h for effect size
    p1 = np.mean(condition1)
    p2 = np.mean(condition2)
    cohens_h = compute_cohens_h(p1, p2)

    return StatisticalResult(
        test_name="McNemar's test",
        statistic=chi2,
        p_value=p_value,
        effect_size=cohens_h,
        effect_size_name="Cohen's h",
        ci_lower=np.nan,
        ci_upper=np.nan,
        df=1,
        n=len(condition1),
        additional_info={"odds_ratio": odds_ratio, "b": b, "c": c},
    )


def mixed_effects_model(
    data: pd.DataFrame,
    formula: str,
    groups: str,
    family: str = "gaussian"
) -> Dict[str, Any]:
    """
    Fit a mixed-effects model for the experimental analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Data with columns for DV, predictors, and grouping variable
    formula : str
        R-style formula (e.g., "accuracy ~ direction * condition")
    groups : str
        Column name for random effects grouping (e.g., "participant_id")
    family : str, default="gaussian"
        Distribution family:
        - "gaussian": Linear mixed model (for continuous proportions)
        - "binomial": Generalized Estimating Equations with logit link

    Returns
    -------
    Dict[str, Any]
        Model results including coefficients, fit statistics, and diagnostics

    Notes
    -----
    For binary outcomes or proportions, consider using:
    1. "gaussian" with proportion data (acceptable when proportions not near 0/1)
    2. "binomial" which uses GEE (population-averaged effects, not subject-specific)

    For true generalized linear mixed models (GLMM), consider using R via rpy2
    or the pymer4 package which wraps R's lme4.
    """
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial, Gaussian
    from statsmodels.genmod.cov_struct import Exchangeable

    n_groups = data[groups].nunique()

    if family == "binomial":
        # Use Generalized Estimating Equations for binomial outcomes
        # GEE provides population-averaged (marginal) effects
        # Parse formula to extract dependent and independent variables
        dv, predictors = formula.split("~")
        dv = dv.strip()
        predictors = predictors.strip()

        # Build design matrix using patsy
        import patsy
        y, X = patsy.dmatrices(f"{dv} ~ {predictors}", data, return_type="dataframe")
        y = y.values.ravel()

        # Fit GEE with exchangeable correlation structure
        gee_model = GEE(
            y,
            X,
            groups=data[groups],
            family=Binomial(),
            cov_struct=Exchangeable()
        )
        result = gee_model.fit(maxiter=10000)

        coefficients = pd.DataFrame({
            "estimate": result.params,
            "std_error": result.bse,
            "z_value": result.tvalues,
            "p_value": result.pvalues,
        })

        return {
            "model": result,
            "model_type": "GEE (Binomial, Exchangeable)",
            "coefficients": coefficients,
            "qic": result.qic(),  # QIC instead of AIC for GEE
            "converged": result.converged,
            "n_observations": len(data),
            "n_groups": n_groups,
            "note": "GEE provides population-averaged effects, not subject-specific"
        }
    else:
        # Linear mixed model for continuous outcomes
        model = smf.mixedlm(
            formula,
            data,
            groups=data[groups],
        )
        result = model.fit(method="lbfgs", maxiter=10000)

        # Extract results
        coefficients = pd.DataFrame({
            "estimate": result.params,
            "std_error": result.bse,
            "z_value": result.tvalues,
            "p_value": result.pvalues,
        })

        return {
            "model": result,
            "model_type": "Linear Mixed Model",
            "coefficients": coefficients,
            "aic": result.aic,
            "bic": result.bic,
            "log_likelihood": result.llf,
            "converged": result.converged,
            "n_observations": result.nobs,
            "n_groups": n_groups,
        }


def anova_2x3_mixed(
    data: pd.DataFrame,
    dv: str,
    within: str,
    between: str,
    subject: str
) -> Dict[str, StatisticalResult]:
    """
    Perform 2x3 mixed ANOVA for the experimental analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data with all variables
    dv : str
        Dependent variable column name
    within : str
        Within-subjects factor column name
    between : str
        Between-subjects factor column name
    subject : str
        Subject identifier column name

    Returns
    -------
    Dict[str, StatisticalResult]
        ANOVA results for main effects and interaction
    """
    try:
        import pingouin as pg

        aov = pg.mixed_anova(
            data=data,
            dv=dv,
            within=within,
            between=between,
            subject=subject,
            correction="auto"
        )

        results = {}

        for _, row in aov.iterrows():
            source = row["Source"]
            results[source] = StatisticalResult(
                test_name=f"F-test ({source})",
                statistic=row["F"],
                p_value=row["p-unc"],
                effect_size=row["np2"],  # partial eta-squared
                effect_size_name="η²p",
                ci_lower=np.nan,
                ci_upper=np.nan,
                df=(row["DF1"], row["DF2"]),
                additional_info={
                    "ss": row.get("SS", np.nan),
                    "ms": row.get("MS", np.nan),
                    "sphericity_correction": row.get("sphericity", "N/A"),
                },
            )

        return results

    except ImportError:
        # Fallback to statsmodels
        return _anova_fallback(data, dv, within, between, subject)


def _anova_fallback(
    data: pd.DataFrame,
    dv: str,
    within: str,
    between: str,
    subject: str
) -> Dict[str, StatisticalResult]:
    """Fallback ANOVA implementation using statsmodels."""
    from statsmodels.stats.anova import AnovaRM

    # Reshape for repeated measures
    results = {}

    # Main effect of between-subjects factor (one-way ANOVA)
    groups = [data[data[between] == level][dv].values
              for level in data[between].unique()]

    f_stat, p_value = stats.f_oneway(*groups)

    # Calculate effect size (eta-squared)
    grand_mean = data[dv].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    results[between] = StatisticalResult(
        test_name=f"F-test ({between})",
        statistic=f_stat,
        p_value=p_value,
        effect_size=eta_sq,
        effect_size_name="η²",
        ci_lower=np.nan,
        ci_upper=np.nan,
        df=(len(groups) - 1, len(data) - len(groups)),
    )

    return results


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_iterations: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for any statistic.

    Uses the percentile method for CI computation.

    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap
    statistic : callable, default=np.mean
        Function to compute the statistic of interest
    n_iterations : int, default=10000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level (0.05 for 95% CI)
    random_state : Optional[int], default=None
        Random seed for reproducibility. If None, uses non-deterministic
        seeding (different results each run). Set to a fixed value (e.g., 42)
        when reproducibility is required.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)

    Notes
    -----
    For reproducible research, always pass an explicit random_state when
    generating final results for publication.
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n = len(data)

    # Vectorized bootstrap for efficiency
    bootstrap_indices = rng.integers(0, n, size=(n_iterations, n))
    bootstrap_samples = data[bootstrap_indices]
    bootstrap_stats = np.apply_along_axis(statistic, 1, bootstrap_samples)

    point_estimate = float(statistic(data))
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def equivalence_test(
    group1: np.ndarray,
    group2: np.ndarray,
    equivalence_bound: float = 0.1,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Two One-Sided Tests (TOST) for equivalence.

    Tests whether the difference between two groups falls within
    the equivalence bounds [-bound, +bound].

    The TOST procedure:
    - Test 1 (lower): H0: diff <= -bound vs H1: diff > -bound
    - Test 2 (upper): H0: diff >= +bound vs H1: diff < +bound

    Equivalence is established if BOTH null hypotheses are rejected,
    i.e., if the confidence interval for the difference falls entirely
    within [-bound, +bound].

    Parameters
    ----------
    group1 : np.ndarray
        First group data
    group2 : np.ndarray
        Second group data
    equivalence_bound : float, default=0.1
        Equivalence margin (as proportion/mean difference)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    Dict[str, Any]
        TOST results including whether equivalence is established
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    mean1, mean2 = float(np.mean(group1)), float(np.mean(group2))
    diff = mean1 - mean2

    n1, n2 = len(group1), len(group2)
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))

    # Use Welch's t-test SE (does not assume equal variances)
    se = np.sqrt(var1 / n1 + var2 / n2)

    # Welch-Satterthwaite degrees of freedom
    if se > 0:
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    else:
        df = n1 + n2 - 2

    # Test 1 (lower): H0: diff <= -bound, H1: diff > -bound
    # We reject H0 if the observed difference is sufficiently greater than -bound
    # p-value = P(T > t_lower) = 1 - cdf(t_lower) = sf(t_lower)
    if se > 0:
        t_lower = (diff - (-equivalence_bound)) / se
        p_lower = float(stats.t.sf(t_lower, df=df))  # sf = survival function = 1 - cdf
    else:
        t_lower = np.inf if diff > -equivalence_bound else -np.inf
        p_lower = 0.0 if diff > -equivalence_bound else 1.0

    # Test 2 (upper): H0: diff >= +bound, H1: diff < +bound
    # We reject H0 if the observed difference is sufficiently less than +bound
    # p-value = P(T < t_upper) = cdf(t_upper)
    if se > 0:
        t_upper = (diff - equivalence_bound) / se
        p_upper = float(stats.t.cdf(t_upper, df=df))
    else:
        t_upper = -np.inf if diff < equivalence_bound else np.inf
        p_upper = 0.0 if diff < equivalence_bound else 1.0

    # Equivalence is established if BOTH null hypotheses are rejected
    # The overall p-value is the maximum of the two one-sided p-values
    p_value = max(p_lower, p_upper)
    equivalent = bool(p_value < alpha)

    # Compute confidence interval for the difference
    t_crit = stats.t.ppf(1 - alpha, df=df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    return {
        "difference": float(diff),
        "equivalence_bound": float(equivalence_bound),
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_value": float(p_value),
        "equivalent": equivalent,
        "df": float(df),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "conclusion": (
            f"Groups are equivalent (p = {p_value:.4f})"
            if equivalent
            else f"Equivalence not established (p = {p_value:.4f})"
        ),
    }


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.95,
    ratio: float = 1.0,
    test_type: str = "two-sample"
) -> Dict[str, float]:
    """
    Perform power analysis to determine required sample size.

    Parameters
    ----------
    effect_size : float
        Expected effect size (Cohen's d or h)
    alpha : float, default=0.05
        Significance level
    power : float, default=0.95
        Desired statistical power
    ratio : float, default=1.0
        Ratio of group sizes (n2/n1)
    test_type : str, default="two-sample"
        Type of test ("two-sample", "paired", "one-sample")

    Returns
    -------
    Dict[str, float]
        Required sample sizes and analysis parameters
    """
    if test_type == "two-sample":
        analysis = TTestIndPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative="two-sided"
        )
        n1 = int(np.ceil(n))
        n2 = int(np.ceil(n * ratio))
        total_n = n1 + n2
    elif test_type == "paired":
        analysis = TTestPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative="two-sided"
        )
        n1 = n2 = int(np.ceil(n))
        total_n = n1
    else:  # one-sample
        analysis = TTestPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative="two-sided"
        )
        n1 = n2 = int(np.ceil(n))
        total_n = n1

    return {
        "n_per_group": n1,
        "n_group2": n2,
        "total_n": total_n,
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
        "test_type": test_type,
    }


def compute_partial_eta_squared(
    ss_effect: float,
    ss_error: float
) -> float:
    """
    Compute partial eta-squared effect size.

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_error : float
        Sum of squares for error

    Returns
    -------
    float
        Partial eta-squared
    """
    return ss_effect / (ss_effect + ss_error)


def compute_omega_squared(
    ss_effect: float,
    ss_total: float,
    ms_error: float,
    df_effect: int
) -> float:
    """
    Compute omega-squared effect size (less biased than eta-squared).

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_total : float
        Total sum of squares
    ms_error : float
        Mean square error
    df_effect : int
        Degrees of freedom for the effect

    Returns
    -------
    float
        Omega-squared
    """
    return (ss_effect - df_effect * ms_error) / (ss_total + ms_error)


def bayes_factor_proportions(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    prior_scale: float = 1.0
) -> float:
    """
    Approximate Bayes Factor for comparing two proportions.

    Uses the BIC approximation for the Bayes Factor.

    Parameters
    ----------
    count1 : int
        Successes in group 1
    n1 : int
        Total in group 1
    count2 : int
        Successes in group 2
    n2 : int
        Total in group 2
    prior_scale : float, default=1.0
        Scale for the prior

    Returns
    -------
    float
        Bayes Factor (BF10: evidence for H1 over H0)
    """
    # Pooled proportion (H0)
    p_pooled = (count1 + count2) / (n1 + n2)

    # Separate proportions (H1)
    p1 = count1 / n1
    p2 = count2 / n2

    # Log-likelihoods
    def log_likelihood(count, n, p):
        if p <= 0 or p >= 1:
            return -np.inf
        return count * np.log(p) + (n - count) * np.log(1 - p)

    ll_h0 = log_likelihood(count1, n1, p_pooled) + log_likelihood(count2, n2, p_pooled)
    ll_h1 = log_likelihood(count1, n1, p1) + log_likelihood(count2, n2, p2)

    # BIC approximation
    bic_h0 = -2 * ll_h0 + 1 * np.log(n1 + n2)  # 1 parameter
    bic_h1 = -2 * ll_h1 + 2 * np.log(n1 + n2)  # 2 parameters

    # BF10 from BIC difference
    bf10 = np.exp((bic_h0 - bic_h1) / 2)

    return bf10


def descriptive_statistics(
    data: np.ndarray,
    decimal_places: int = 3
) -> Dict[str, float]:
    """
    Compute comprehensive descriptive statistics.

    Parameters
    ----------
    data : np.ndarray
        Data array
    decimal_places : int, default=3
        Number of decimal places for rounding

    Returns
    -------
    Dict[str, float]
        Descriptive statistics
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    return {
        "n": len(data),
        "mean": round(np.mean(data), decimal_places),
        "std": round(np.std(data, ddof=1), decimal_places),
        "se": round(stats.sem(data), decimal_places),
        "median": round(np.median(data), decimal_places),
        "min": round(np.min(data), decimal_places),
        "max": round(np.max(data), decimal_places),
        "q1": round(np.percentile(data, 25), decimal_places),
        "q3": round(np.percentile(data, 75), decimal_places),
        "iqr": round(stats.iqr(data), decimal_places),
        "skewness": round(stats.skew(data), decimal_places),
        "kurtosis": round(stats.kurtosis(data), decimal_places),
    }
