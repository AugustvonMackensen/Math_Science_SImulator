r"""Hypothesis testing — a uniform :class:`TestResult` over common tests.

Every test returns the same record (statistic, p-value, optional dof, and a
decision at level ``alpha``), so calling code can treat them interchangeably:
one- and two-sample / paired t-tests, one-way ANOVA, chi-square goodness-of-fit
and independence, the Kolmogorov-Smirnov test, Shapiro-Wilk normality, and a
Pearson correlation test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _ss

from core.exceptions import ModelError


@dataclass(slots=True)
class TestResult:
    """Outcome of a statistical hypothesis test."""

    name: str
    statistic: float
    pvalue: float
    alpha: float
    dof: float | None = None

    @property
    def reject_null(self) -> bool:
        """True if the null hypothesis is rejected at level ``alpha``."""
        return self.pvalue < self.alpha

    def summary(self) -> str:
        verdict = "reject H0" if self.reject_null else "fail to reject H0"
        dof = f", dof={self.dof:g}" if self.dof is not None else ""
        return (f"{self.name}: stat={self.statistic:.4g}, "
                f"p={self.pvalue:.4g}{dof} -> {verdict} (alpha={self.alpha})")


def t_test_one_sample(data, popmean: float, *, alpha: float = 0.05) -> TestResult:
    """One-sample t-test of ``H0: mean == popmean``."""
    x = np.asarray(data, dtype=float)
    res = _ss.ttest_1samp(x, popmean)
    return TestResult("one-sample t-test", float(res.statistic), float(res.pvalue),
                      alpha, dof=x.size - 1)


def t_test_two_sample(a, b, *, equal_var: bool = False, alpha: float = 0.05) -> TestResult:
    """Two-sample t-test (Welch's by default) of equal means."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = _ss.ttest_ind(a, b, equal_var=equal_var)
    name = "Student t-test" if equal_var else "Welch t-test"
    return TestResult(name, float(res.statistic), float(res.pvalue), alpha)


def t_test_paired(a, b, *, alpha: float = 0.05) -> TestResult:
    """Paired t-test of ``H0: mean difference == 0``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ModelError("paired samples must have equal length")
    res = _ss.ttest_rel(a, b)
    return TestResult("paired t-test", float(res.statistic), float(res.pvalue),
                      alpha, dof=a.size - 1)


def anova_oneway(*groups, alpha: float = 0.05) -> TestResult:
    """One-way ANOVA F-test that several group means are equal."""
    if len(groups) < 2:
        raise ModelError("ANOVA needs at least two groups")
    res = _ss.f_oneway(*[np.asarray(g, dtype=float) for g in groups])
    return TestResult("one-way ANOVA", float(res.statistic), float(res.pvalue), alpha)


def chi_square_goodness_of_fit(observed, expected=None, *, alpha: float = 0.05) -> TestResult:
    """Chi-square goodness-of-fit test (expected defaults to uniform)."""
    obs = np.asarray(observed, dtype=float)
    exp = None if expected is None else np.asarray(expected, dtype=float)
    res = _ss.chisquare(obs, exp)
    dof = obs.size - 1
    return TestResult("chi-square GOF", float(res.statistic), float(res.pvalue), alpha, dof=dof)


def chi_square_independence(table, *, alpha: float = 0.05) -> TestResult:
    """Chi-square test of independence on a contingency table."""
    chi2, p, dof, _ = _ss.chi2_contingency(np.asarray(table, dtype=float))
    return TestResult("chi-square independence", float(chi2), float(p), alpha, dof=float(dof))


def ks_test(data, dist: str = "norm", *args, alpha: float = 0.05) -> TestResult:
    """One-sample Kolmogorov-Smirnov test against a named distribution."""
    res = _ss.kstest(np.asarray(data, dtype=float), dist, args=args)
    return TestResult(f"KS test vs {dist}", float(res.statistic), float(res.pvalue), alpha)


def shapiro_normality(data, *, alpha: float = 0.05) -> TestResult:
    """Shapiro-Wilk test of normality."""
    res = _ss.shapiro(np.asarray(data, dtype=float))
    return TestResult("Shapiro-Wilk", float(res.statistic), float(res.pvalue), alpha)


def correlation_test(x, y, *, alpha: float = 0.05) -> TestResult:
    """Pearson correlation test of ``H0: rho == 0``; statistic is the correlation r."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ModelError("x and y must have equal length")
    res = _ss.pearsonr(x, y)
    return TestResult("Pearson correlation", float(res.statistic), float(res.pvalue), alpha)
