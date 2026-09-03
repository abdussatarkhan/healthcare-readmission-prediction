"""
Hypothesis Testing: Discharge Timing Dynamics vs 30-Day Readmission
Preventable Readmissions Risk Stratifier

Investigates whether operational discharge timing (after-hours, weekends, Friday afternoons)
demonstrates statistically significant association with 30-day preventable readmissions.

Tests conducted:
1. Pearson's Chi-Squared Test of Independence (with Yates' continuity correction)
2. Fisher's Exact Test for 2x2 contingency analysis
3. Odds Ratio (OR) and Relative Risk (RR) with 95% Confidence Intervals
4. Cramér's V measure of nominal association
5. Confounder-adjusted Logistic Regression (controlling for age, CCI, and prior admissions)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to python path for relative script imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.utils import setup_logger, load_config, save_dataframe

logger = setup_logger("hypothesis_testing")


class DischargeTimingHypothesisTester:
    """
    Performs inferential statistics and hypothesis testing on clinical care transitions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.alpha = self.config.get("hypothesis_testing", {}).get("significance_level", 0.05)
        self.reports_dir = Path(self.config.get("paths", {}).get("reports_dir", "reports"))
        self.images_dir = Path(self.config.get("paths", {}).get("images_dir", "images"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def test_chi_squared_independence(
        self,
        df: pd.DataFrame,
        timing_col: str = "is_weekend_discharge",
        target_col: str = "readmitted_30d"
    ) -> Dict[str, Any]:
        """
        Calculates Pearson's Chi-Squared test of independence, odds ratio,
        confidence intervals, and Cramér's V effect size.
        """
        logger.info(f"Running Chi-Squared test: {timing_col} vs {target_col}")

        contingency_table = pd.crosstab(df[timing_col], df[target_col])
        logger.info(f"Contingency Table:\n{contingency_table}")

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table, correction=True)

        # 2x2 specific calculations: Odds Ratio and Fisher's exact test
        if contingency_table.shape == (2, 2):
            odds_ratio, p_fisher = stats.fisher_exact(contingency_table)
            
            # Confidence interval for Odds Ratio (Woolf's method)
            a = contingency_table.iloc[1, 1]
            b = contingency_table.iloc[1, 0]
            c = contingency_table.iloc[0, 1]
            d = contingency_table.iloc[0, 0]

            se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_lower = np.exp(np.log(odds_ratio) - 1.96 * se_log_or)
            ci_upper = np.exp(np.log(odds_ratio) + 1.96 * se_log_or)
        else:
            odds_ratio, p_fisher, ci_lower, ci_upper = None, None, None, None

        # Cramér's V calculation
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

        is_significant = bool(p_val < self.alpha)

        results = {
            "hypothesis": f"Independence between {timing_col} and {target_col}",
            "sample_size": int(n),
            "chi2_statistic": round(float(chi2), 4),
            "degrees_of_freedom": int(dof),
            "p_value": float(p_val),
            "significance_level": self.alpha,
            "statistically_significant": is_significant,
            "odds_ratio": round(float(odds_ratio), 4) if odds_ratio else None,
            "odds_ratio_95_ci": [round(float(ci_lower), 4), round(float(ci_upper), 4)] if ci_lower else None,
            "fisher_exact_p_value": float(p_fisher) if p_fisher else None,
            "cramers_v": round(float(cramers_v), 4)
        }

        logger.info(f"Results: Chi2={chi2:.4f}, p={p_val:.5e}, Significant={is_significant}")
        return results

    def run_adjusted_multivariate_test(
        self,
        df: pd.DataFrame,
        timing_col: str = "is_weekend_discharge",
        target_col: str = "readmitted_30d"
    ) -> Dict[str, Any]:
        """
        Performs multivariate logistic regression adjusting for age and comorbidity
        to test if the discharge timing effect is an independent risk factor.
        """
        logger.info("Conducting confounder-adjusted multivariable logistic regression...")

        # Prepare model formula
        cols_available = df.columns
        covariates = [timing_col]
        
        for candidate in ["anchor_age", "charlson_age_adjusted_score", "los_days", "prior_admissions_count"]:
            if candidate in cols_available:
                covariates.append(candidate)

        formula = f"{target_col} ~ " + " + ".join(covariates)
        logger.info(f"Regression Formula: {formula}")

        logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

        timing_coef = logit_model.params[timing_col]
        timing_p = logit_model.pvalues[timing_col]
        adjusted_or = np.exp(timing_coef)
        conf_int = np.exp(logit_model.conf_int().loc[timing_col])

        adjusted_results = {
            "formula": formula,
            "covariates_controlled": covariates[1:],
            "timing_coefficient": round(float(timing_coef), 4),
            "timing_p_value": float(timing_p),
            "adjusted_odds_ratio": round(float(adjusted_or), 4),
            "adjusted_95_ci": [round(float(conf_int[0]), 4), round(float(conf_int[1]), 4)],
            "pseudo_r_squared": round(float(logit_model.prsquared), 4)
        }

        logger.info(f"Adjusted OR: {adjusted_or:.3f} (95% CI: {conf_int[0]:.3f} - {conf_int[1]:.3f}), p={timing_p:.4e}")
        return adjusted_results

    def generate_discharge_temporal_plots(self, df: pd.DataFrame) -> Path:
        """
        Generates visualizations of readmission rates across hour of day and day of week.
        """
        logger.info("Generating temporal discharge risk distributions...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Hour of Day Readmission Rate
        if "discharge_hour" in df.columns and "readmitted_30d" in df.columns:
            hourly = df.groupby("discharge_hour")["readmitted_30d"].agg(["mean", "count"]).reset_index()
            ax1.bar(hourly["discharge_hour"], hourly["mean"] * 100, color="#2b5c8f", edgecolor="black", alpha=0.85)
            ax1.axvline(17, color="red", linestyle="--", label="After-Hours Threshold (17:00)")
            ax1.set_xlabel("Discharge Hour (24h Clock)", fontsize=11)
            ax1.set_ylabel("30-Day Readmission Rate (%)", fontsize=11)
            ax1.set_title("Readmission Rate by Hour of Discharge", fontsize=12, weight="bold")
            ax1.set_xticks(range(0, 24, 2))
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

        # 2. Day of Week Readmission Rate
        if "discharge_dayofweek" in df.columns and "readmitted_30d" in df.columns:
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            daily = df.groupby("discharge_dayofweek")["readmitted_30d"].agg(["mean", "count"]).reset_index()
            colors = ["#2b5c8f" if d < 5 else "#e24a33" for d in daily["discharge_dayofweek"]]
            ax2.bar(daily["discharge_dayofweek"], daily["mean"] * 100, color=colors, edgecolor="black", alpha=0.85)
            ax2.set_xticks(range(7))
            ax2.set_xticklabels(dow_names)
            ax2.set_xlabel("Day of Discharge", fontsize=11)
            ax2.set_ylabel("30-Day Readmission Rate (%)", fontsize=11)
            ax2.set_title("Readmission Rate by Day of Week (Red = Weekend)", fontsize=12, weight="bold")
            ax2.grid(True, alpha=0.3)

        plot_path = self.images_dir / "discharge_timing_analysis.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved temporal analysis figure to {plot_path}")
        return plot_path

    def run_complete_hypothesis_suite(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Orchestrates full inferential test suite across all timing hypotheses.
        """
        logger.info("Executing comprehensive discharge timing hypothesis suite...")

        timing_features = [
            ("is_weekend_discharge", "Weekend vs Weekday Discharge"),
            ("is_evening_discharge", "Evening (>=17:00) vs Daytime Discharge"),
            ("discharge_friday_afternoon", "Friday Afternoon vs Other Discharge")
        ]

        full_report = {}
        for col, desc in timing_features:
            if col in df.columns:
                logger.info(f"Testing: {desc} ({col})")
                chi2_res = self.test_chi_squared_independence(df, timing_col=col)
                adj_res = self.run_adjusted_multivariate_test(df, timing_col=col)
                full_report[col] = {
                    "description": desc,
                    "bivariate_test": chi2_res,
                    "adjusted_multivariate_test": adj_res
                }

        self.generate_discharge_temporal_plots(df)

        report_file = self.reports_dir / "hypothesis_test_results.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        logger.info(f"Hypothesis testing results saved to {report_file}")
        return full_report


def main():
    config = load_config()
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    feat_path = processed_dir / "features_matrix.parquet"

    if not feat_path.is_file():
        logger.info("Features file missing. Executing upstream feature pipeline...")
        from scripts.feature_engineering import main as fe_main
        fe_main()

    df = pd.read_parquet(feat_path)
    tester = DischargeTimingHypothesisTester(config)
    results = tester.run_complete_hypothesis_suite(df)
    print("Hypothesis Testing Finished. Summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
