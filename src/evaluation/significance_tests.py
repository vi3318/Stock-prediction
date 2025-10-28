"""
Statistical Significance Testing Module
Bootstrap resampling, hypothesis tests, and effect size calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
from dataclasses import dataclass
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class SignificanceTestResult:
    """Results from a statistical significance test"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    interpretation: str


class BootstrapResampler:
    """
    Bootstrap resampling for confidence intervals and hypothesis testing
    
    Bootstrap is crucial for financial ML because:
    - No assumptions about data distribution
    - Works with non-normal, skewed metrics (Sharpe, returns)
    - Provides robust confidence intervals
    - Publication-standard for empirical research
    """
    
    def __init__(self, n_iterations: int = 10000, confidence_level: float = 0.95, random_seed: int = 42):
        """
        Args:
            n_iterations: Number of bootstrap samples (10000+ for publication)
            confidence_level: Confidence level for intervals (typically 0.95)
            random_seed: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.logger = logging.getLogger(self.__class__.__name__)
        
        np.random.seed(random_seed)
    
    def bootstrap_metric(
        self,
        data: np.ndarray,
        metric_fn: Callable,
        **metric_kwargs
    ) -> Dict:
        """
        Bootstrap confidence interval for a metric
        
        Args:
            data: Sample data
            metric_fn: Function that computes metric from data
            **metric_kwargs: Additional arguments for metric_fn
            
        Returns:
            Dictionary with mean, CI, and bootstrap distribution
        """
        self.logger.info(f"Running bootstrap with {self.n_iterations:,} iterations...")
        
        n = len(data)
        bootstrap_estimates = []
        
        for i in range(self.n_iterations):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            
            # Compute metric on bootstrap sample
            estimate = metric_fn(bootstrap_sample, **metric_kwargs)
            bootstrap_estimates.append(estimate)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Compute original metric
        original_estimate = metric_fn(data, **metric_kwargs)
        
        # Compute confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_estimates, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_estimates, (1 - alpha/2) * 100)
        
        # Standard error
        se = np.std(bootstrap_estimates)
        
        results = {
            'original_estimate': float(original_estimate),
            'bootstrap_mean': float(np.mean(bootstrap_estimates)),
            'bootstrap_std': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_level': self.confidence_level,
            'n_iterations': self.n_iterations,
            'bootstrap_distribution': bootstrap_estimates
        }
        
        self.logger.info(
            f"Estimate: {original_estimate:.4f}, "
            f"{self.confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
        )
        
        return results
    
    def bootstrap_difference(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        metric_fn: Callable,
        **metric_kwargs
    ) -> Dict:
        """
        Bootstrap confidence interval for difference between two groups
        
        Args:
            data1: First group data
            data2: Second group data
            metric_fn: Metric function
            **metric_kwargs: Additional metric arguments
            
        Returns:
            Dictionary with difference estimate and CI
        """
        self.logger.info("Bootstrapping difference between two groups...")
        
        n1, n2 = len(data1), len(data2)
        bootstrap_differences = []
        
        for i in range(self.n_iterations):
            # Resample both groups
            sample1 = np.random.choice(data1, size=n1, replace=True)
            sample2 = np.random.choice(data2, size=n2, replace=True)
            
            # Compute metrics
            metric1 = metric_fn(sample1, **metric_kwargs)
            metric2 = metric_fn(sample2, **metric_kwargs)
            
            # Difference
            diff = metric1 - metric2
            bootstrap_differences.append(diff)
        
        bootstrap_differences = np.array(bootstrap_differences)
        
        # Original difference
        original_diff = metric_fn(data1, **metric_kwargs) - metric_fn(data2, **metric_kwargs)
        
        # CI for difference
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_differences, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_differences, (1 - alpha/2) * 100)
        
        # Test if CI excludes zero (significant difference)
        is_significant = not (ci_lower <= 0 <= ci_upper)
        
        results = {
            'original_difference': float(original_diff),
            'bootstrap_mean_diff': float(np.mean(bootstrap_differences)),
            'bootstrap_std_diff': float(np.std(bootstrap_differences)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'is_significant': is_significant,
            'p_value_bootstrap': float(np.mean(bootstrap_differences <= 0) * 2),  # Two-tailed
            'bootstrap_distribution': bootstrap_differences
        }
        
        self.logger.info(
            f"Difference: {original_diff:.4f}, "
            f"CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"Significant: {is_significant}"
        )
        
        return results


class HypothesisTests:
    """
    Classical hypothesis tests for model comparison
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (typically 0.05)
        """
        self.alpha = alpha
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def paired_t_test(
        self,
        model1_scores: np.ndarray,
        model2_scores: np.ndarray,
        alternative: str = 'two-sided'
    ) -> SignificanceTestResult:
        """
        Paired t-test for comparing two models on same test folds
        
        Use when:
        - Comparing two models on same data splits
        - Same folds in walk-forward CV
        - Paired observations
        
        Args:
            model1_scores: Scores from model 1
            model2_scores: Scores from model 2
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            SignificanceTestResult
        """
        self.logger.info("Running paired t-test...")
        
        # Compute differences
        differences = model1_scores - model2_scores
        
        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(model1_scores, model2_scores, alternative=alternative)
        
        # Effect size (Cohen's d for paired samples)
        cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Confidence interval for difference
        se = stats.sem(differences)
        ci = stats.t.interval(0.95, len(differences)-1, loc=np.mean(differences), scale=se)
        
        is_significant = p_value < self.alpha
        
        # Interpretation
        if is_significant:
            winner = "Model 1" if np.mean(differences) > 0 else "Model 2"
            interpretation = f"{winner} significantly better (p={p_value:.4f}, d={cohen_d:.3f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"
        
        self.logger.info(interpretation)
        
        return SignificanceTestResult(
            test_name="Paired t-test",
            statistic=float(t_statistic),
            p_value=float(p_value),
            effect_size=float(cohen_d),
            confidence_interval=(float(ci[0]), float(ci[1])),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def wilcoxon_signed_rank_test(
        self,
        model1_scores: np.ndarray,
        model2_scores: np.ndarray,
        alternative: str = 'two-sided'
    ) -> SignificanceTestResult:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        
        Use when:
        - Data is not normally distributed
        - Outliers present
        - Small sample sizes
        
        Args:
            model1_scores: Scores from model 1
            model2_scores: Scores from model 2
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            SignificanceTestResult
        """
        self.logger.info("Running Wilcoxon signed-rank test...")
        
        # Wilcoxon test
        statistic, p_value = stats.wilcoxon(
            model1_scores, model2_scores,
            alternative=alternative
        )
        
        # Effect size (rank-biserial correlation)
        differences = model1_scores - model2_scores
        n = len(differences)
        r = 1 - (2 * statistic) / (n * (n + 1) / 2)
        
        is_significant = p_value < self.alpha
        
        if is_significant:
            winner = "Model 1" if np.median(differences) > 0 else "Model 2"
            interpretation = f"{winner} significantly better (p={p_value:.4f}, r={r:.3f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"
        
        self.logger.info(interpretation)
        
        return SignificanceTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(r),
            confidence_interval=(np.nan, np.nan),  # Not applicable
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def mcnemar_test(
        self,
        model1_predictions: np.ndarray,
        model2_predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> SignificanceTestResult:
        """
        McNemar's test for comparing classification models
        
        Use for:
        - Classification tasks
        - Comparing error patterns
        - Same test set
        
        Args:
            model1_predictions: Predictions from model 1
            model2_predictions: Predictions from model 2
            true_labels: True labels
            
        Returns:
            SignificanceTestResult
        """
        self.logger.info("Running McNemar's test...")
        
        # Create contingency table
        model1_correct = model1_predictions == true_labels
        model2_correct = model2_predictions == true_labels
        
        # Count discordant pairs
        n10 = np.sum(model1_correct & ~model2_correct)  # M1 correct, M2 wrong
        n01 = np.sum(~model1_correct & model2_correct)  # M1 wrong, M2 correct
        
        # McNemar's test statistic
        if (n10 + n01) == 0:
            statistic = 0
            p_value = 1.0
        else:
            statistic = (abs(n10 - n01) - 1)**2 / (n10 + n01)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        # Effect size (odds ratio)
        if n01 == 0:
            odds_ratio = np.inf
        else:
            odds_ratio = n10 / n01
        
        is_significant = p_value < self.alpha
        
        if is_significant:
            winner = "Model 1" if n10 > n01 else "Model 2"
            interpretation = f"{winner} significantly better (p={p_value:.4f}, OR={odds_ratio:.2f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"
        
        self.logger.info(interpretation)
        self.logger.info(f"M1 correct, M2 wrong: {n10}, M1 wrong, M2 correct: {n01}")
        
        return SignificanceTestResult(
            test_name="McNemar's test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(odds_ratio) if odds_ratio != np.inf else np.nan,
            confidence_interval=(np.nan, np.nan),
            is_significant=is_significant,
            interpretation=interpretation
        )


class EffectSizeCalculator:
    """Calculate effect sizes for practical significance"""
    
    @staticmethod
    def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Cohen's d effect size
        
        Interpretation:
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8+
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        d = (mean1 - mean2) / pooled_std
        return float(d)
    
    @staticmethod
    def interpret_cohen_d(d: float) -> str:
        """Interpret Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


class ModelComparisonFramework:
    """
    Complete framework for statistically rigorous model comparison
    """
    
    def __init__(
        self,
        bootstrap_iterations: int = 10000,
        confidence_level: float = 0.95,
        alpha: float = 0.05
    ):
        """
        Args:
            bootstrap_iterations: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            alpha: Significance level for hypothesis tests
        """
        self.bootstrap = BootstrapResampler(bootstrap_iterations, confidence_level)
        self.hypothesis_tests = HypothesisTests(alpha)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compare_models(
        self,
        model1_results: Dict[str, np.ndarray],
        model2_results: Dict[str, np.ndarray],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        metrics: List[str] = None
    ) -> Dict:
        """
        Comprehensive comparison of two models
        
        Args:
            model1_results: Dict mapping metric_name -> scores array
            model2_results: Dict mapping metric_name -> scores array
            model1_name: Name of first model
            model2_name: Name of second model
            metrics: List of metrics to compare (None = all)
            
        Returns:
            Complete comparison results
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"COMPARING: {model1_name} vs {model2_name}")
        self.logger.info(f"{'='*70}\n")
        
        if metrics is None:
            metrics = list(model1_results.keys())
        
        comparison_results = {}
        
        for metric in metrics:
            self.logger.info(f"\nMetric: {metric}")
            self.logger.info("-" * 50)
            
            scores1 = model1_results[metric]
            scores2 = model2_results[metric]
            
            # Bootstrap CI for each model
            boot1 = self.bootstrap.bootstrap_metric(scores1, np.mean)
            boot2 = self.bootstrap.bootstrap_metric(scores2, np.mean)
            
            # Bootstrap CI for difference
            boot_diff = self.bootstrap.bootstrap_difference(scores1, scores2, np.mean)
            
            # Paired t-test
            t_test = self.hypothesis_tests.paired_t_test(scores1, scores2)
            
            # Wilcoxon test
            wilcoxon = self.hypothesis_tests.wilcoxon_signed_rank_test(scores1, scores2)
            
            # Effect size
            cohen_d = EffectSizeCalculator.cohen_d(scores1, scores2)
            effect_interpretation = EffectSizeCalculator.interpret_cohen_d(cohen_d)
            
            comparison_results[metric] = {
                model1_name: {
                    'mean': boot1['original_estimate'],
                    'ci': (boot1['ci_lower'], boot1['ci_upper']),
                    'std': boot1['bootstrap_std']
                },
                model2_name: {
                    'mean': boot2['original_estimate'],
                    'ci': (boot2['ci_lower'], boot2['ci_upper']),
                    'std': boot2['bootstrap_std']
                },
                'difference': {
                    'value': boot_diff['original_difference'],
                    'ci': (boot_diff['ci_lower'], boot_diff['ci_upper']),
                    'is_significant': boot_diff['is_significant']
                },
                't_test': {
                    'statistic': t_test.statistic,
                    'p_value': t_test.p_value,
                    'is_significant': t_test.is_significant
                },
                'wilcoxon': {
                    'statistic': wilcoxon.statistic,
                    'p_value': wilcoxon.p_value,
                    'is_significant': wilcoxon.is_significant
                },
                'effect_size': {
                    'cohen_d': cohen_d,
                    'interpretation': effect_interpretation
                }
            }
            
            # Log summary
            self.logger.info(f"{model1_name}: {boot1['original_estimate']:.4f} "
                           f"[{boot1['ci_lower']:.4f}, {boot1['ci_upper']:.4f}]")
            self.logger.info(f"{model2_name}: {boot2['original_estimate']:.4f} "
                           f"[{boot2['ci_lower']:.4f}, {boot2['ci_upper']:.4f}]")
            self.logger.info(f"Difference: {boot_diff['original_difference']:.4f} "
                           f"[{boot_diff['ci_lower']:.4f}, {boot_diff['ci_upper']:.4f}]")
            self.logger.info(f"p-value (t-test): {t_test.p_value:.4f}")
            self.logger.info(f"Effect size: {cohen_d:.3f} ({effect_interpretation})")
            self.logger.info(f"Conclusion: {t_test.interpretation}")
        
        return comparison_results
    
    def save_comparison(self, results: Dict, filepath: str = 'results/model_comparison.json'):
        """Save comparison results to JSON"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            return obj
        
        results_serializable = convert(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Comparison results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Simulate model scores from walk-forward CV
    np.random.seed(42)
    n_folds = 8
    
    # Model 1: Baseline (lower performance)
    model1_accuracy = np.random.normal(0.65, 0.03, n_folds)
    model1_sharpe = np.random.normal(1.0, 0.2, n_folds)
    
    # Model 2: Our model (better performance)
    model2_accuracy = np.random.normal(0.68, 0.03, n_folds)  # +3% improvement
    model2_sharpe = np.random.normal(1.3, 0.2, n_folds)      # +0.3 improvement
    
    model1_results = {
        'accuracy': model1_accuracy,
        'sharpe_ratio': model1_sharpe
    }
    
    model2_results = {
        'accuracy': model2_accuracy,
        'sharpe_ratio': model2_sharpe
    }
    
    # Compare models
    framework = ModelComparisonFramework(bootstrap_iterations=10000)
    
    comparison = framework.compare_models(
        model1_results,
        model2_results,
        model1_name="Baseline (Sentiment Only)",
        model2_name="Our Model (Full System)"
    )
    
    # Save results
    framework.save_comparison(comparison, 'results/statistical_comparison.json')
    
    print("\n" + "="*70)
    print("Statistical significance testing complete!")
    print("Results demonstrate rigorous, publication-quality evaluation")
    print("="*70)
