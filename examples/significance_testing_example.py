"""
Example: Statistical Significance Testing
Demonstrates rigorous statistical comparison of models
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.significance_tests import (
    BootstrapResampler,
    HypothesisTests,
    EffectSizeCalculator,
    ModelComparisonFramework
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_bootstrap_ci():
    """
    Example 1: Bootstrap confidence intervals
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Bootstrap Confidence Intervals")
    logger.info("="*70 + "\n")
    
    # Simulate accuracy scores from 8 CV folds
    np.random.seed(42)
    accuracy_scores = np.array([0.67, 0.69, 0.68, 0.66, 0.70, 0.67, 0.68, 0.69])
    
    logger.info(f"Observed accuracy scores from 8 folds: {accuracy_scores}")
    logger.info(f"Sample mean: {np.mean(accuracy_scores):.4f}")
    logger.info(f"Sample std: {np.std(accuracy_scores, ddof=1):.4f}")
    
    # Bootstrap CI
    bootstrapper = BootstrapResampler(n_iterations=10000)
    results = bootstrapper.bootstrap_metric(accuracy_scores, np.mean)
    
    logger.info(f"\nBootstrap Results (10,000 iterations):")
    logger.info(f"  Estimate: {results['original_estimate']:.4f}")
    logger.info(f"  Bootstrap Mean: {results['bootstrap_mean']:.4f}")
    logger.info(f"  Bootstrap Std: {results['bootstrap_std']:.4f}")
    logger.info(f"  95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    
    logger.info("\n✅ Interpretation:")
    logger.info(f"   We are 95% confident the true accuracy is between")
    logger.info(f"   {results['ci_lower']:.1%} and {results['ci_upper']:.1%}")
    logger.info(f"   (Point estimate: {results['original_estimate']:.1%})")


def example_model_comparison():
    """
    Example 2: Compare two models with statistical tests
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Statistical Model Comparison")
    logger.info("="*70 + "\n")
    
    np.random.seed(42)
    n_folds = 8
    
    # Baseline model (sentiment-only)
    baseline_accuracy = np.array([0.62, 0.64, 0.63, 0.61, 0.65, 0.62, 0.64, 0.63])
    baseline_sharpe = np.array([0.8, 0.9, 0.85, 0.75, 0.95, 0.80, 0.88, 0.82])
    
    # Our full model (with all features)
    our_model_accuracy = np.array([0.67, 0.69, 0.68, 0.66, 0.70, 0.67, 0.69, 0.68])
    our_model_sharpe = np.array([1.2, 1.3, 1.25, 1.15, 1.4, 1.2, 1.3, 1.25])
    
    logger.info("Baseline (Sentiment Only):")
    logger.info(f"  Accuracy: {baseline_accuracy.mean():.4f} ± {baseline_accuracy.std():.4f}")
    logger.info(f"  Sharpe:   {baseline_sharpe.mean():.4f} ± {baseline_sharpe.std():.4f}")
    
    logger.info("\nOur Model (Full System):")
    logger.info(f"  Accuracy: {our_model_accuracy.mean():.4f} ± {our_model_accuracy.std():.4f}")
    logger.info(f"  Sharpe:   {our_model_sharpe.mean():.4f} ± {our_model_sharpe.std():.4f}")
    
    # Paired t-test
    logger.info("\n" + "-"*70)
    logger.info("Statistical Tests:")
    logger.info("-"*70)
    
    tester = HypothesisTests(alpha=0.05)
    
    # Test accuracy
    logger.info("\nAccuracy Comparison:")
    t_result_acc = tester.paired_t_test(our_model_accuracy, baseline_accuracy)
    
    logger.info(f"  t-statistic: {t_result_acc.statistic:.4f}")
    logger.info(f"  p-value: {t_result_acc.p_value:.4f}")
    logger.info(f"  Effect size (Cohen's d): {t_result_acc.effect_size:.3f}")
    logger.info(f"  95% CI for difference: [{t_result_acc.confidence_interval[0]:.4f}, "
               f"{t_result_acc.confidence_interval[1]:.4f}]")
    logger.info(f"  Significant: {t_result_acc.is_significant}")
    logger.info(f"  {t_result_acc.interpretation}")
    
    # Test Sharpe ratio
    logger.info("\nSharpe Ratio Comparison:")
    t_result_sharpe = tester.paired_t_test(our_model_sharpe, baseline_sharpe)
    
    logger.info(f"  t-statistic: {t_result_sharpe.statistic:.4f}")
    logger.info(f"  p-value: {t_result_sharpe.p_value:.4f}")
    logger.info(f"  Effect size (Cohen's d): {t_result_sharpe.effect_size:.3f}")
    logger.info(f"  Significant: {t_result_sharpe.is_significant}")
    logger.info(f"  {t_result_sharpe.interpretation}")
    
    # Effect size interpretation
    logger.info("\n" + "="*70)
    logger.info("Effect Size Interpretation:")
    logger.info("="*70)
    logger.info("Cohen's d Effect Size Guide:")
    logger.info("  < 0.2: Negligible")
    logger.info("  0.2-0.5: Small")
    logger.info("  0.5-0.8: Medium")
    logger.info("  > 0.8: Large")
    
    logger.info(f"\nAccuracy: d = {t_result_acc.effect_size:.3f} "
               f"({EffectSizeCalculator.interpret_cohen_d(t_result_acc.effect_size)})")
    logger.info(f"Sharpe:   d = {t_result_sharpe.effect_size:.3f} "
               f"({EffectSizeCalculator.interpret_cohen_d(t_result_sharpe.effect_size)})")


def example_comprehensive_comparison():
    """
    Example 3: Comprehensive comparison with bootstrap + tests
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Comprehensive Model Comparison Framework")
    logger.info("="*70 + "\n")
    
    np.random.seed(42)
    n_folds = 10
    
    # Simulate results from walk-forward CV
    baseline_results = {
        'accuracy': np.random.normal(0.63, 0.025, n_folds),
        'f1_score': np.random.normal(0.61, 0.03, n_folds),
        'sharpe_ratio': np.random.normal(0.95, 0.15, n_folds),
        'max_drawdown': np.random.normal(-0.18, 0.04, n_folds)
    }
    
    our_model_results = {
        'accuracy': np.random.normal(0.68, 0.025, n_folds),     # +5% improvement
        'f1_score': np.random.normal(0.66, 0.03, n_folds),       # +5% improvement
        'sharpe_ratio': np.random.normal(1.25, 0.15, n_folds),   # +0.3 improvement
        'max_drawdown': np.random.normal(-0.12, 0.03, n_folds)   # -6% improvement (less drawdown)
    }
    
    # Comprehensive comparison
    framework = ModelComparisonFramework(
        bootstrap_iterations=10000,
        confidence_level=0.95,
        alpha=0.05
    )
    
    comparison = framework.compare_models(
        baseline_results,
        our_model_results,
        model1_name="Baseline (Sentiment Only)",
        model2_name="Our Full System"
    )
    
    # Save results
    os.makedirs('results', exist_ok=True)
    framework.save_comparison(comparison, 'results/example_comparison.json')
    
    # Create summary table
    logger.info("\n" + "="*70)
    logger.info("SUMMARY TABLE (Publication Format)")
    logger.info("="*70)
    
    print(f"\n{'Metric':<20} {'Baseline':<25} {'Our Model':<25} {'p-value':<10} {'Effect':<10}")
    print("-" * 95)
    
    for metric, results in comparison.items():
        baseline_mean = results['Baseline (Sentiment Only)']['mean']
        baseline_ci = results['Baseline (Sentiment Only)']['ci']
        
        our_mean = results['Our Full System']['mean']
        our_ci = results['Our Full System']['ci']
        
        p_value = results['t_test']['p_value']
        effect = results['effect_size']['cohen_d']
        
        baseline_str = f"{baseline_mean:.3f} [{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]"
        our_str = f"{our_mean:.3f} [{our_ci[0]:.3f}, {our_ci[1]:.3f}]"
        
        sig_marker = "*" if p_value < 0.05 else ""
        effect_interp = results['effect_size']['interpretation'][:3]
        
        print(f"{metric:<20} {baseline_str:<25} {our_str:<25} {p_value:<10.4f}{sig_marker} {effect:>6.3f} ({effect_interp})")
    
    print("\nNote: * indicates p < 0.05 (statistically significant)")
    print("Effect size interpretation: neg=negligible, sma=small, med=medium, lar=large")


def example_publication_reporting():
    """
    Example 4: How to report results in research paper
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Publication-Quality Reporting")
    logger.info("="*70 + "\n")
    
    logger.info("📊 How to Report Statistical Results in Your Paper:")
    logger.info("-" * 70)
    
    logger.info("\n1. DESCRIPTIVE STATISTICS:")
    logger.info("""
    Example: "Our model achieved an accuracy of 67.8% ± 3.2% 
    (mean ± SD, 95% CI: [65.1%, 70.5%]) across 8 folds of 
    walk-forward cross-validation."
    """)
    
    logger.info("\n2. HYPOTHESIS TEST RESULTS:")
    logger.info("""
    Example: "Our full system significantly outperformed the 
    sentiment-only baseline on accuracy (t(7) = 4.23, p < 0.01, 
    Cohen's d = 1.50), representing a large effect size. The 
    mean improvement was 5.0 percentage points (95% CI: [2.8, 7.2])."
    """)
    
    logger.info("\n3. EFFECT SIZE REPORTING:")
    logger.info("""
    Example: "The improvement in Sharpe ratio showed a medium-to-large 
    effect size (d = 0.68), with our model achieving 1.24 compared to 
    the baseline's 0.95 (p = 0.012)."
    """)
    
    logger.info("\n4. MULTI-METRIC COMPARISON:")
    logger.info("""
    Example: "Table 2 presents comprehensive results across all metrics. 
    Our model showed statistically significant improvements on accuracy 
    (p < 0.01), F1-score (p = 0.02), and Sharpe ratio (p = 0.01). 
    The maximum drawdown was reduced by 6 percentage points, though 
    this did not reach statistical significance (p = 0.08)."
    """)
    
    logger.info("\n5. CONFIDENCE INTERVALS:")
    logger.info("""
    Example: "Bootstrap resampling with 10,000 iterations yielded 
    95% confidence intervals for all metrics. The accuracy confidence 
    interval [65.1%, 70.5%] did not overlap with the baseline's 
    [60.2%, 65.8%], providing strong evidence of superior performance."
    """)
    
    logger.info("\n✅ Best Practices:")
    logger.info("  • Always report effect sizes (not just p-values)")
    logger.info("  • Use confidence intervals (shows precision)")
    logger.info("  • Bootstrap for non-normal metrics (Sharpe, returns)")
    logger.info("  • Correct for multiple comparisons if testing many metrics")
    logger.info("  • Report both statistical and practical significance")


def main():
    """Run all examples"""
    logger.info("="*70)
    logger.info("    Statistical Significance Testing - Examples")
    logger.info("    Publication-Quality Statistical Analysis")
    logger.info("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    try:
        example_bootstrap_ci()
        example_model_comparison()
        example_comprehensive_comparison()
        example_publication_reporting()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\n📊 Key Takeaways:")
        logger.info("  1. Bootstrap provides robust CI without distributional assumptions")
        logger.info("  2. Paired tests account for fold-to-fold variability")
        logger.info("  3. Effect sizes show practical significance")
        logger.info("  4. Comprehensive reporting builds publication credibility")
        logger.info("  5. Statistical rigor separates good from great research")
        
        logger.info("\n🔬 For Your Research Paper:")
        logger.info("  • Use ModelComparisonFramework for all model comparisons")
        logger.info("  • Report bootstrap CI (10,000+ iterations)")
        logger.info("  • Include paired t-test p-values")
        logger.info("  • Always report Cohen's d effect sizes")
        logger.info("  • Create publication-quality comparison tables")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
