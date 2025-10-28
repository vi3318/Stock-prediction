"""
Example: Multi-Sector Evaluation
Demonstrates generalization across diverse market sectors
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.multi_sector_evaluator import (
    MultiSectorEvaluator,
    SectorRegistry
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_sector_registry():
    """
    Example 1: Explore available sectors
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Sector Registry")
    logger.info("="*70 + "\n")
    
    logger.info("Available Sectors:")
    logger.info("-" * 70)
    
    for sector in SectorRegistry.get_all_sectors():
        logger.info(f"\n{sector.name.upper()}")
        logger.info(f"  Industry Group: {sector.industry_group}")
        logger.info(f"  Description: {sector.description}")
        logger.info(f"  Tickers ({len(sector.tickers)}): {', '.join(sector.tickers)}")
    
    logger.info("\n" + "-"*70)
    all_tickers = SectorRegistry.get_all_tickers()
    logger.info(f"Total tickers across all sectors: {len(all_tickers)}")
    logger.info(f"Tickers: {', '.join(all_tickers)}")


def example_basic_multi_sector():
    """
    Example 2: Basic multi-sector evaluation
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Basic Multi-Sector Evaluation")
    logger.info("="*70 + "\n")
    
    # Mock model class
    class MockModel:
        def __init__(self, config):
            self.config = config
        
        def get_learned_weights(self):
            return {
                'earnings': np.random.uniform(1.5, 2.5),
                'merger': np.random.uniform(2.0, 3.0),
                'product_launch': np.random.uniform(1.2, 2.0),
                'temporal_decay_halflife': np.random.uniform(2.0, 5.0)
            }
    
    # Mock config
    config = {
        'model': {'type': 'late_fusion'},
        'temporal': {'half_life_days': 3}
    }
    
    # Mock data loader
    def data_loader(ticker):
        np.random.seed(hash(ticker) % 2**32)
        n_samples = 1000
        return {
            'X': np.random.randn(n_samples, 50),
            'y': np.random.randint(0, 3, n_samples)
        }
    
    # Create evaluator for 2 sectors
    evaluator = MultiSectorEvaluator(
        model_class=MockModel,
        config=config,
        sectors=['technology', 'financial']
    )
    
    # Evaluate
    results = evaluator.evaluate_all_sectors(data_loader)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    
    for sector_name, result in results.items():
        logger.info(f"\n{sector_name.upper()}")
        logger.info(f"  Tickers evaluated: {result.n_tickers}")
        logger.info(f"  Accuracy: {result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}")
        logger.info(f"  95% CI: [{result.ci_accuracy[0]:.3f}, {result.ci_accuracy[1]:.3f}]")
        logger.info(f"  Sharpe:   {result.mean_sharpe:.2f} ± {result.std_sharpe:.2f}")
        logger.info(f"  CAGR:     {result.mean_cagr:.1%} ± {result.std_cagr:.1%}")


def example_comprehensive_evaluation():
    """
    Example 3: Comprehensive multi-sector evaluation
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Comprehensive Multi-Sector Evaluation")
    logger.info("="*70 + "\n")
    
    # Mock setup
    class MockModel:
        def __init__(self, config):
            self.config = config
    
    config = {'model': {'type': 'late_fusion'}}
    
    def data_loader(ticker):
        np.random.seed(hash(ticker) % 2**32)
        return {
            'X': np.random.randn(1000, 50),
            'y': np.random.randint(0, 3, 1000)
        }
    
    # Evaluate ALL sectors
    evaluator = MultiSectorEvaluator(
        model_class=MockModel,
        config=config,
        sectors=None  # None = all sectors
    )
    
    results = evaluator.evaluate_all_sectors(data_loader)
    
    # Create comparison table
    logger.info("\n" + "="*70)
    logger.info("SECTOR COMPARISON TABLE")
    logger.info("="*70 + "\n")
    
    comparison_df = evaluator.compare_sectors(results)
    print(comparison_df.to_string(index=False))
    
    # Statistical significance tests
    logger.info("\n" + "="*70)
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info("="*70)
    
    sig_tests = evaluator.test_sector_differences(results, metric='accuracy')
    
    logger.info("\nPairwise Comparisons (Accuracy):")
    logger.info("-" * 70)
    
    for (s1, s2), test in sig_tests.items():
        logger.info(f"\n{s1} vs {s2}:")
        logger.info(f"  Mean difference: {test['mean_diff']:+.4f}")
        logger.info(f"  t-statistic: {test['t_statistic']:.3f}")
        logger.info(f"  p-value: {test['p_value']:.4f}")
        logger.info(f"  Cohen's d: {test['cohen_d']:.3f}")
        
        # Interpretation
        if test['significant']:
            logger.info(f"  ✅ SIGNIFICANT difference (p < 0.05)")
        else:
            logger.info(f"  ❌ Not significant (p >= 0.05)")
        
        # Effect size interpretation
        d = abs(test['cohen_d'])
        if d < 0.2:
            effect = "negligible"
        elif d < 0.5:
            effect = "small"
        elif d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        logger.info(f"  Effect size: {effect}")
    
    # Save results
    os.makedirs('results/multi_sector', exist_ok=True)
    evaluator.save_results(results, 'results/multi_sector')
    
    logger.info("\n✅ Results saved to results/multi_sector/")


def example_learned_weights_comparison():
    """
    Example 4: Compare learned weights across sectors
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Learned Weights Across Sectors")
    logger.info("="*70 + "\n")
    
    # Mock setup with learned weights
    class MockModel:
        def __init__(self, config):
            self.config = config
        
        def get_learned_weights(self):
            # Simulate different learned weights per sector
            # (based on random seed from ticker)
            return {
                'earnings': np.random.uniform(1.5, 2.5),
                'merger': np.random.uniform(2.0, 3.5),
                'product_launch': np.random.uniform(1.2, 2.2),
                'guidance': np.random.uniform(1.3, 2.0),
                'temporal_decay_halflife': np.random.uniform(2.0, 6.0)
            }
    
    config = {'model': {'type': 'late_fusion'}}
    
    def data_loader(ticker):
        np.random.seed(hash(ticker) % 2**32)
        return {
            'X': np.random.randn(1000, 50),
            'y': np.random.randint(0, 3, 1000)
        }
    
    # Evaluate 3 sectors
    evaluator = MultiSectorEvaluator(
        model_class=MockModel,
        config=config,
        sectors=['technology', 'financial', 'healthcare']
    )
    
    results = evaluator.evaluate_all_sectors(data_loader)
    
    # Display learned weights
    logger.info("Learned Weights by Sector:")
    logger.info("-" * 70)
    
    for sector_name, result in results.items():
        logger.info(f"\n{sector_name.upper()}")
        
        if result.avg_learned_weights:
            for param, value in result.avg_learned_weights.items():
                std = result.weight_std[param]
                logger.info(f"  {param:30s}: {value:.3f} ± {std:.3f}")
    
    # Insights
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHTS")
    logger.info("="*70)
    
    # Find which event has highest weight in each sector
    for sector_name, result in results.items():
        if result.avg_learned_weights:
            weights = result.avg_learned_weights
            
            # Exclude temporal_decay_halflife from event weights
            event_weights = {k: v for k, v in weights.items() 
                           if k != 'temporal_decay_halflife'}
            
            max_event = max(event_weights, key=event_weights.get)
            max_value = event_weights[max_event]
            
            logger.info(f"\n{sector_name}:")
            logger.info(f"  Most important event: {max_event.replace('_', ' ').title()}")
            logger.info(f"  Weight: {max_value:.3f}")
            
            halflife = weights.get('temporal_decay_halflife', 0)
            logger.info(f"  News half-life: {halflife:.2f} days")
    
    logger.info("\n💡 Observations:")
    logger.info("  • Different sectors may value different event types")
    logger.info("  • Temporal decay rates can vary by sector")
    logger.info("  • Learnable weights adapt to sector characteristics")


def example_generalization_analysis():
    """
    Example 5: Analyze model generalization across sectors
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: Generalization Analysis")
    logger.info("="*70 + "\n")
    
    # Mock setup
    class MockModel:
        def __init__(self, config):
            self.config = config
    
    config = {'model': {'type': 'late_fusion'}}
    
    # Simulate sector-specific performance variance
    sector_difficulty = {
        'technology': 0.68,   # Easier to predict
        'financial': 0.63,    # Harder (more volatile)
        'healthcare': 0.66,   # Medium
        'energy': 0.64,       # Medium-hard
        'consumer': 0.67      # Easier
    }
    
    def data_loader(ticker):
        # Find sector for this ticker
        for sector_name, sector_def in SectorRegistry.SECTORS.items():
            if ticker in sector_def.tickers:
                base_acc = sector_difficulty[sector_name]
                break
        else:
            base_acc = 0.65
        
        np.random.seed(hash(ticker) % 2**32)
        
        # Simulate performance around sector baseline
        actual_acc = base_acc + np.random.normal(0, 0.02)
        
        return {
            'X': np.random.randn(1000, 50),
            'y': np.random.randint(0, 3, 1000),
            '_simulated_accuracy': actual_acc
        }
    
    # Evaluate all sectors
    evaluator = MultiSectorEvaluator(
        model_class=MockModel,
        config=config,
        sectors=None
    )
    
    results = evaluator.evaluate_all_sectors(data_loader)
    
    # Analyze variance
    logger.info("Generalization Analysis:")
    logger.info("-" * 70)
    
    all_accuracies = []
    sector_accuracies = {}
    
    for sector_name, result in results.items():
        sector_accs = [r.accuracy for r in result.ticker_results]
        all_accuracies.extend(sector_accs)
        sector_accuracies[sector_name] = sector_accs
        
        logger.info(f"\n{sector_name}:")
        logger.info(f"  Mean: {result.mean_accuracy:.3f}")
        logger.info(f"  Std:  {result.std_accuracy:.3f}")
        logger.info(f"  CV:   {result.std_accuracy / result.mean_accuracy:.1%}")
    
    # Overall statistics
    overall_mean = np.mean(all_accuracies)
    overall_std = np.std(all_accuracies)
    
    logger.info("\n" + "-"*70)
    logger.info(f"Overall (all sectors):")
    logger.info(f"  Mean: {overall_mean:.3f}")
    logger.info(f"  Std:  {overall_std:.3f}")
    logger.info(f"  CV:   {overall_std / overall_mean:.1%}")
    
    logger.info("\n" + "="*70)
    logger.info("GENERALIZATION ASSESSMENT")
    logger.info("="*70)
    
    # Find best and worst sectors
    sector_means = {s: results[s].mean_accuracy for s in results}
    best_sector = max(sector_means, key=sector_means.get)
    worst_sector = min(sector_means, key=sector_means.get)
    
    logger.info(f"\n✅ Best performance: {best_sector} ({sector_means[best_sector]:.3f})")
    logger.info(f"❌ Worst performance: {worst_sector} ({sector_means[worst_sector]:.3f})")
    logger.info(f"📊 Performance gap: {sector_means[best_sector] - sector_means[worst_sector]:.3f}")
    
    if overall_std / overall_mean < 0.05:
        logger.info("\n✅ EXCELLENT generalization (CV < 5%)")
    elif overall_std / overall_mean < 0.10:
        logger.info("\n✅ GOOD generalization (CV < 10%)")
    else:
        logger.info("\n⚠️  MODERATE generalization (CV > 10%)")
        logger.info("   Consider sector-specific fine-tuning")


def main():
    """Run all examples"""
    logger.info("="*70)
    logger.info("    Multi-Sector Evaluation - Examples")
    logger.info("    Testing Generalization Across Market Sectors")
    logger.info("="*70)
    
    try:
        example_sector_registry()
        example_basic_multi_sector()
        example_comprehensive_evaluation()
        example_learned_weights_comparison()
        example_generalization_analysis()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\n🎯 Why Multi-Sector Evaluation Matters:")
        logger.info("  1. Tests generalization beyond single stock")
        logger.info("  2. Demonstrates robustness across market conditions")
        logger.info("  3. Reveals sector-specific patterns and weights")
        logger.info("  4. Essential for real-world deployment")
        logger.info("  5. Strengthens publication novelty claims")
        
        logger.info("\n📊 For Your Research Paper:")
        logger.info("  • Report cross-sector mean ± std with 95% CI")
        logger.info("  • Test statistical significance between sectors")
        logger.info("  • Analyze learned weight differences by sector")
        logger.info("  • Discuss generalization vs specialization tradeoff")
        logger.info("  • Include sector comparison visualization")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
