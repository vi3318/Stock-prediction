"""
Example: Ablation Study
Demonstrates systematic component evaluation
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.ablation_study import AblationStudy, AblationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_ablation_results():
    """
    Simulate realistic ablation study results
    
    Returns results as if we ran a real ablation study
    """
    logger.info("Simulating ablation study results...")
    
    # Base configuration
    config = {
        'model': {
            'text_encoder': {
                'lstm_units': 128,
                'attention': True
            },
            'numerical_encoder': {
                'gru_units': 64
            },
            'type': 'late_fusion'
        },
        'temporal': {
            'half_life_days': 3,
            'event_weight_multipliers': {
                'earnings': 2.0,
                'merger': 2.5
            },
            'use_source_credibility': True
        },
        'features': {
            'technical_indicators': ['SMA', 'RSI', 'MACD']
        }
    }
    
    # Mock model class
    class MockModel:
        pass
    
    # Create ablation study
    ablation_study = AblationStudy(config, MockModel)
    
    # Simulate results for each configuration
    # These numbers simulate what we'd actually get from training
    simulated_metrics = {
        'full_model': {
            'accuracy': 0.678,
            'f1': 0.660,
            'sharpe': 1.24,
            'cagr': 0.187,
            'max_drawdown': -0.124
        },
        'no_event_weighting': {
            'accuracy': 0.645,  # -3.3% drop (significant)
            'f1': 0.628,
            'sharpe': 1.05,
            'cagr': 0.145,
            'max_drawdown': -0.156
        },
        'no_temporal_decay': {
            'accuracy': 0.632,  # -4.6% drop (most important!)
            'f1': 0.615,
            'sharpe': 0.95,
            'cagr': 0.128,
            'max_drawdown': -0.172
        },
        'no_attention': {
            'accuracy': 0.651,  # -2.7% drop
            'f1': 0.635,
            'sharpe': 1.12,
            'cagr': 0.162,
            'max_drawdown': -0.138
        },
        'no_news': {
            'accuracy': 0.615,  # -6.3% drop (very significant)
            'f1': 0.598,
            'sharpe': 0.78,
            'cagr': 0.095,
            'max_drawdown': -0.195
        },
        'no_technical_indicators': {
            'accuracy': 0.664,  # -1.4% drop (least important)
            'f1': 0.648,
            'sharpe': 1.18,
            'cagr': 0.175,
            'max_drawdown': -0.132
        },
        'no_source_credibility': {
            'accuracy': 0.668,  # -1.0% drop
            'f1': 0.652,
            'sharpe': 1.20,
            'cagr': 0.181,
            'max_drawdown': -0.128
        },
        'no_late_fusion': {
            'accuracy': 0.642,  # -3.6% drop
            'f1': 0.625,
            'sharpe': 1.02,
            'cagr': 0.138,
            'max_drawdown': -0.163
        },
        'random_baseline': {
            'accuracy': 0.333,  # Random 3-class
            'f1': 0.310,
            'sharpe': 0.0,
            'cagr': 0.0,
            'max_drawdown': -0.450
        }
    }
    
    # Create AblationResult objects
    results = {}
    full_model_metrics = simulated_metrics['full_model']
    
    for config_name, metrics in simulated_metrics.items():
        config_info = ablation_study.ablation_configs[config_name]
        
        # Calculate performance drops
        if config_name != 'full_model':
            performance_drop = {
                metric: full_model_metrics[metric] - value
                for metric, value in metrics.items()
            }
        else:
            performance_drop = {}
        
        result = AblationResult(
            configuration=config_info['name'],
            disabled_components=config_info['disabled'],
            metrics=metrics,
            performance_drop=performance_drop
        )
        
        results[config_name] = result
    
    return results, ablation_study


def example_ablation_study():
    """
    Example 1: Complete ablation study
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Complete Ablation Study")
    logger.info("="*70 + "\n")
    
    # Get simulated results
    results, ablation_study = simulate_ablation_results()
    
    # Display results
    logger.info("Ablation Study Results:")
    logger.info("-" * 70)
    
    full_acc = results['full_model'].metrics['accuracy']
    
    for config_name, result in results.items():
        acc = result.metrics['accuracy']
        drop = full_acc - acc
        drop_pct = (drop / full_acc * 100) if full_acc > 0 else 0
        
        logger.info(f"\n{result.configuration:30s}")
        logger.info(f"  Accuracy: {acc:.3f} (drop: {drop:.3f} / {drop_pct:.1f}%)")
        logger.info(f"  F1-Score: {result.metrics['f1']:.3f}")
        logger.info(f"  Sharpe:   {result.metrics['sharpe']:.2f}")
        logger.info(f"  CAGR:     {result.metrics['cagr']:.1%}")
    
    # Create comparison table
    logger.info("\n" + "="*70)
    logger.info("COMPARISON TABLE")
    logger.info("="*70)
    
    df = ablation_study.create_comparison_table(results)
    
    # Display table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print("\n" + df.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    ablation_study.save_results(results, 'results/ablation_study')
    
    logger.info("\n✅ Ablation study complete!")
    logger.info("   Results saved to results/ablation_study/")


def example_component_ranking():
    """
    Example 2: Rank components by importance
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Component Importance Ranking")
    logger.info("="*70 + "\n")
    
    results, _ = simulate_ablation_results()
    
    # Rank by accuracy drop
    component_importance = []
    
    for config_name, result in results.items():
        if config_name not in ['full_model', 'random_baseline']:
            component = result.disabled_components[0] if result.disabled_components else 'unknown'
            drop = result.performance_drop.get('accuracy', 0)
            
            component_importance.append({
                'component': component,
                'accuracy_drop': drop,
                'accuracy_drop_pct': drop / results['full_model'].metrics['accuracy'] * 100,
                'configuration': result.configuration
            })
    
    # Sort by importance (descending)
    component_importance.sort(key=lambda x: x['accuracy_drop'], reverse=True)
    
    logger.info("Component Importance Ranking (by accuracy drop when removed):")
    logger.info("-" * 70)
    
    for i, comp in enumerate(component_importance, 1):
        logger.info(f"\n{i}. {comp['component'].upper().replace('_', ' ')}")
        logger.info(f"   Configuration: {comp['configuration']}")
        logger.info(f"   Accuracy Drop: {comp['accuracy_drop']:.4f} ({comp['accuracy_drop_pct']:.1f}%)")
        
        # Interpretation
        if comp['accuracy_drop'] > 0.04:
            importance = "CRITICAL"
            emoji = "🔴"
        elif comp['accuracy_drop'] > 0.025:
            importance = "HIGH"
            emoji = "🟠"
        elif comp['accuracy_drop'] > 0.015:
            importance = "MEDIUM"
            emoji = "🟡"
        else:
            importance = "LOW"
            emoji = "🟢"
        
        logger.info(f"   Importance: {emoji} {importance}")
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHTS:")
    logger.info("="*70)
    
    top_3 = component_importance[:3]
    logger.info(f"\nTop 3 Most Important Components:")
    for i, comp in enumerate(top_3, 1):
        logger.info(f"  {i}. {comp['component'].replace('_', ' ').title()} "
                   f"({comp['accuracy_drop_pct']:.1f}% drop)")
    
    logger.info(f"\nThese components are ESSENTIAL to the system's performance!")


def example_statistical_significance():
    """
    Example 3: Statistical testing of ablation results
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Statistical Significance of Component Contributions")
    logger.info("="*70 + "\n")
    
    # Simulate per-fold results (walk-forward CV)
    np.random.seed(42)
    n_folds = 8
    
    # Full model performance across folds
    full_model_folds = np.random.normal(0.678, 0.025, n_folds)
    
    # No temporal decay (significant drop)
    no_decay_folds = np.random.normal(0.632, 0.025, n_folds)
    
    # No source credibility (small drop)
    no_credibility_folds = np.random.normal(0.668, 0.025, n_folds)
    
    logger.info("Per-fold accuracy scores:")
    logger.info(f"  Full Model:           {full_model_folds}")
    logger.info(f"  No Temporal Decay:    {no_decay_folds}")
    logger.info(f"  No Source Credibility: {no_credibility_folds}")
    
    # Paired t-tests
    from scipy import stats
    
    logger.info("\n" + "-"*70)
    logger.info("Statistical Tests:")
    logger.info("-"*70)
    
    # Test 1: Full vs No Temporal Decay
    t_stat1, p_value1 = stats.ttest_rel(full_model_folds, no_decay_folds)
    cohen_d1 = (full_model_folds.mean() - no_decay_folds.mean()) / np.std(full_model_folds - no_decay_folds, ddof=1)
    
    logger.info(f"\n1. Temporal Decay Contribution:")
    logger.info(f"   Mean difference: {full_model_folds.mean() - no_decay_folds.mean():.4f}")
    logger.info(f"   t-statistic: {t_stat1:.3f}")
    logger.info(f"   p-value: {p_value1:.4f}")
    logger.info(f"   Cohen's d: {cohen_d1:.3f}")
    
    if p_value1 < 0.05:
        logger.info(f"   ✅ SIGNIFICANT (p < 0.05)")
        logger.info(f"   Temporal decay makes a STATISTICALLY SIGNIFICANT contribution!")
    else:
        logger.info(f"   ❌ Not significant (p >= 0.05)")
    
    # Test 2: Full vs No Source Credibility
    t_stat2, p_value2 = stats.ttest_rel(full_model_folds, no_credibility_folds)
    cohen_d2 = (full_model_folds.mean() - no_credibility_folds.mean()) / np.std(full_model_folds - no_credibility_folds, ddof=1)
    
    logger.info(f"\n2. Source Credibility Contribution:")
    logger.info(f"   Mean difference: {full_model_folds.mean() - no_credibility_folds.mean():.4f}")
    logger.info(f"   t-statistic: {t_stat2:.3f}")
    logger.info(f"   p-value: {p_value2:.4f}")
    logger.info(f"   Cohen's d: {cohen_d2:.3f}")
    
    if p_value2 < 0.05:
        logger.info(f"   ✅ SIGNIFICANT (p < 0.05)")
    else:
        logger.info(f"   ❌ Not significant (p >= 0.05)")
        logger.info(f"   Source credibility has marginal impact")


def example_paper_reporting():
    """
    Example 4: How to report ablation study in research paper
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Reporting Ablation Study in Research Paper")
    logger.info("="*70 + "\n")
    
    logger.info("📝 How to Write the Ablation Study Section:")
    logger.info("-" * 70)
    
    logger.info("""
### Ablation Study

To understand the contribution of each system component, we conducted 
a comprehensive ablation study. We systematically removed individual 
components and measured the impact on model performance.

**Table X: Ablation Study Results**

| Configuration          | Accuracy | F1    | Sharpe | Acc. Drop (%) |
|------------------------|----------|-------|--------|---------------|
| Full Model (Baseline)  | 0.678    | 0.660 | 1.24   | —             |
| w/o Temporal Decay     | 0.632    | 0.615 | 0.95   | -6.8%         |
| w/o News Signals       | 0.615    | 0.598 | 0.78   | -9.3%         |
| w/o Late Fusion        | 0.642    | 0.625 | 1.02   | -5.3%         |
| w/o Event Weighting    | 0.645    | 0.628 | 1.05   | -4.9%         |
| w/o Attention          | 0.651    | 0.635 | 1.12   | -4.0%         |
| w/o Source Credibility | 0.668    | 0.652 | 1.20   | -1.5%         |
| w/o Technical Indic.   | 0.664    | 0.648 | 1.18   | -2.1%         |
| Random Baseline        | 0.333    | 0.310 | 0.00   | -50.9%        |

**Key Findings:**

1. **Temporal Decay** is the most critical component, with removal 
   causing a 6.8% accuracy drop (p < 0.01, Cohen's d = 1.82). This 
   demonstrates that news relevance decreases exponentially with time.

2. **News Signals** contribute 9.3% to accuracy, confirming that 
   textual information provides value beyond price data alone 
   (p < 0.001, d = 2.15).

3. **Event Weighting** shows a 4.9% contribution (p < 0.01, d = 1.45), 
   validating our hypothesis that different event types have different 
   market impacts.

4. **Attention Mechanism** improves performance by 4.0% (p = 0.02, 
   d = 1.12), indicating the model successfully identifies important 
   news articles.

5. **Source Credibility** has the smallest but still measurable impact 
   (1.5% drop, p = 0.08), suggesting high-credibility sources provide 
   marginally better signals.

All components contribute positively, with the full model significantly 
outperforming all ablated versions (p < 0.05 for all except source 
credibility). This validates our multi-component architecture design.
""")
    
    logger.info("\n✅ Best Practices for Ablation Study Reporting:")
    logger.info("  • Test each component in isolation")
    logger.info("  • Report multiple metrics (not just accuracy)")
    logger.info("  • Include statistical significance tests")
    logger.info("  • Rank components by importance")
    logger.info("  • Explain why each component matters")
    logger.info("  • Include random baseline for sanity check")


def main():
    """Run all examples"""
    logger.info("="*70)
    logger.info("    Ablation Study - Examples")
    logger.info("    Systematic Component Evaluation")
    logger.info("="*70)
    
    os.makedirs('results/ablation_study', exist_ok=True)
    
    try:
        example_ablation_study()
        example_component_ranking()
        example_statistical_significance()
        example_paper_reporting()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\n📊 Key Takeaways:")
        logger.info("  1. Ablation studies prove component necessity")
        logger.info("  2. Systematic removal reveals importance ranking")
        logger.info("  3. Statistical tests ensure rigor")
        logger.info("  4. Essential for demonstrating novelty")
        logger.info("  5. Reviewers expect comprehensive ablation analysis")
        
        logger.info("\n🔬 For Your Research Paper:")
        logger.info("  • Ablate ONE component at a time")
        logger.info("  • Test on same data splits for fair comparison")
        logger.info("  • Report p-values and effect sizes")
        logger.info("  • Create clear visualization (bar charts)")
        logger.info("  • Include random baseline (sanity check)")
        logger.info("  • Explain WHY components matter")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
