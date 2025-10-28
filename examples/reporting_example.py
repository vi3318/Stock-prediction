"""
Example: Automated Results Reporting
Demonstrates automated aggregation and LaTeX table generation
"""

import numpy as np
import pandas as pd
import logging
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.reporting.automated_reports import (
    ResultsAggregator,
    MetricsSummarizer,
    LaTeXTableGenerator,
    ReportGenerator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mock_results():
    """Create mock results for demonstration"""
    logger.info("Setting up mock results...")
    
    # Create results directory structure
    os.makedirs('results/walk_forward', exist_ok=True)
    os.makedirs('results/ablation_study', exist_ok=True)
    os.makedirs('results/multi_sector', exist_ok=True)
    os.makedirs('results/significance_tests', exist_ok=True)
    
    # Mock walk-forward results
    wf_results = {
        'aggregate_metrics': {
            'accuracy': {
                'mean': 0.678,
                'std': 0.025,
                'ci_lower': 0.653,
                'ci_upper': 0.703,
                'min': 0.642,
                'max': 0.715,
                'n_samples': 8
            },
            'f1_score': {
                'mean': 0.660,
                'std': 0.028,
                'ci_lower': 0.632,
                'ci_upper': 0.688,
                'min': 0.615,
                'max': 0.698,
                'n_samples': 8
            },
            'sharpe_ratio': {
                'mean': 1.24,
                'std': 0.18,
                'ci_lower': 1.06,
                'ci_upper': 1.42,
                'min': 0.95,
                'max': 1.52,
                'n_samples': 8
            },
            'cagr': {
                'mean': 0.187,
                'std': 0.042,
                'ci_lower': 0.145,
                'ci_upper': 0.229,
                'min': 0.128,
                'max': 0.245,
                'n_samples': 8
            }
        },
        'num_folds': 8,
        'window_type': 'expanding'
    }
    
    with open('results/walk_forward/results.json', 'w') as f:
        json.dump(wf_results, f, indent=2)
    
    # Mock ablation study results
    ablation_data = {
        'Configuration': [
            'Full Model',
            'w/o Temporal Decay',
            'w/o News Signals',
            'w/o Event Weighting',
            'w/o Attention',
            'w/o Late Fusion',
            'Random Baseline'
        ],
        'Accuracy': [0.678, 0.632, 0.615, 0.645, 0.651, 0.642, 0.333],
        'F1-Score': [0.660, 0.615, 0.598, 0.628, 0.635, 0.625, 0.310],
        'Sharpe': [1.24, 0.95, 0.78, 1.05, 1.12, 1.02, 0.00],
        'Acc. Drop (%)': ['—', '-6.8%', '-9.3%', '-4.9%', '-4.0%', '-5.3%', '-50.9%']
    }
    
    pd.DataFrame(ablation_data).to_csv('results/ablation_study/ablation_comparison.csv', index=False)
    
    # Mock multi-sector results
    sector_data = {
        'Technology': {
            'aggregate': {
                'mean_accuracy': 0.685,
                'std_accuracy': 0.023,
                'mean_sharpe': 1.28,
                'mean_cagr': 0.192
            }
        },
        'Financial': {
            'aggregate': {
                'mean_accuracy': 0.632,
                'std_accuracy': 0.031,
                'mean_sharpe': 1.05,
                'mean_cagr': 0.145
            }
        },
        'Healthcare': {
            'aggregate': {
                'mean_accuracy': 0.661,
                'std_accuracy': 0.027,
                'mean_sharpe': 1.18,
                'mean_cagr': 0.173
            }
        }
    }
    
    with open('results/multi_sector/sector_results.json', 'w') as f:
        json.dump(sector_data, f, indent=2)
    
    # Mock sector comparison CSV
    sector_comparison = {
        'Sector': ['Technology', 'Financial', 'Healthcare'],
        'N': [4, 4, 4],
        'Accuracy': ['68.5% ± 2.3%', '63.2% ± 3.1%', '66.1% ± 2.7%'],
        'Sharpe': ['1.28 ± 0.15', '1.05 ± 0.22', '1.18 ± 0.18'],
        'CAGR': ['19.2% ± 4.1%', '14.5% ± 5.3%', '17.3% ± 4.7%']
    }
    
    pd.DataFrame(sector_comparison).to_csv('results/multi_sector/sector_comparison.csv', index=False)
    
    logger.info("✅ Mock results created")


def example_results_aggregation():
    """
    Example 1: Aggregate results from all modules
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Results Aggregation")
    logger.info("="*70 + "\n")
    
    # Create aggregator
    aggregator = ResultsAggregator()
    
    # Load all results
    aggregated = aggregator.aggregate_all_results()
    
    logger.info("Aggregated Results Summary:")
    logger.info("-" * 70)
    
    # Walk-forward results
    if aggregated.get('walk_forward'):
        logger.info("\n✅ Walk-Forward CV Results:")
        wf = aggregated['walk_forward']
        logger.info(f"   Folds: {wf.get('num_folds')}")
        logger.info(f"   Window: {wf.get('window_type')}")
        logger.info(f"   Metrics: {len(wf.get('aggregate_metrics', {}))}")
    
    # Ablation results
    if aggregated.get('ablation'):
        logger.info("\n✅ Ablation Study Results: Available")
    
    # Multi-sector results
    if aggregated.get('multi_sector'):
        logger.info("\n✅ Multi-Sector Results:")
        ms = aggregated['multi_sector']
        logger.info(f"   Sectors: {len([k for k in ms.keys() if k != 'timestamp'])}")
    
    logger.info("\n" + "="*70)


def example_metrics_summary():
    """
    Example 2: Create metrics summary table
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Metrics Summary Table")
    logger.info("="*70 + "\n")
    
    # Create summarizer
    summarizer = MetricsSummarizer(confidence_level=0.95)
    
    # Mock metric data (from walk-forward CV)
    metrics_dict = {
        'Accuracy': np.random.normal(0.678, 0.025, 8),
        'F1-Score': np.random.normal(0.660, 0.028, 8),
        'Sharpe Ratio': np.random.normal(1.24, 0.18, 8),
        'CAGR': np.random.normal(0.187, 0.042, 8)
    }
    
    # Create summary table
    summary_df = summarizer.create_summary_table(metrics_dict)
    
    logger.info("Metrics Summary:")
    logger.info("-" * 70)
    print("\n" + summary_df.to_string(index=False))
    
    logger.info("\n✅ Summary includes:")
    logger.info("   • Mean ± Standard Deviation")
    logger.info("   • 95% Confidence Intervals")
    logger.info("   • Min/Max values")
    logger.info("   • Sample sizes")


def example_latex_table_generation():
    """
    Example 3: Generate LaTeX tables for publication
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: LaTeX Table Generation")
    logger.info("="*70 + "\n")
    
    # Create generator
    latex_gen = LaTeXTableGenerator(table_style='ieee')
    
    # 1. Metrics table
    logger.info("1. Main Metrics Table:")
    logger.info("-" * 70)
    
    metrics_data = pd.DataFrame([
        {
            'metric_name': 'Accuracy',
            'mean': 0.678,
            'std': 0.025,
            'ci_lower': 0.653,
            'ci_upper': 0.703,
            'min': 0.642,
            'max': 0.715,
            'n_samples': 8
        },
        {
            'metric_name': 'Sharpe Ratio',
            'mean': 1.24,
            'std': 0.18,
            'ci_lower': 1.06,
            'ci_upper': 1.42,
            'min': 0.95,
            'max': 1.52,
            'n_samples': 8
        }
    ])
    
    latex_metrics = latex_gen.generate_metrics_table(
        metrics_data,
        caption="Walk-Forward Cross-Validation Results",
        label="tab:results"
    )
    
    print(latex_metrics)
    
    # Save to file
    os.makedirs('results/reports', exist_ok=True)
    with open('results/reports/table_metrics_example.tex', 'w') as f:
        f.write(latex_metrics)
    
    logger.info("✅ Saved to results/reports/table_metrics_example.tex")
    
    # 2. Ablation table
    logger.info("\n2. Ablation Study Table:")
    logger.info("-" * 70)
    
    ablation_df = pd.read_csv('results/ablation_study/ablation_comparison.csv')
    latex_ablation = latex_gen.generate_ablation_table(
        ablation_df,
        caption="Ablation Study Results",
        label="tab:ablation"
    )
    
    print(latex_ablation)
    
    with open('results/reports/table_ablation_example.tex', 'w') as f:
        f.write(latex_ablation)
    
    logger.info("✅ Saved to results/reports/table_ablation_example.tex")


def example_complete_report_generation():
    """
    Example 4: Generate complete automated report
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Complete Report Generation")
    logger.info("="*70 + "\n")
    
    # Create report generator
    generator = ReportGenerator(
        results_dir='results',
        output_dir='results/reports'
    )
    
    # Generate complete report
    report_files = generator.generate_complete_report()
    
    logger.info("\nGenerated Report Files:")
    logger.info("="*70)
    
    for name, path in report_files.items():
        logger.info(f"  {name:20s}: {path}")
        
        # Display file size
        if os.path.exists(path):
            size = os.path.getsize(path)
            logger.info(f"                        ({size:,} bytes)")
    
    # Display executive summary
    if 'executive_summary' in report_files:
        logger.info("\n" + "="*70)
        logger.info("EXECUTIVE SUMMARY PREVIEW")
        logger.info("="*70)
        
        with open(report_files['executive_summary'], 'r') as f:
            summary = f.read()
        
        # Print first 30 lines
        lines = summary.split('\n')[:30]
        print('\n'.join(lines))
        
        if len(summary.split('\n')) > 30:
            logger.info("\n... (truncated) ...")


def example_using_in_paper():
    """
    Example 5: How to use generated tables in your LaTeX paper
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: Using Tables in Your Research Paper")
    logger.info("="*70 + "\n")
    
    logger.info("📝 How to Include Generated Tables in Your Paper:\n")
    
    logger.info("1. In your main.tex file:")
    logger.info("-" * 70)
    print("""
\\documentclass{IEEEtran}

\\begin{document}

\\section{Experimental Results}

Our model was evaluated using walk-forward cross-validation
with 8 folds. Table~\\ref{tab:results} presents the aggregate
performance metrics with 95\\% confidence intervals.

% Include the auto-generated table
\\input{results/reports/table_metrics.tex}

As shown in Table~\\ref{tab:results}, the model achieves
67.8\\% $\\pm$ 2.5\\% accuracy with a Sharpe ratio of 1.24.

\\subsection{Ablation Study}

To understand component contributions, we conducted a 
comprehensive ablation study (Table~\\ref{tab:ablation}).

\\input{results/reports/table_ablation.tex}

The ablation study reveals that temporal decay is the most
critical component, with its removal causing a 6.8\\% drop
in accuracy.

\\end{document}
""")
    
    logger.info("\n2. Copy generated .tex files to your paper directory:")
    logger.info("-" * 70)
    logger.info("   cp results/reports/*.tex research_paper/tables/")
    
    logger.info("\n3. Update paths in main.tex:")
    logger.info("-" * 70)
    logger.info("   \\input{tables/table_metrics.tex}")
    logger.info("   \\input{tables/table_ablation.tex}")
    
    logger.info("\n✅ Benefits of automated table generation:")
    logger.info("   • No manual copying of numbers (error-prone)")
    logger.info("   • Consistent formatting across all tables")
    logger.info("   • Easy to regenerate when results change")
    logger.info("   • LaTeX code is publication-ready")
    logger.info("   • Meets IEEE/ACM/Springer formatting standards")


def main():
    """Run all examples"""
    logger.info("="*70)
    logger.info("    Automated Results Reporting - Examples")
    logger.info("    Generate Publication-Ready Tables and Reports")
    logger.info("="*70)
    
    # Setup mock results
    setup_mock_results()
    
    try:
        example_results_aggregation()
        example_metrics_summary()
        example_latex_table_generation()
        example_complete_report_generation()
        example_using_in_paper()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\n📊 Why Automated Reporting Matters:")
        logger.info("  1. Eliminates manual transcription errors")
        logger.info("  2. Saves hours of table formatting work")
        logger.info("  3. Ensures consistent number precision")
        logger.info("  4. Easy to regenerate when models improve")
        logger.info("  5. Publication-ready LaTeX output")
        
        logger.info("\n📝 For Your Research Paper:")
        logger.info("  • Use \\input{} to include generated tables")
        logger.info("  • All tables follow IEEE formatting standards")
        logger.info("  • Confidence intervals included automatically")
        logger.info("  • Sample sizes reported for reproducibility")
        logger.info("  • Significance tests integrated into comparisons")
        
        logger.info("\n💡 Next Steps:")
        logger.info("  1. Run actual experiments to generate real results")
        logger.info("  2. Execute: python -m src.reporting.automated_reports")
        logger.info("  3. Copy .tex files to research_paper/tables/")
        logger.info("  4. Include in your paper with \\input{}")
        logger.info("  5. Cite results with Table~\\ref{} references")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
