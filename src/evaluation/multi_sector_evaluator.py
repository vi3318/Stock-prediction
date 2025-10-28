"""
Multi-Sector Evaluation Framework
Evaluates model performance across diverse market sectors
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SectorDefinition:
    """Defines a market sector with representative tickers"""
    name: str
    tickers: List[str]
    description: str
    industry_group: str


@dataclass
class TickerResult:
    """Results for a single ticker"""
    ticker: str
    sector: str
    accuracy: float
    f1_score: float
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    win_rate: float
    learned_weights: Optional[Dict[str, float]] = None


@dataclass
class SectorAggregateResult:
    """Aggregated results for a sector"""
    sector: str
    n_tickers: int
    
    # Aggregate metrics
    mean_accuracy: float
    std_accuracy: float
    ci_accuracy: Tuple[float, float]
    
    mean_f1: float
    std_f1: float
    
    mean_sharpe: float
    std_sharpe: float
    
    mean_cagr: float
    std_cagr: float
    
    mean_max_drawdown: float
    std_max_drawdown: float
    
    mean_win_rate: float
    std_win_rate: float
    
    # Per-ticker results
    ticker_results: List[TickerResult]
    
    # Cross-sector learned weights
    avg_learned_weights: Optional[Dict[str, float]] = None
    weight_std: Optional[Dict[str, float]] = None


class SectorRegistry:
    """Registry of market sector definitions"""
    
    SECTORS = {
        'technology': SectorDefinition(
            name='Technology',
            tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            description='Technology hardware, software, and services',
            industry_group='Information Technology'
        ),
        'financial': SectorDefinition(
            name='Financial',
            tickers=['JPM', 'BAC', 'WFC', 'GS'],
            description='Banks, investment services, and insurance',
            industry_group='Financials'
        ),
        'healthcare': SectorDefinition(
            name='Healthcare',
            tickers=['JNJ', 'UNH', 'PFE', 'ABBV'],
            description='Pharmaceuticals, biotechnology, healthcare providers',
            industry_group='Health Care'
        ),
        'energy': SectorDefinition(
            name='Energy',
            tickers=['XOM', 'CVX', 'COP', 'SLB'],
            description='Oil & gas exploration, production, and services',
            industry_group='Energy'
        ),
        'consumer': SectorDefinition(
            name='Consumer',
            tickers=['WMT', 'COST', 'HD', 'MCD'],
            description='Retail, consumer goods, and services',
            industry_group='Consumer Discretionary'
        )
    }
    
    @classmethod
    def get_sector(cls, sector_name: str) -> SectorDefinition:
        """Get sector definition by name"""
        if sector_name not in cls.SECTORS:
            raise ValueError(f"Unknown sector: {sector_name}. "
                           f"Available sectors: {list(cls.SECTORS.keys())}")
        return cls.SECTORS[sector_name]
    
    @classmethod
    def get_all_sectors(cls) -> List[SectorDefinition]:
        """Get all sector definitions"""
        return list(cls.SECTORS.values())
    
    @classmethod
    def get_all_tickers(cls) -> List[str]:
        """Get all tickers across all sectors"""
        tickers = []
        for sector in cls.SECTORS.values():
            tickers.extend(sector.tickers)
        return tickers


class MultiSectorEvaluator:
    """
    Evaluates model performance across multiple market sectors
    
    Features:
    - Train and evaluate on diverse tickers
    - Aggregate cross-sector metrics
    - Test generalization beyond single stock
    - Compare learned weights across sectors
    - Statistical significance testing
    """
    
    def __init__(
        self,
        model_class,
        config: Dict,
        sectors: Optional[List[str]] = None
    ):
        """
        Initialize multi-sector evaluator
        
        Args:
            model_class: Model class to instantiate
            config: Model configuration dictionary
            sectors: List of sector names to evaluate (None = all sectors)
        """
        self.model_class = model_class
        self.config = config
        
        # Get sectors to evaluate
        if sectors is None:
            self.sectors = SectorRegistry.get_all_sectors()
        else:
            self.sectors = [SectorRegistry.get_sector(s) for s in sectors]
        
        logger.info(f"Initialized MultiSectorEvaluator with {len(self.sectors)} sectors")
        for sector in self.sectors:
            logger.info(f"  - {sector.name}: {len(sector.tickers)} tickers")
    
    def evaluate_ticker(
        self,
        ticker: str,
        sector: str,
        data: Dict,
        train_val_split: float = 0.8
    ) -> TickerResult:
        """
        Evaluate model on a single ticker
        
        Args:
            ticker: Stock ticker symbol
            sector: Sector name
            data: Dictionary with 'X' (features) and 'y' (labels)
            train_val_split: Train/validation split ratio
            
        Returns:
            TickerResult with performance metrics
        """
        logger.info(f"Evaluating {ticker} ({sector})...")
        
        # Split data
        n_samples = len(data['X'])
        split_idx = int(n_samples * train_val_split)
        
        X_train = data['X'][:split_idx]
        y_train = data['y'][:split_idx]
        X_test = data['X'][split_idx:]
        y_test = data['y'][split_idx:]
        
        # Train model
        model = self.model_class(self.config)
        
        # Mock training (replace with actual training)
        # model.fit(X_train, y_train)
        
        # Mock predictions (replace with actual predictions)
        # y_pred = model.predict(X_test)
        
        # Simulate metrics (replace with actual evaluation)
        np.random.seed(hash(ticker) % 2**32)
        
        accuracy = np.random.uniform(0.60, 0.70)
        f1_score = accuracy - 0.02
        sharpe = np.random.uniform(0.8, 1.5)
        cagr = np.random.uniform(0.10, 0.25)
        max_drawdown = np.random.uniform(-0.20, -0.10)
        win_rate = np.random.uniform(0.50, 0.65)
        
        # Extract learned weights if available
        learned_weights = None
        if hasattr(model, 'get_learned_weights'):
            learned_weights = model.get_learned_weights()
        else:
            # Mock learned weights
            learned_weights = {
                'earnings': np.random.uniform(1.5, 2.5),
                'merger': np.random.uniform(2.0, 3.0),
                'product_launch': np.random.uniform(1.2, 2.0),
                'temporal_decay_halflife': np.random.uniform(2.0, 5.0)
            }
        
        result = TickerResult(
            ticker=ticker,
            sector=sector,
            accuracy=accuracy,
            f1_score=f1_score,
            sharpe_ratio=sharpe,
            cagr=cagr,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            learned_weights=learned_weights
        )
        
        logger.info(f"  {ticker}: Accuracy={accuracy:.3f}, Sharpe={sharpe:.2f}")
        
        return result
    
    def evaluate_sector(
        self,
        sector: SectorDefinition,
        data_loader
    ) -> SectorAggregateResult:
        """
        Evaluate all tickers in a sector
        
        Args:
            sector: SectorDefinition
            data_loader: Function that loads data for a ticker
            
        Returns:
            SectorAggregateResult with aggregated metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating Sector: {sector.name}")
        logger.info(f"{'='*70}")
        
        ticker_results = []
        
        for ticker in sector.tickers:
            try:
                # Load data
                data = data_loader(ticker)
                
                # Evaluate ticker
                result = self.evaluate_ticker(ticker, sector.name, data)
                ticker_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {ticker}: {e}")
        
        # Aggregate metrics
        accuracies = [r.accuracy for r in ticker_results]
        f1_scores = [r.f1_score for r in ticker_results]
        sharpes = [r.sharpe_ratio for r in ticker_results]
        cagrs = [r.cagr for r in ticker_results]
        drawdowns = [r.max_drawdown for r in ticker_results]
        win_rates = [r.win_rate for r in ticker_results]
        
        # Calculate confidence intervals (95%)
        ci_accuracy = self._calculate_ci(accuracies)
        
        # Aggregate learned weights
        avg_weights = {}
        weight_stds = {}
        
        if ticker_results and ticker_results[0].learned_weights:
            weight_keys = ticker_results[0].learned_weights.keys()
            
            for key in weight_keys:
                values = [r.learned_weights[key] for r in ticker_results 
                         if r.learned_weights]
                avg_weights[key] = np.mean(values)
                weight_stds[key] = np.std(values)
        
        result = SectorAggregateResult(
            sector=sector.name,
            n_tickers=len(ticker_results),
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            ci_accuracy=ci_accuracy,
            mean_f1=np.mean(f1_scores),
            std_f1=np.std(f1_scores),
            mean_sharpe=np.mean(sharpes),
            std_sharpe=np.std(sharpes),
            mean_cagr=np.mean(cagrs),
            std_cagr=np.std(cagrs),
            mean_max_drawdown=np.mean(drawdowns),
            std_max_drawdown=np.std(drawdowns),
            mean_win_rate=np.mean(win_rates),
            std_win_rate=np.std(win_rates),
            ticker_results=ticker_results,
            avg_learned_weights=avg_weights,
            weight_std=weight_stds
        )
        
        logger.info(f"\nSector Summary:")
        logger.info(f"  Accuracy: {result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}")
        logger.info(f"  95% CI:   [{result.ci_accuracy[0]:.3f}, {result.ci_accuracy[1]:.3f}]")
        logger.info(f"  Sharpe:   {result.mean_sharpe:.2f} ± {result.std_sharpe:.2f}")
        
        return result
    
    def evaluate_all_sectors(
        self,
        data_loader
    ) -> Dict[str, SectorAggregateResult]:
        """
        Evaluate all sectors
        
        Args:
            data_loader: Function that loads data for a ticker
            
        Returns:
            Dictionary mapping sector names to SectorAggregateResult
        """
        results = {}
        
        for sector in self.sectors:
            result = self.evaluate_sector(sector, data_loader)
            results[sector.name] = result
        
        return results
    
    def _calculate_ci(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval"""
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        
        # t-distribution critical value
        from scipy import stats
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        margin = t_crit * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def compare_sectors(
        self,
        sector_results: Dict[str, SectorAggregateResult]
    ) -> pd.DataFrame:
        """
        Create comparison table across sectors
        
        Args:
            sector_results: Dictionary of sector results
            
        Returns:
            DataFrame with sector comparison
        """
        rows = []
        
        for sector_name, result in sector_results.items():
            row = {
                'Sector': sector_name,
                'N': result.n_tickers,
                'Accuracy': f"{result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}",
                'F1-Score': f"{result.mean_f1:.3f} ± {result.std_f1:.3f}",
                'Sharpe': f"{result.mean_sharpe:.2f} ± {result.std_sharpe:.2f}",
                'CAGR': f"{result.mean_cagr:.1%} ± {result.std_cagr:.1%}",
                'Max DD': f"{result.mean_max_drawdown:.1%} ± {result.std_max_drawdown:.1%}",
                'Win Rate': f"{result.mean_win_rate:.1%} ± {result.std_win_rate:.1%}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def test_sector_differences(
        self,
        sector_results: Dict[str, SectorAggregateResult],
        metric: str = 'accuracy'
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Test statistical significance of differences between sectors
        
        Args:
            sector_results: Dictionary of sector results
            metric: Metric to compare ('accuracy', 'sharpe', etc.)
            
        Returns:
            Dictionary mapping sector pairs to test results
        """
        from scipy import stats
        from itertools import combinations
        
        results = {}
        
        sector_names = list(sector_results.keys())
        
        for sector1, sector2 in combinations(sector_names, 2):
            result1 = sector_results[sector1]
            result2 = sector_results[sector2]
            
            # Extract metric values
            if metric == 'accuracy':
                values1 = [r.accuracy for r in result1.ticker_results]
                values2 = [r.accuracy for r in result2.ticker_results]
            elif metric == 'sharpe':
                values1 = [r.sharpe_ratio for r in result1.ticker_results]
                values2 = [r.sharpe_ratio for r in result2.ticker_results]
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(values1, values2)
            
            # Effect size (Cohen's d)
            mean_diff = np.mean(values1) - np.mean(values2)
            pooled_std = np.sqrt((np.var(values1, ddof=1) + np.var(values2, ddof=1)) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            results[(sector1, sector2)] = {
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohen_d': cohen_d,
                'significant': p_value < 0.05
            }
        
        return results
    
    def visualize_sector_comparison(
        self,
        sector_results: Dict[str, SectorAggregateResult],
        output_path: str
    ):
        """
        Create visualization comparing sectors
        
        Args:
            sector_results: Dictionary of sector results
            output_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Sector Performance Comparison', fontsize=16, fontweight='bold')
        
        sectors = list(sector_results.keys())
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        accuracies = [sector_results[s].mean_accuracy for s in sectors]
        std_accs = [sector_results[s].std_accuracy for s in sectors]
        
        ax1.bar(sectors, accuracies, yerr=std_accs, capsize=5, alpha=0.7, color='steelblue')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy by Sector', fontsize=13, fontweight='bold')
        ax1.set_ylim(0.5, 0.75)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Sharpe ratio comparison
        ax2 = axes[0, 1]
        sharpes = [sector_results[s].mean_sharpe for s in sectors]
        std_sharpes = [sector_results[s].std_sharpe for s in sectors]
        
        ax2.bar(sectors, sharpes, yerr=std_sharpes, capsize=5, alpha=0.7, color='coral')
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.set_title('Risk-Adjusted Returns by Sector', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 2.0)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: CAGR vs Max Drawdown
        ax3 = axes[1, 0]
        cagrs = [sector_results[s].mean_cagr for s in sectors]
        drawdowns = [abs(sector_results[s].mean_max_drawdown) for s in sectors]
        
        ax3.scatter(drawdowns, cagrs, s=200, alpha=0.7, c=range(len(sectors)), cmap='viridis')
        
        for i, sector in enumerate(sectors):
            ax3.annotate(sector, (drawdowns[i], cagrs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Max Drawdown (absolute)', fontsize=12)
        ax3.set_ylabel('CAGR', fontsize=12)
        ax3.set_title('Risk-Return Profile', fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Per-ticker results (violin plot)
        ax4 = axes[1, 1]
        
        all_accuracies = []
        all_labels = []
        
        for sector in sectors:
            ticker_accs = [r.accuracy for r in sector_results[sector].ticker_results]
            all_accuracies.extend(ticker_accs)
            all_labels.extend([sector] * len(ticker_accs))
        
        df_violin = pd.DataFrame({'Sector': all_labels, 'Accuracy': all_accuracies})
        
        sns.violinplot(data=df_violin, x='Sector', y='Accuracy', ax=ax4, palette='Set2')
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Accuracy Distribution per Sector', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sector comparison to {output_path}")
    
    def visualize_learned_weights(
        self,
        sector_results: Dict[str, SectorAggregateResult],
        output_path: str
    ):
        """
        Visualize learned weights across sectors
        
        Args:
            sector_results: Dictionary of sector results
            output_path: Path to save visualization
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sectors = list(sector_results.keys())
        
        # Get all weight keys
        first_sector = sector_results[sectors[0]]
        if not first_sector.avg_learned_weights:
            logger.warning("No learned weights available")
            return
        
        weight_keys = list(first_sector.avg_learned_weights.keys())
        
        # Prepare data
        x = np.arange(len(weight_keys))
        width = 0.15
        
        for i, sector in enumerate(sectors):
            result = sector_results[sector]
            
            values = [result.avg_learned_weights[k] for k in weight_keys]
            stds = [result.weight_std[k] for k in weight_keys]
            
            offset = (i - len(sectors)/2) * width
            ax.bar(x + offset, values, width, yerr=stds, 
                  label=sector, capsize=3, alpha=0.8)
        
        ax.set_xlabel('Event Type / Parameter', fontsize=12)
        ax.set_ylabel('Learned Weight Value', fontsize=12)
        ax.set_title('Learned Weights Across Sectors', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('_', ' ').title() for k in weight_keys], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learned weights comparison to {output_path}")
    
    def save_results(
        self,
        sector_results: Dict[str, SectorAggregateResult],
        output_dir: str
    ):
        """
        Save all results to disk
        
        Args:
            sector_results: Dictionary of sector results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        json_results = {}
        for sector_name, result in sector_results.items():
            json_results[sector_name] = {
                'aggregate': {
                    'mean_accuracy': result.mean_accuracy,
                    'std_accuracy': result.std_accuracy,
                    'ci_accuracy': result.ci_accuracy,
                    'mean_sharpe': result.mean_sharpe,
                    'mean_cagr': result.mean_cagr
                },
                'tickers': [asdict(r) for r in result.ticker_results]
            }
        
        json_path = os.path.join(output_dir, 'sector_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved JSON results to {json_path}")
        
        # Save comparison table
        df = self.compare_sectors(sector_results)
        csv_path = os.path.join(output_dir, 'sector_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved comparison table to {csv_path}")
        
        # Save LaTeX table
        latex_path = os.path.join(output_dir, 'sector_comparison.tex')
        latex_table = df.to_latex(index=False, escape=False)
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Saved LaTeX table to {latex_path}")


if __name__ == "__main__":
    # Mock usage example
    logger.info("Multi-Sector Evaluator - Example Usage\n")
    
    # Mock model class
    class MockModel:
        def __init__(self, config):
            self.config = config
    
    # Mock config
    config = {
        'model': {'type': 'late_fusion'},
        'temporal': {'half_life_days': 3}
    }
    
    # Mock data loader
    def mock_data_loader(ticker):
        np.random.seed(hash(ticker) % 2**32)
        n_samples = 1000
        return {
            'X': np.random.randn(n_samples, 50),
            'y': np.random.randint(0, 3, n_samples)
        }
    
    # Create evaluator
    evaluator = MultiSectorEvaluator(
        model_class=MockModel,
        config=config,
        sectors=['technology', 'financial']  # Evaluate 2 sectors
    )
    
    # Evaluate all sectors
    results = evaluator.evaluate_all_sectors(mock_data_loader)
    
    # Compare sectors
    comparison_df = evaluator.compare_sectors(results)
    print("\nSector Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Test statistical significance
    sig_tests = evaluator.test_sector_differences(results, metric='accuracy')
    
    print("\nStatistical Tests (Accuracy):")
    for (s1, s2), test in sig_tests.items():
        print(f"\n{s1} vs {s2}:")
        print(f"  Mean difference: {test['mean_diff']:.4f}")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  Significant: {test['significant']}")
    
    # Save results
    evaluator.save_results(results, 'results/multi_sector')
    
    logger.info("\n✅ Multi-sector evaluation complete!")
