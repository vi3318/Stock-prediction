"""
Ablation Study Module
Systematically evaluates contribution of each system component
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Results from a single ablation experiment"""
    configuration: str
    disabled_components: List[str]
    metrics: Dict[str, float]
    performance_drop: Dict[str, float]


class AblationStudy:
    """
    Comprehensive ablation study framework
    
    Systematically disables components to measure their contribution:
    - Event importance weighting
    - Temporal decay
    - Attention mechanism
    - News signals (text features)
    - Technical indicators (numerical features)
    - Source credibility weighting
    """
    
    def __init__(self, config: Dict, base_model_class):
        """
        Args:
            config: Base configuration
            base_model_class: Model class to use for experiments
        """
        self.config = config
        self.base_model_class = base_model_class
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define ablation configurations
        self.ablation_configs = self._define_ablation_configs()
    
    def _define_ablation_configs(self) -> Dict[str, Dict]:
        """
        Define all ablation configurations
        
        Returns:
            Dictionary mapping config_name -> config_modifications
        """
        configs = {
            'full_model': {
                'name': 'Full Model (Baseline)',
                'disabled': [],
                'description': 'Complete system with all components'
            },
            'no_event_weighting': {
                'name': 'No Event Weighting',
                'disabled': ['event_weighting'],
                'description': 'All events weighted equally (no importance multipliers)',
                'modifications': {
                    'temporal': {
                        'event_weight_multipliers': {k: 1.0 for k in [
                            'earnings', 'merger', 'acquisition', 'partnership',
                            'lawsuit', 'product_launch', 'regulatory'
                        ]}
                    }
                }
            },
            'no_temporal_decay': {
                'name': 'No Temporal Decay',
                'disabled': ['temporal_decay'],
                'description': 'News weighted equally regardless of age',
                'modifications': {
                    'temporal': {
                        'half_life_days': 10000  # Effectively no decay
                    }
                }
            },
            'no_attention': {
                'name': 'No Attention Mechanism',
                'disabled': ['attention'],
                'description': 'LSTM without multi-head attention',
                'modifications': {
                    'model': {
                        'text_encoder': {
                            'attention': False
                        }
                    }
                }
            },
            'no_news': {
                'name': 'No News Signals',
                'disabled': ['news_signals'],
                'description': 'Price data only (no text features)',
                'modifications': {
                    'use_text_features': False
                }
            },
            'no_technical_indicators': {
                'name': 'No Technical Indicators',
                'disabled': ['technical_indicators'],
                'description': 'Raw price data only (no engineered features)',
                'modifications': {
                    'features': {
                        'technical_indicators': []  # Empty list
                    }
                }
            },
            'no_source_credibility': {
                'name': 'No Source Credibility',
                'disabled': ['source_credibility'],
                'description': 'All news sources weighted equally',
                'modifications': {
                    'temporal': {
                        'use_source_credibility': False
                    }
                }
            },
            'no_late_fusion': {
                'name': 'Early Fusion (vs Late)',
                'disabled': ['late_fusion'],
                'description': 'Concatenate features early instead of late fusion',
                'modifications': {
                    'model': {
                        'type': 'early_fusion'
                    }
                }
            },
            'random_baseline': {
                'name': 'Random Baseline',
                'disabled': ['all'],
                'description': 'Random predictions (sanity check)',
                'modifications': {
                    'use_random_predictions': True
                }
            }
        }
        
        return configs
    
    def run_ablation_experiment(
        self,
        config_name: str,
        train_fn: Callable,
        evaluate_fn: Callable,
        data: Tuple
    ) -> AblationResult:
        """
        Run single ablation experiment
        
        Args:
            config_name: Name of ablation configuration
            train_fn: Function to train model
            evaluate_fn: Function to evaluate model
            data: Training/validation/test data
            
        Returns:
            AblationResult with metrics
        """
        ablation_config = self.ablation_configs[config_name]
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"ABLATION: {ablation_config['name']}")
        self.logger.info(f"Disabled: {ablation_config['disabled']}")
        self.logger.info(f"Description: {ablation_config['description']}")
        self.logger.info(f"{'='*70}\n")
        
        # Create modified config
        modified_config = self._apply_modifications(
            self.config.copy(),
            ablation_config.get('modifications', {})
        )
        
        # Handle special cases
        if config_name == 'random_baseline':
            metrics = self._run_random_baseline(data)
        else:
            # Train and evaluate with modified config
            model = train_fn(modified_config, data)
            metrics = evaluate_fn(model, data)
        
        result = AblationResult(
            configuration=ablation_config['name'],
            disabled_components=ablation_config['disabled'],
            metrics=metrics,
            performance_drop={}  # Will be computed later
        )
        
        self.logger.info(f"Results for {ablation_config['name']}:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        return result
    
    def run_complete_ablation_study(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        data: Tuple,
        configs_to_run: Optional[List[str]] = None
    ) -> Dict[str, AblationResult]:
        """
        Run complete ablation study across all configurations
        
        Args:
            train_fn: Training function
            evaluate_fn: Evaluation function
            data: Data tuple
            configs_to_run: List of config names to run (None = all)
            
        Returns:
            Dictionary mapping config_name -> AblationResult
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPREHENSIVE ABLATION STUDY")
        self.logger.info("="*70 + "\n")
        
        if configs_to_run is None:
            configs_to_run = list(self.ablation_configs.keys())
        
        results = {}
        
        # Run full model first (baseline)
        if 'full_model' not in configs_to_run:
            configs_to_run.insert(0, 'full_model')
        
        for config_name in configs_to_run:
            result = self.run_ablation_experiment(
                config_name, train_fn, evaluate_fn, data
            )
            results[config_name] = result
        
        # Compute performance drops relative to full model
        full_model_metrics = results['full_model'].metrics
        
        for config_name, result in results.items():
            if config_name != 'full_model':
                result.performance_drop = {
                    metric: full_model_metrics[metric] - result.metrics[metric]
                    for metric in full_model_metrics.keys()
                }
        
        return results
    
    def _apply_modifications(self, config: Dict, modifications: Dict) -> Dict:
        """Apply modifications to config (deep merge)"""
        def deep_merge(base, updates):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        return deep_merge(config, modifications)
    
    def _run_random_baseline(self, data: Tuple) -> Dict[str, float]:
        """Run random predictions (sanity check)"""
        _, _, y_test = data
        
        # Random predictions
        n_classes = len(np.unique(y_test))
        y_pred = np.random.choice(n_classes, size=len(y_test))
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred, average='weighted')),
            'sharpe': 0.0,  # No trading strategy
            'cagr': 0.0
        }
        
        return metrics
    
    def create_comparison_table(
        self,
        results: Dict[str, AblationResult],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Create comparison table for ablation results
        
        Args:
            results: Ablation results
            metrics: Metrics to include (None = all)
            
        Returns:
            DataFrame with comparison
        """
        if metrics is None:
            metrics = list(results['full_model'].metrics.keys())
        
        rows = []
        
        for config_name, result in results.items():
            row = {
                'Configuration': result.configuration,
                'Disabled Components': ', '.join(result.disabled_components) if result.disabled_components else 'None'
            }
            
            for metric in metrics:
                row[f'{metric}'] = result.metrics[metric]
                
                if config_name != 'full_model':
                    drop = result.performance_drop.get(metric, 0)
                    row[f'{metric}_drop'] = drop
                    row[f'{metric}_drop_pct'] = (drop / results['full_model'].metrics[metric] * 100) if results['full_model'].metrics[metric] != 0 else 0
        
        df = pd.DataFrame(rows)
        
        # Sort by accuracy (or first metric)
        if metrics:
            df = df.sort_values(metrics[0], ascending=False)
        
        return df
    
    def visualize_ablation_results(
        self,
        results: Dict[str, AblationResult],
        save_path: str = 'results/ablation_study.png'
    ):
        """
        Create visualization of ablation study results
        
        Args:
            results: Ablation results
            save_path: Path to save figure
        """
        self.logger.info("Creating ablation study visualization...")
        
        # Prepare data
        configs = []
        metrics_data = {
            'accuracy': [],
            'f1': [],
            'sharpe': []
        }
        
        for config_name, result in results.items():
            configs.append(result.configuration)
            for metric in metrics_data.keys():
                metrics_data[metric].append(result.metrics.get(metric, 0))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Bar chart of accuracy
        ax1 = axes[0, 0]
        colors = ['#2ecc71' if i == 0 else '#e74c3c' if configs[i] == 'Random Baseline' else '#3498db' 
                  for i in range(len(configs))]
        bars = ax1.barh(configs, metrics_data['accuracy'], color=colors, alpha=0.8)
        ax1.set_xlabel('Accuracy', fontsize=12)
        ax1.set_title('Classification Accuracy by Configuration', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Plot 2: Performance drop
        ax2 = axes[0, 1]
        if 'full_model' in results:
            full_acc = results['full_model'].metrics['accuracy']
            drops = [full_acc - acc for acc in metrics_data['accuracy']]
            drop_pct = [(full_acc - acc) / full_acc * 100 for acc in metrics_data['accuracy']]
            
            bars2 = ax2.barh(configs, drop_pct, color=colors, alpha=0.8)
            ax2.set_xlabel('Performance Drop (%)', fontsize=12)
            ax2.set_title('Accuracy Drop vs Full Model', fontsize=13, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                if width > 0:
                    ax2.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        # Plot 3: Multi-metric comparison (top configurations)
        ax3 = axes[1, 0]
        top_n = min(5, len(configs))
        top_configs = configs[:top_n]
        
        x = np.arange(top_n)
        width = 0.25
        
        acc_vals = metrics_data['accuracy'][:top_n]
        f1_vals = metrics_data['f1'][:top_n]
        sharpe_vals = [s / 2 for s in metrics_data['sharpe'][:top_n]]  # Scale for visibility
        
        ax3.bar(x - width, acc_vals, width, label='Accuracy', color='#3498db', alpha=0.8)
        ax3.bar(x, f1_vals, width, label='F1-Score', color='#2ecc71', alpha=0.8)
        ax3.bar(x + width, sharpe_vals, width, label='Sharpe/2', color='#f39c12', alpha=0.8)
        
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title(f'Multi-Metric Comparison (Top {top_n})', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([c[:20] for c in top_configs], rotation=45, ha='right', fontsize=9)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Component contribution
        ax4 = axes[1, 1]
        
        component_importance = {}
        for config_name, result in results.items():
            if config_name != 'full_model' and result.disabled_components:
                component = result.disabled_components[0] if len(result.disabled_components) == 1 else 'multiple'
                drop = result.performance_drop.get('accuracy', 0)
                if component not in component_importance:
                    component_importance[component] = []
                component_importance[component].append(drop)
        
        # Average importance per component
        comp_names = []
        comp_importance = []
        for comp, drops in component_importance.items():
            comp_names.append(comp.replace('_', ' ').title())
            comp_importance.append(np.mean(drops))
        
        # Sort by importance
        sorted_indices = np.argsort(comp_importance)[::-1]
        comp_names = [comp_names[i] for i in sorted_indices]
        comp_importance = [comp_importance[i] for i in sorted_indices]
        
        colors4 = ['#e74c3c' if imp > 0.03 else '#f39c12' if imp > 0.01 else '#95a5a6' 
                   for imp in comp_importance]
        
        ax4.barh(comp_names, comp_importance, color=colors4, alpha=0.8)
        ax4.set_xlabel('Accuracy Drop (when disabled)', fontsize=12)
        ax4.set_title('Component Importance Ranking', fontsize=13, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        for i, (name, imp) in enumerate(zip(comp_names, comp_importance)):
            ax4.text(imp, i, f'{imp:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Ablation study visualization saved to {save_path}")
    
    def save_results(
        self,
        results: Dict[str, AblationResult],
        output_dir: str = 'results/ablation_study'
    ):
        """
        Save ablation study results
        
        Args:
            results: Ablation results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        results_dict = {
            config_name: {
                'configuration': result.configuration,
                'disabled_components': result.disabled_components,
                'metrics': result.metrics,
                'performance_drop': result.performance_drop
            }
            for config_name, result in results.items()
        }
        
        json_path = os.path.join(output_dir, 'ablation_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to {json_path}")
        
        # Save comparison table
        df = self.create_comparison_table(results)
        csv_path = os.path.join(output_dir, 'ablation_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Comparison table saved to {csv_path}")
        
        # Create visualization
        viz_path = os.path.join(output_dir, 'ablation_visualization.png')
        self.visualize_ablation_results(results, viz_path)
        
        # Generate LaTeX table
        latex_path = os.path.join(output_dir, 'ablation_table.tex')
        self._generate_latex_table(df, latex_path)
        
        self.logger.info(f"LaTeX table saved to {latex_path}")
    
    def _generate_latex_table(self, df: pd.DataFrame, output_path: str):
        """Generate LaTeX table for paper inclusion"""
        
        # Select relevant columns
        columns = ['Configuration', 'accuracy', 'f1', 'sharpe', 'accuracy_drop_pct']
        df_latex = df[columns].copy()
        
        # Format numbers
        df_latex['accuracy'] = df_latex['accuracy'].apply(lambda x: f"{x:.3f}")
        df_latex['f1'] = df_latex['f1'].apply(lambda x: f"{x:.3f}")
        df_latex['sharpe'] = df_latex['sharpe'].apply(lambda x: f"{x:.2f}")
        df_latex['accuracy_drop_pct'] = df_latex['accuracy_drop_pct'].apply(
            lambda x: f"{x:.1f}\\%" if pd.notna(x) else "—"
        )
        
        # Rename columns
        df_latex.columns = ['Configuration', 'Accuracy', 'F1-Score', 'Sharpe', 'Acc. Drop (\\%)']
        
        # Generate LaTeX
        latex_str = df_latex.to_latex(index=False, escape=False)
        
        # Add table environment
        latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Ablation Study Results: Impact of Removing System Components}}
\\label{{tab:ablation}}
{latex_str}
\\end{{table}}
"""
        
        with open(output_path, 'w') as f:
            f.write(latex_table)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock configuration
    config = {
        'model': {'text_encoder': {'attention': True}},
        'temporal': {'half_life_days': 3},
        'features': {'technical_indicators': ['SMA', 'RSI']}
    }
    
    # Mock model class
    class MockModel:
        pass
    
    # Create ablation study
    ablation = AblationStudy(config, MockModel)
    
    # Mock data
    data = (
        np.random.randn(100, 30, 768),  # X_text
        np.random.randn(100, 30, 50),   # X_num
        np.random.randint(0, 3, 100)    # y
    )
    
    # Mock functions
    def mock_train(config, data):
        return MockModel()
    
    def mock_evaluate(model, data):
        return {
            'accuracy': np.random.uniform(0.6, 0.7),
            'f1': np.random.uniform(0.58, 0.68),
            'sharpe': np.random.uniform(0.9, 1.3),
            'cagr': np.random.uniform(0.1, 0.2)
        }
    
    # Run ablation study
    results = ablation.run_complete_ablation_study(
        mock_train,
        mock_evaluate,
        data,
        configs_to_run=['full_model', 'no_attention', 'no_temporal_decay']
    )
    
    # Save results
    ablation.save_results(results)
    
    print("\n" + "="*70)
    print("Ablation study framework ready!")
    print("Systematically evaluates contribution of each component")
    print("="*70)
