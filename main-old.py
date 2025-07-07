#!/usr/bin/env python3
# main.py - Ultimate Auto Trading System Runner
# รันระบบเทรดอัตโนมัติครบวงจร

import os
import sys
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from utils.config_manager import get_config
from utils.logger import get_logger
from data.data_manager import DataManager
from ai.deep_learning_models import DeepLearningModelFactory
from ai.reinforcement_learning import RLModelManager
from backtesting.advanced_backtesting import AdvancedBacktestManager

class UltimateAutoTradingSystem:
    """ระบบเทรดอัตโนมัติครบวงจร"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the Ultimate Auto Trading System"""
        
        # Load configuration
        self.config = get_config(config_path)
        self.logger = get_logger(config_path)
        
        # Initialize components
        self.data_manager = DataManager()
        self.rl_manager = RLModelManager()
        self.backtest_manager = AdvancedBacktestManager()
        
        # System state
        self.current_models = {}
        self.backtest_results = {}
        self.paper_trade_results = {}
        
        # Targets from config
        self.targets = self.config.get('targets')
        self.target_ai_win_rate = self.targets['ai_win_rate']
        self.target_backtest_return = self.targets['backtest_return']
        self.target_paper_trade_return = self.targets['paper_trade_return']
        
        # System info
        system_info = self.config.get_system_info()
        self.logger.system_status("initialized", 
                                f"v{system_info['version']} | "
                                f"Capital: ${system_info['initial_capital']:,} | "
                                f"GPU: {system_info['gpu_enabled']}")
        
        # Create directories
        self.config.create_directories()
        
        # Setup GPU
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU configuration"""
        try:
            import torch
            
            if torch.cuda.is_available() and self.config.get('gpu.enabled'):
                # Set memory fraction
                memory_fraction = self.config.get('gpu.memory_fraction', 0.6)
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                
                # Log GPU info
                gpu_name = torch.cuda.get_device_name()
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                used_memory = total_memory * memory_fraction
                
                self.logger.gpu_status(gpu_name, used_memory, total_memory)
                
            else:
                self.logger.warning("GPU not available or disabled, using CPU")
                
        except ImportError:
            self.logger.warning("PyTorch not available")
    
    async def run_full_pipeline(self):
        """Run the complete trading pipeline"""
        
        try:
            self.logger.info("Starting Ultimate Auto Trading Pipeline")
            self.logger.info(f"Targets: AI {self.target_ai_win_rate}%, "
                           f"Backtest {self.target_backtest_return}%, "
                           f"Paper Trade {self.target_paper_trade_return}%")
            
            # Step 1: Train AI Models
            self.logger.info("="*60)
            self.logger.info("STEP 1: AI MODEL TRAINING")
            self.logger.info("="*60)
            
            training_success = await self._train_ai_models()
            if not training_success:
                self.logger.error("AI training failed, stopping pipeline")
                return False
            
            # Step 2: Advanced Backtesting
            self.logger.info("="*60)
            self.logger.info("STEP 2: ADVANCED BACKTESTING")
            self.logger.info("="*60)
            
            backtest_success = await self._run_advanced_backtesting()
            if not backtest_success:
                self.logger.error("Backtesting failed, stopping pipeline")
                return False
            
            # Step 3: Paper Trading
            self.logger.info("="*60)
            self.logger.info("STEP 3: PAPER TRADING")
            self.logger.info("="*60)
            
            paper_trade_success = await self._run_paper_trading()
            if not paper_trade_success:
                self.logger.error("Paper trading failed, stopping pipeline")
                return False
            
            # Step 4: Final Results
            self.logger.info("="*60)
            self.logger.info("STEP 4: FINAL RESULTS")
            self.logger.info("="*60)
            
            await self._generate_final_results()
            
            self.logger.success("Ultimate Auto Trading Pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _train_ai_models(self) -> bool:
        """Train AI models until target win rate achieved"""
        
        try:
            max_attempts = self.config.get('retry.max_training_attempts', 10)
            
            for attempt in range(1, max_attempts + 1):
                self.logger.training_attempt(attempt, max_attempts)
                
                try:
                    # Generate training data
                    symbols = self.config.get('symbols')
                    samples_per_symbol = self.config.get('training.samples_per_symbol', 1000)
                    
                    training_data = {}
                    for symbol in symbols:
                        data = self.data_manager.generate_synthetic_data(symbol, samples_per_symbol)
                        training_data[symbol] = data
                    
                    # Train traditional ML models
                    ml_results = await self._train_traditional_ml(training_data)
                    
                    # Train deep learning models
                    dl_results = await self._train_deep_learning(training_data)
                    
                    # Train reinforcement learning models
                    rl_results = await self._train_reinforcement_learning(training_data)
                    
                    # Combine all results
                    all_results = {**ml_results, **dl_results, **rl_results}
                    
                    # Find best model
                    best_model_name, best_model, best_metrics = self._find_best_model(all_results)
                    
                    if best_model_name and best_metrics.get('win_rate', 0) >= self.target_ai_win_rate:
                        self.logger.target_achieved("AI Win Rate", 
                                                   best_metrics['win_rate'], 
                                                   self.target_ai_win_rate)
                        
                        # Save best model
                        self.current_models['best'] = {
                            'name': best_model_name,
                            'model': best_model,
                            'metrics': best_metrics
                        }
                        
                        # Save all models for ensemble
                        self.current_models['all'] = all_results
                        
                        return True
                    
                    else:
                        best_win_rate = best_metrics.get('win_rate', 0) if best_metrics else 0
                        self.logger.warning(f"Attempt {attempt}: Best win rate {best_win_rate:.2f}% "
                                          f"< Target {self.target_ai_win_rate}%")
                        
                        # Wait before retry
                        await asyncio.sleep(self.config.get('retry.delay_between_attempts', 5))
                
                except Exception as e:
                    self.logger.error(f"Training attempt {attempt} failed: {e}")
                    continue
            
            self.logger.error(f"Failed to achieve target win rate after {max_attempts} attempts")
            return False
            
        except Exception as e:
            self.logger.error(f"AI training failed: {e}")
            return False
    
    async def _train_traditional_ml(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train traditional ML models"""
        
        try:
            self.logger.info("Training traditional ML models...")
            
            # Import traditional ML trainer
            from ai.traditional_ml import TraditionalMLTrainer
            
            ml_trainer = TraditionalMLTrainer()
            results = await ml_trainer.train_all_models(training_data)
            
            self.logger.success(f"Traditional ML training completed: {len(results)} models")
            return results
            
        except Exception as e:
            self.logger.error(f"Traditional ML training failed: {e}")
            return {}
    
    async def _train_deep_learning(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train deep learning models"""
        
        try:
            self.logger.info("Training deep learning models...")
            
            # Import deep learning trainer
            from ai.deep_learning_trainer import DeepLearningTrainer
            
            dl_trainer = DeepLearningTrainer()
            results = await dl_trainer.train_all_models(training_data)
            
            self.logger.success(f"Deep learning training completed: {len(results)} models")
            return results
            
        except Exception as e:
            self.logger.error(f"Deep learning training failed: {e}")
            return {}
    
    async def _train_reinforcement_learning(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train reinforcement learning models"""
        
        try:
            self.logger.info("Training reinforcement learning models...")
            
            # Combine training data
            combined_data = []
            for symbol, data in training_data.items():
                combined_data.append(data)
            
            import pandas as pd
            full_data = pd.concat(combined_data, ignore_index=True)
            
            # Train RL models
            rl_results = self.rl_manager.train_all_rl_models(full_data)
            
            self.logger.success(f"RL training completed: {len(rl_results)} models")
            return rl_results
            
        except Exception as e:
            self.logger.error(f"RL training failed: {e}")
            return {}
    
    def _find_best_model(self, all_results: Dict[str, Any]) -> Tuple[str, Any, Dict[str, float]]:
        """Find the best performing model"""
        
        best_model_name = None
        best_model = None
        best_metrics = None
        best_score = 0
        
        for model_name, result in all_results.items():
            if isinstance(result, tuple) and len(result) == 2:
                model, metrics = result
            elif isinstance(result, dict) and 'model' in result:
                model = result['model']
                metrics = result.get('metrics', {})
            else:
                continue
            
            # Calculate composite score
            win_rate = metrics.get('win_rate', metrics.get('mean_reward', 0))
            accuracy = metrics.get('accuracy', metrics.get('final_mean_reward', 0))
            
            # Weighted score
            score = win_rate * 0.7 + accuracy * 30
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model = model
                best_metrics = metrics
        
        return best_model_name, best_model, best_metrics
    
    async def _run_advanced_backtesting(self) -> bool:
        """Run advanced backtesting"""
        
        try:
            max_attempts = self.config.get('retry.max_backtest_attempts', 5)
            
            for attempt in range(1, max_attempts + 1):
                self.logger.info(f"Backtesting attempt {attempt}/{max_attempts}")
                
                try:
                    # Get best model
                    best_model_info = self.current_models.get('best')
                    if not best_model_info:
                        self.logger.error("No trained model available for backtesting")
                        return False
                    
                    # Generate test data
                    symbols = self.config.get('symbols')
                    test_data = {}
                    
                    for symbol in symbols:
                        data = self.data_manager.generate_synthetic_data(symbol, 1000)
                        test_data[symbol] = data
                    
                    # Run backtesting on each symbol
                    backtest_results = {}
                    total_return = 0
                    
                    for symbol, data in test_data.items():
                        try:
                            # Run comprehensive backtest
                            symbol_results = self.backtest_manager.run_comprehensive_backtest(
                                data, 
                                best_model_info['model'],
                                None,  # scaler would be needed for real models
                                initial_cash=10000
                            )
                            
                            backtest_results[symbol] = symbol_results
                            
                            # Get best return from different frameworks
                            best_return = 0
                            for framework, result in symbol_results.items():
                                if framework != 'comparison' and 'total_return' in result:
                                    best_return = max(best_return, result['total_return'])
                            
                            total_return += best_return
                            
                            self.logger.backtest_result(
                                symbol, 10000, 10000 * (1 + best_return/100), 
                                best_return, 0
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Backtest failed for {symbol}: {e}")
                            continue
                    
                    # Calculate average return
                    avg_return = total_return / len(symbols) if symbols else 0
                    
                    # Check if target achieved
                    if avg_return >= self.target_backtest_return:
                        self.logger.target_achieved("Backtest Return", 
                                                   avg_return, 
                                                   self.target_backtest_return)
                        
                        self.backtest_results = {
                            'average_return': avg_return,
                            'symbol_results': backtest_results,
                            'target_achieved': True
                        }
                        
                        return True
                    
                    else:
                        self.logger.warning(f"Backtest attempt {attempt}: Return {avg_return:.2f}% "
                                          f"< Target {self.target_backtest_return}%")
                        
                        # If not target achieved, retrain models
                        if attempt < max_attempts:
                            self.logger.info("Retraining models for better backtest performance...")
                            await self._train_ai_models()
                        
                        await asyncio.sleep(self.config.get('retry.delay_between_attempts', 5))
                
                except Exception as e:
                    self.logger.error(f"Backtesting attempt {attempt} failed: {e}")
                    continue
            
            self.logger.error(f"Failed to achieve target backtest return after {max_attempts} attempts")
            return False
            
        except Exception as e:
            self.logger.error(f"Advanced backtesting failed: {e}")
            return False
    
    async def _run_paper_trading(self) -> bool:
        """Run paper trading"""
        
        try:
            max_attempts = self.config.get('retry.max_paper_trade_attempts', 3)
            
            for attempt in range(1, max_attempts + 1):
                self.logger.info(f"Paper trading attempt {attempt}/{max_attempts}")
                
                try:
                    # Get paper trade config
                    paper_config = self.config.get_paper_trade_config()
                    
                    # Import paper trading system
                    from paper_trading.realtime_paper_trading import RealtimePaperTrader
                    
                    paper_trader = RealtimePaperTrader(
                        self.current_models['best']['model'],
                        None,  # scaler
                        paper_config
                    )
                    
                    # Run paper trading
                    self.logger.info(f"Running paper trading for {paper_config['duration_minutes']} minutes...")
                    
                    paper_result = await paper_trader.run_paper_trading(
                        duration_minutes=paper_config['duration_minutes'],
                        update_interval=paper_config['update_interval']
                    )
                    
                    # Check results
                    total_return = paper_result.get('total_return', 0)
                    
                    if total_return >= self.target_paper_trade_return:
                        self.logger.target_achieved("Paper Trade Return", 
                                                   total_return, 
                                                   self.target_paper_trade_return)
                        
                        self.paper_trade_results = paper_result
                        
                        # Log detailed results
                        self.logger.paper_trade_result(
                            paper_result['initial_capital'],
                            paper_result['final_portfolio_value'],
                            total_return,
                            paper_result['total_trades'],
                            f"{paper_config['duration_minutes']} minutes"
                        )
                        
                        return True
                    
                    else:
                        self.logger.warning(f"Paper trading attempt {attempt}: Return {total_return:.2f}% "
                                          f"< Target {self.target_paper_trade_return}%")
                        
                        # Retrain if not last attempt
                        if attempt < max_attempts:
                            self.logger.info("Retraining models for better paper trading performance...")
                            await self._train_ai_models()
                            await self._run_advanced_backtesting()
                        
                        await asyncio.sleep(self.config.get('retry.delay_between_attempts', 5))
                
                except Exception as e:
                    self.logger.error(f"Paper trading attempt {attempt} failed: {e}")
                    continue
            
            self.logger.error(f"Failed to achieve target paper trade return after {max_attempts} attempts")
            return False
            
        except Exception as e:
            self.logger.error(f"Paper trading failed: {e}")
            return False
    
    async def _generate_final_results(self):
        """Generate final results and reports"""
        
        try:
            # Create final results
            final_results = {
                'timestamp': datetime.now(),
                'system_info': self.config.get_system_info(),
                'targets': self.targets,
                'targets_achieved': {
                    'ai_win_rate': self.current_models.get('best', {}).get('metrics', {}).get('win_rate', 0) >= self.target_ai_win_rate,
                    'backtest_return': self.backtest_results.get('average_return', 0) >= self.target_backtest_return,
                    'paper_trade_return': self.paper_trade_results.get('total_return', 0) >= self.target_paper_trade_return
                },
                'best_model': {
                    'name': self.current_models.get('best', {}).get('name'),
                    'metrics': self.current_models.get('best', {}).get('metrics', {})
                },
                'backtest_results': self.backtest_results,
                'paper_trade_results': self.paper_trade_results
            }
            
            # Save results
            import json
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"results/ultimate_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            self.logger.save_session(results_file)
            
            # Generate summary report
            self._generate_summary_report(final_results)
            
            # Generate performance plots
            await self._generate_performance_plots(final_results)
            
        except Exception as e:
            self.logger.error(f"Error generating final results: {e}")
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate summary report"""
        
        try:
            report = []
            report.append("🏆 ULTIMATE AUTO TRADING SYSTEM - FINAL REPORT")
            report.append("=" * 70)
            
            # System info
            system_info = results['system_info']
            report.append(f"\n📊 SYSTEM INFORMATION:")
            report.append(f"   System: {system_info['name']} v{system_info['version']}")
            report.append(f"   Initial Capital: ${system_info['initial_capital']:,}")
            report.append(f"   Symbols: {', '.join(system_info['symbols'])}")
            report.append(f"   GPU Enabled: {system_info['gpu_enabled']}")
            
            # Targets
            targets = results['targets']
            targets_achieved = results['targets_achieved']
            report.append(f"\n🎯 TARGETS & ACHIEVEMENTS:")
            report.append(f"   AI Win Rate: {targets['ai_win_rate']}% "
                         f"{'✅' if targets_achieved['ai_win_rate'] else '❌'}")
            report.append(f"   Backtest Return: {targets['backtest_return']}% "
                         f"{'✅' if targets_achieved['backtest_return'] else '❌'}")
            report.append(f"   Paper Trade Return: {targets['paper_trade_return']}% "
                         f"{'✅' if targets_achieved['paper_trade_return'] else '❌'}")
            
            # Best model
            best_model = results['best_model']
            if best_model['name']:
                report.append(f"\n🤖 BEST MODEL:")
                report.append(f"   Model: {best_model['name']}")
                metrics = best_model['metrics']
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report.append(f"   {key}: {value:.4f}")
            
            # Performance summary
            if results['backtest_results']:
                br = results['backtest_results']
                report.append(f"\n🔙 BACKTEST SUMMARY:")
                report.append(f"   Average Return: {br.get('average_return', 0):.2f}%")
                report.append(f"   Target Achieved: {'✅' if br.get('target_achieved') else '❌'}")
            
            if results['paper_trade_results']:
                ptr = results['paper_trade_results']
                report.append(f"\n📄 PAPER TRADE SUMMARY:")
                report.append(f"   Total Return: {ptr.get('total_return', 0):.2f}%")
                report.append(f"   Total Trades: {ptr.get('total_trades', 0)}")
                report.append(f"   Win Rate: {ptr.get('win_rate', 0):.2f}%")
            
            # Overall status
            all_targets_achieved = all(targets_achieved.values())
            report.append(f"\n🏁 FINAL STATUS:")
            if all_targets_achieved:
                report.append("   🎉 ALL TARGETS ACHIEVED! System ready for live trading.")
            else:
                report.append("   ⚠️  Some targets not achieved. Review and retrain if needed.")
            
            report.append("\n" + "=" * 70)
            
            # Print and log report
            report_text = "\n".join(report)
            print(report_text)
            self.logger.info("Final report generated")
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"results/summary_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(report_text)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
    
    async def _generate_performance_plots(self, results: Dict[str, Any]):
        """Generate performance visualization plots"""
        
        try:
            # Import plotting libraries
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('dark_background')
            
            # Create performance summary plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Targets vs Achieved
            targets = results['targets']
            achieved = results['targets_achieved']
            
            target_names = ['AI Win Rate', 'Backtest Return', 'Paper Trade Return']
            target_values = [targets['ai_win_rate'], targets['backtest_return'], targets['paper_trade_return']]
            achieved_status = [achieved['ai_win_rate'], achieved['backtest_return'], achieved['paper_trade_return']]
            
            colors = ['#00ff41' if status else '#ff4444' for status in achieved_status]
            bars = ax1.bar(target_names, target_values, color=colors, alpha=0.7)
            ax1.set_title('🎯 Target Achievement Status', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Target Value (%)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value, status in zip(bars, target_values, achieved_status):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value}% {"✅" if status else "❌"}',
                        ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Model Performance (if available)
            if 'best_model' in results and results['best_model']['metrics']:
                metrics = results['best_model']['metrics']
                metric_names = list(metrics.keys())[:5]  # Show top 5 metrics
                metric_values = [metrics[name] for name in metric_names]
                
                ax2.bar(metric_names, metric_values, color='#00ff41', alpha=0.7)
                ax2.set_title('🤖 Best Model Performance', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Metric Value')
                ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Backtest Results (if available)
            if 'backtest_results' in results and results['backtest_results'].get('symbol_results'):
                symbol_results = results['backtest_results']['symbol_results']
                symbols = list(symbol_results.keys())
                returns = []
                
                for symbol in symbols:
                    # Get best return from different frameworks
                    best_return = 0
                    for framework, result in symbol_results[symbol].items():
                        if framework != 'comparison' and isinstance(result, dict) and 'total_return' in result:
                            best_return = max(best_return, result['total_return'])
                    returns.append(best_return)
                
                ax3.bar(symbols, returns, color='#4ecdc4', alpha=0.7)
                ax3.set_title('🔙 Backtest Returns by Symbol', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Return (%)')
                ax3.axhline(y=targets['backtest_return'], color='#ff0040', linestyle='--', alpha=0.7, label=f'Target ({targets["backtest_return"]}%)')
                ax3.legend()
            
            # Plot 4: Paper Trade Summary (if available)
            if 'paper_trade_results' in results and results['paper_trade_results']:
                ptr = results['paper_trade_results']
                
                summary_data = {
                    'Total Return': ptr.get('total_return', 0),
                    'Win Rate': ptr.get('win_rate', 0),
                    'Max Drawdown': ptr.get('max_drawdown', 0),
                    'Sharpe Ratio': ptr.get('sharpe_ratio', 0)
                }
                
                bars = ax4.bar(summary_data.keys(), summary_data.values(), color='#96ceb4', alpha=0.7)
                ax4.set_title('📄 Paper Trade Performance', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Value')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, summary_data.values()):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.2f}',
                            ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"results/performance_summary_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            
            self.logger.info(f"Performance plots saved to: {plot_file}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping plots")
        except Exception as e:
            self.logger.error(f"Error generating performance plots: {e}")

async def main():
    """Main entry point"""
    
    print("🚀 ULTIMATE AUTO TRADING SYSTEM")
    print("=" * 70)
    print("🎯 Target: 90% AI Win Rate → 85% Backtest Return → 100% Paper Trade Return")
    print("⚡ GPU: 60% Memory Limit")
    print("🔄 Fully Automated Pipeline")
    print("=" * 70)
    
    try:
        # Initialize system
        system = UltimateAutoTradingSystem()
        
        # Run full pipeline
        success = await system.run_full_pipeline()
        
        if success:
            print("\n🎉 MISSION ACCOMPLISHED!")
            print("✅ All targets achieved")
            print("🚀 System ready for live trading")
        else:
            print("\n❌ MISSION FAILED")
            print("⚠️  Some targets not achieved")
            print("🔄 Review logs and retry")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  System interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 System error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Run the ultimate system
    exit_code = asyncio.run(main())
    sys.exit(exit_code)