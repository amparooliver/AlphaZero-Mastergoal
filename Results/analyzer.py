import pstats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import re
from scipy import stats

class ProfilerComparison:
    def __init__(self, baseline_profile, optimized_profile, baseline_training=None, optimized_training=None):
        """
        Initialize the comparison tool with paths to profiler and training data.
        
        Args:
            baseline_profile: Path to the baseline profile results (.prof file)
            optimized_profile: Path to the optimized profile results (.prof file)
            baseline_training: Path to baseline training metrics (CSV or JSON)
            optimized_training: Path to optimized training metrics (CSV or JSON)
        """
        self.baseline_profile_path = baseline_profile
        self.optimized_profile_path = optimized_profile
        self.baseline_training_path = baseline_training
        self.optimized_training_path = optimized_training
        
        # Load profile data
        self.baseline_stats = pstats.Stats(baseline_profile)
        self.optimized_stats = pstats.Stats(optimized_profile)
        
        # Will hold processed data
        self.baseline_profile_df = None
        self.optimized_profile_df = None
        self.training_comparison_df = None
        
        # Analysis results
        self.results = {
            "runtime_comparison": {},
            "hotspot_analysis": {},
            "memory_patterns": {},
            "training_metrics": {},
            "efficiency_metrics": {},
            "optimization_impact": {}
        }

    def process_profile_data(self):
        """Convert pstats data to pandas DataFrames for easier analysis"""
        # Process baseline profile
        baseline_data = []
        self.baseline_stats.sort_stats('cumulative')
        for func, (cc, nc, tt, ct, callers) in self.baseline_stats.stats.items():
            file_path, line, func_name = func
            if 'site-packages' not in file_path:  # Filter out library code
                baseline_data.append({
                    'file_path': file_path,
                    'line': line,
                    'function': func_name,
                    'calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'time_per_call': tt/nc if nc > 0 else 0
                })
        self.baseline_profile_df = pd.DataFrame(baseline_data)
        
        # Process optimized profile
        optimized_data = []
        self.optimized_stats.sort_stats('cumulative')
        for func, (cc, nc, tt, ct, callers) in self.optimized_stats.stats.items():
            file_path, line, func_name = func
            if 'site-packages' not in file_path:  # Filter out library code
                optimized_data.append({
                    'file_path': file_path,
                    'line': line,
                    'function': func_name,
                    'calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'time_per_call': tt/nc if nc > 0 else 0
                })
        self.optimized_profile_df = pd.DataFrame(optimized_data)
        
        return self.baseline_profile_df, self.optimized_profile_df

    def compare_runtimes(self):
        """Compare total runtime and function-level performance metrics"""
        if self.baseline_profile_df is None or self.optimized_profile_df is None:
            self.process_profile_data()
            
        # Total runtime comparison
        baseline_total = self.baseline_profile_df['total_time'].sum()
        optimized_total = self.optimized_profile_df['total_time'].sum()
        
        self.results["runtime_comparison"] = {
            "baseline_total_runtime": baseline_total,
            "optimized_total_runtime": optimized_total,
            "improvement_percentage": ((baseline_total - optimized_total) / baseline_total) * 100,
            "speedup_factor": baseline_total / optimized_total if optimized_total > 0 else float('inf')
        }
        
        # Function-level comparison
        # Create a merged dataframe with functions that appear in both profiles
        merged_df = pd.merge(
            self.baseline_profile_df, 
            self.optimized_profile_df,
            on=['file_path', 'function'], 
            suffixes=('_baseline', '_optimized'),
            how='outer'
        ).fillna(0)
        
        # Calculate improvements
        merged_df['time_diff'] = merged_df['total_time_baseline'] - merged_df['total_time_optimized']
        merged_df['time_improvement_pct'] = (merged_df['time_diff'] / merged_df['total_time_baseline']) * 100
        merged_df['calls_diff'] = merged_df['calls_baseline'] - merged_df['calls_optimized']
        merged_df['time_per_call_improvement_pct'] = ((merged_df['time_per_call_baseline'] - merged_df['time_per_call_optimized']) / 
                                                     merged_df['time_per_call_baseline']) * 100
        
        # Find top improved functions
        top_improved = merged_df.sort_values('time_diff', ascending=False).head(10)
        
        self.results["function_improvements"] = top_improved.to_dict('records')
        
        return self.results["runtime_comparison"], top_improved

    def analyze_hotspots(self):
        """Identify and compare hotspots between the two profiles"""
        if self.baseline_profile_df is None or self.optimized_profile_df is None:
            self.process_profile_data()
            
        # Get top 10 hotspots from each profile
        baseline_hotspots = self.baseline_profile_df.sort_values('cumulative_time', ascending=False).head(10)
        optimized_hotspots = self.optimized_profile_df.sort_values('cumulative_time', ascending=False).head(10)
        
        # Find hotspots that were fixed (present in baseline but not in top of optimized)
        baseline_top_funcs = set([(row['file_path'], row['function']) for _, row in baseline_hotspots.iterrows()])
        optimized_top_funcs = set([(row['file_path'], row['function']) for _, row in optimized_hotspots.iterrows()])
        
        fixed_hotspots = baseline_top_funcs - optimized_top_funcs
        new_hotspots = optimized_top_funcs - baseline_top_funcs
        
        self.results["hotspot_analysis"] = {
            "baseline_top_hotspots": baseline_hotspots[['function', 'file_path', 'cumulative_time']].to_dict('records'),
            "optimized_top_hotspots": optimized_hotspots[['function', 'file_path', 'cumulative_time']].to_dict('records'),
            "fixed_hotspots": [{"file_path": f, "function": func} for f, func in fixed_hotspots],
            "new_hotspots": [{"file_path": f, "function": func} for f, func in new_hotspots]
        }
        
        return self.results["hotspot_analysis"]

    def load_training_data(self):
        """Load training metrics from files"""
        # Skip if paths not provided
        if not self.baseline_training_path or not self.optimized_training_path:
            return None, None
            
        # Determine file type and load accordingly
        baseline_ext = os.path.splitext(self.baseline_training_path)[1].lower()
        optimized_ext = os.path.splitext(self.optimized_training_path)[1].lower()
        
        if baseline_ext == '.csv':
            baseline_training = pd.read_csv(self.baseline_training_path)
        elif baseline_ext in ['.json', '.jsonl']:
            with open(self.baseline_training_path, 'r') as f:
                baseline_training = pd.DataFrame(json.load(f))
        else:
            baseline_training = None
            
        if optimized_ext == '.csv':
            optimized_training = pd.read_csv(self.optimized_training_path)
        elif optimized_ext in ['.json', '.jsonl']:
            with open(self.optimized_training_path, 'r') as f:
                optimized_training = pd.DataFrame(json.load(f))
        else:
            optimized_training = None
            
        return baseline_training, optimized_training

    def analyze_training_metrics(self):
        """Compare training metrics between baseline and optimized versions"""
        baseline_training, optimized_training = self.load_training_data()
        
        if baseline_training is None or optimized_training is None:
            self.results["training_metrics"] = {"error": "Training data not available"}
            return self.results["training_metrics"]
            
        # Ensure both have similar columns and structure
        metrics = ['loss', 'policy_loss', 'value_loss', 'total_loss']
        available_metrics = [m for m in metrics if m in baseline_training.columns and m in optimized_training.columns]
        
        metrics_results = {}
        
        for metric in available_metrics:
            # Calculate convergence speed
            threshold = baseline_training[metric].min() * 1.1  # 10% above minimum loss
            
            baseline_convergence = np.argmax(baseline_training[metric].values <= threshold)
            optimized_convergence = np.argmax(optimized_training[metric].values <= threshold)
            
            if baseline_convergence == 0 and baseline_training[metric].values[0] > threshold:
                baseline_convergence = len(baseline_training)
            if optimized_convergence == 0 and optimized_training[metric].values[0] > threshold:
                optimized_convergence = len(optimized_training)
                
            # Final values
            baseline_final = baseline_training[metric].iloc[-1]
            optimized_final = optimized_training[metric].iloc[-1]
            
            # Stability (standard deviation over last 10% of training)
            baseline_stability = baseline_training[metric].iloc[-int(len(baseline_training)*0.1):].std()
            optimized_stability = optimized_training[metric].iloc[-int(len(optimized_training)*0.1):].std()
            
            metrics_results[metric] = {
                "baseline_epochs_to_converge": int(baseline_convergence) if baseline_convergence < len(baseline_training) else "Did not converge",
                "optimized_epochs_to_converge": int(optimized_convergence) if optimized_convergence < len(optimized_training) else "Did not converge",
                "convergence_improvement": ((baseline_convergence - optimized_convergence) / baseline_convergence * 100) if baseline_convergence > 0 and optimized_convergence < len(optimized_training) else "N/A",
                "baseline_final_value": baseline_final,
                "optimized_final_value": optimized_final,
                "final_value_improvement": ((baseline_final - optimized_final) / baseline_final * 100) if baseline_final > 0 else "N/A",
                "baseline_stability": baseline_stability,
                "optimized_stability": optimized_stability,
                "stability_improvement": ((baseline_stability - optimized_stability) / baseline_stability * 100) if baseline_stability > 0 else "N/A"
            }
            
        self.results["training_metrics"] = metrics_results
        return metrics_results

    def calculate_efficiency_metrics(self):
        """Calculate efficiency metrics that combine profiling and training data"""
        if "runtime_comparison" not in self.results or not self.results["runtime_comparison"]:
            self.compare_runtimes()
            
        baseline_training, optimized_training = self.load_training_data()
        
        if baseline_training is None or optimized_training is None:
            self.results["efficiency_metrics"] = {"error": "Training data not available for efficiency calculations"}
            return self.results["efficiency_metrics"]
            
        # Metrics to analyze
        metrics = ['loss', 'policy_loss', 'value_loss', 'total_loss']
        available_metrics = [m for m in metrics if m in baseline_training.columns and m in optimized_training.columns]
        
        baseline_runtime = self.results["runtime_comparison"]["baseline_total_runtime"]
        optimized_runtime = self.results["runtime_comparison"]["optimized_total_runtime"]
        
        efficiency_results = {}
        
        for metric in available_metrics:
            # Time to quality ratio
            baseline_final = baseline_training[metric].iloc[-1]
            optimized_final = optimized_training[metric].iloc[-1]
            
            # Measure of how much loss reduction we get per second of computation
            baseline_efficiency = (baseline_training[metric].iloc[0] - baseline_final) / baseline_runtime
            optimized_efficiency = (optimized_training[metric].iloc[0] - optimized_final) / optimized_runtime
            
            efficiency_results[metric] = {
                "baseline_time_to_quality": baseline_efficiency,
                "optimized_time_to_quality": optimized_efficiency,
                "efficiency_improvement": (optimized_efficiency / baseline_efficiency - 1) * 100 if baseline_efficiency > 0 else "N/A"
            }
            
        self.results["efficiency_metrics"] = efficiency_results
        return efficiency_results

    def generate_report(self, output_path="profiling_comparison_report.html"):
        """Generate a comprehensive HTML report with visualizations"""
        # Ensure we have all data
        if self.baseline_profile_df is None:
            self.process_profile_data()
        
        if "runtime_comparison" not in self.results or not self.results["runtime_comparison"]:
            self.compare_runtimes()
            
        if "hotspot_analysis" not in self.results or not self.results["hotspot_analysis"]:
            self.analyze_hotspots()
            
        if "training_metrics" not in self.results:
            self.analyze_training_metrics()
            
        if "efficiency_metrics" not in self.results:
            self.calculate_efficiency_metrics()
            
        # Generate visualizations
        fig_runtime = plt.figure(figsize=(10, 6))
        plt.bar(['Baseline', 'Optimized'], 
                [self.results["runtime_comparison"]["baseline_total_runtime"], 
                 self.results["runtime_comparison"]["optimized_total_runtime"]])
        plt.title('Total Runtime Comparison')
        plt.ylabel('Time (seconds)')
        plt.savefig('runtime_comparison.png')
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Python Project Performance Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .improved {{ color: green; }}
                .degraded {{ color: red; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Performance Optimization Analysis Report</h1>
            <div class="section">
                <h2>Runtime Comparison</h2>
                <p>Baseline Total Runtime: {self.results["runtime_comparison"]["baseline_total_runtime"]:.4f} seconds</p>
                <p>Optimized Total Runtime: {self.results["runtime_comparison"]["optimized_total_runtime"]:.4f} seconds</p>
                <p>Improvement: <span class="improved">{self.results["runtime_comparison"]["improvement_percentage"]:.2f}%</span></p>
                <p>Speedup Factor: <span class="improved">{self.results["runtime_comparison"]["speedup_factor"]:.2f}x</span></p>
                <div class="chart">
                    <img src="runtime_comparison.png" alt="Runtime Comparison Chart">
                </div>
            </div>
            
            <div class="section">
                <h2>Top Function Improvements</h2>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>File</th>
                        <th>Baseline Time (s)</th>
                        <th>Optimized Time (s)</th>
                        <th>Improvement (%)</th>
                        <th>Calls Difference</th>
                    </tr>
        """
        
        # Add function improvements
        for func in self.results["function_improvements"][:10]:
            html_content += f"""
                    <tr>
                        <td>{func['function']}</td>
                        <td>{os.path.basename(func['file_path'])}</td>
                        <td>{func['total_time_baseline']:.4f}</td>
                        <td>{func['total_time_optimized']:.4f}</td>
                        <td class="{'improved' if func['time_improvement_pct'] > 0 else 'degraded'}">{func['time_improvement_pct']:.2f}%</td>
                        <td>{int(func['calls_diff'])}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Hotspot Analysis</h2>
                <h3>Baseline Top Hotspots</h3>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>File</th>
                        <th>Cumulative Time (s)</th>
                    </tr>
        """
        
        # Add baseline hotspots
        for hotspot in self.results["hotspot_analysis"]["baseline_top_hotspots"]:
            html_content += f"""
                    <tr>
                        <td>{hotspot['function']}</td>
                        <td>{os.path.basename(hotspot['file_path'])}</td>
                        <td>{hotspot['cumulative_time']:.4f}</td>
                    </tr>
            """
            
        html_content += """
                </table>
                
                <h3>Optimized Top Hotspots</h3>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>File</th>
                        <th>Cumulative Time (s)</th>
                    </tr>
        """
        
        # Add optimized hotspots
        for hotspot in self.results["hotspot_analysis"]["optimized_top_hotspots"]:
            html_content += f"""
                    <tr>
                        <td>{hotspot['function']}</td>
                        <td>{os.path.basename(hotspot['file_path'])}</td>
                        <td>{hotspot['cumulative_time']:.4f}</td>
                    </tr>
            """
        
        # Add training metrics if available
        if "training_metrics" in self.results and "error" not in self.results["training_metrics"]:
            html_content += """
            </table>
            </div>
            
            <div class="section">
                <h2>Training Metrics Comparison</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline Epochs to Converge</th>
                        <th>Optimized Epochs to Converge</th>
                        <th>Convergence Improvement</th>
                        <th>Baseline Final Value</th>
                        <th>Optimized Final Value</th>
                        <th>Final Value Improvement</th>
                    </tr>
            """
            
            for metric, data in self.results["training_metrics"].items():
                html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{data['baseline_epochs_to_converge']}</td>
                        <td>{data['optimized_epochs_to_converge']}</td>
                        <td class="{'improved' if data['convergence_improvement'] != 'N/A' and float(data['convergence_improvement']) > 0 else 'degraded'}">{data['convergence_improvement']}%</td>
                        <td>{data['baseline_final_value']:.6f}</td>
                        <td>{data['optimized_final_value']:.6f}</td>
                        <td class="{'improved' if data['final_value_improvement'] != 'N/A' and float(data['final_value_improvement']) > 0 else 'degraded'}">{data['final_value_improvement']}%</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
            
            # Add efficiency metrics
            html_content += """
            <div class="section">
                <h2>Efficiency Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline Efficiency</th>
                        <th>Optimized Efficiency</th>
                        <th>Improvement</th>
                    </tr>
            """
            
            for metric, data in self.results["efficiency_metrics"].items():
                html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{data['baseline_time_to_quality']:.6f}</td>
                        <td>{data['optimized_time_to_quality']:.6f}</td>
                        <td class="{'improved' if data['efficiency_improvement'] != 'N/A' and float(data['efficiency_improvement']) > 0 else 'degraded'}">{data['efficiency_improvement']}%</td>
                    </tr>
                """
                
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
            <div class="section">
                <h2>Conclusion</h2>
                <p>Overall assessment of the optimization impact:</p>
        """
        
        # Generate conclusion
        if self.results["runtime_comparison"]["improvement_percentage"] > 20:
            html_content += f"""
                <p>The optimizations have resulted in a <span class="improved">significant performance improvement of {self.results["runtime_comparison"]["improvement_percentage"]:.2f}%</span> 
                in total runtime, with a speedup factor of {self.results["runtime_comparison"]["speedup_factor"]:.2f}x.</p>
            """
        elif self.results["runtime_comparison"]["improvement_percentage"] > 5:
            html_content += f"""
                <p>The optimizations have resulted in a <span class="improved">moderate performance improvement of {self.results["runtime_comparison"]["improvement_percentage"]:.2f}%</span> 
                in total runtime, with a speedup factor of {self.results["runtime_comparison"]["speedup_factor"]:.2f}x.</p>
            """
        else:
            html_content += f"""
                <p>The optimizations have resulted in a <span class="{'improved' if self.results["runtime_comparison"]["improvement_percentage"] > 0 else 'degraded'}">
                minimal performance change of {self.results["runtime_comparison"]["improvement_percentage"]:.2f}%</span> 
                in total runtime, with a speedup factor of {self.results["runtime_comparison"]["speedup_factor"]:.2f}x.</p>
            """
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write the HTML report
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path

    def plot_comparisons(self, output_dir="comparison_plots"):
        """Generate detailed comparison plots"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load training data if available
        baseline_training, optimized_training = self.load_training_data()
        
        if baseline_training is not None and optimized_training is not None:
            # Plot training curves
            metrics = ['loss', 'policy_loss', 'value_loss', 'total_loss']
            available_metrics = [m for m in metrics if m in baseline_training.columns and m in optimized_training.columns]
            
            for metric in available_metrics:
                plt.figure(figsize=(10, 6))
                plt.plot(baseline_training.index, baseline_training[metric], label='Baseline')
                plt.plot(optimized_training.index, optimized_training[metric], label='Optimized')
                plt.title(f'{metric.replace("_", " ").title()} Comparison')
                plt.xlabel('Epoch')
                plt.ylabel(metric.replace("_", " ").title())
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
                plt.close()
                
                # Log scale plot
                plt.figure(figsize=(10, 6))
                plt.semilogy(baseline_training.index, baseline_training[metric], label='Baseline')
                plt.semilogy(optimized_training.index, optimized_training[metric], label='Optimized')
                plt.title(f'{metric.replace("_", " ").title()} Comparison (Log Scale)')
                plt.xlabel('Epoch')
                plt.ylabel(f'{metric.replace("_", " ").title()} (log scale)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(output_dir, f'{metric}_comparison_log.png'))
                plt.close()
        
        # Plot function-level comparisons
        if self.baseline_profile_df is None:
            self.process_profile_data()
            
        # Merge datasets for comparison
        merged_df = pd.merge(
            self.baseline_profile_df, 
            self.optimized_profile_df,
            on=['file_path', 'function'], 
            suffixes=('_baseline', '_optimized'),
            how='outer'
        ).fillna(0)
        
        # Select top functions by baseline time
        top_funcs = merged_df.sort_values('total_time_baseline', ascending=False).head(15)
        
        # Plot top functions comparison
        plt.figure(figsize=(12, 8))
        bar_width = 0.35
        index = np.arange(len(top_funcs))
        
        plt.barh(index, top_funcs['total_time_baseline'], bar_width, label='Baseline')
        plt.barh(index + bar_width, top_funcs['total_time_optimized'], bar_width, label='Optimized')
        
        plt.yticks(index + bar_width/2, [f"{func} ({os.path.basename(path)})" for path, func in 
                                         zip(top_funcs['file_path'], top_funcs['function'])])
        plt.xlabel('Time (seconds)')
        plt.title('Top Functions by Execution Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_functions_comparison.png'))
        plt.close()
        
        return output_dir


def analyze_profiles(baseline_profile, optimized_profile, baseline_training=None, optimized_training=None, 
                     output_dir="profile_analysis", report_name="profiling_report.html"):
    """
    Main function to analyze and compare profiling results.
    
    Args:
        baseline_profile: Path to baseline profile results (.prof file)
        optimized_profile: Path to optimized profile results (.prof file)
        baseline_training: Path to baseline training metrics (optional)
        optimized_training: Path to optimized training metrics (optional)
        output_dir: Directory to save results
        report_name: Name of the HTML report file
        
    Returns:
        Path to the generated HTML report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize comparison tool
    comparison = ProfilerComparison(
        baseline_profile=baseline_profile,
        optimized_profile=optimized_profile,
        baseline_training=baseline_training,
        optimized_training=optimized_training
    )
    
    # Process profiles
    comparison.process_profile_data()
    
    # Run all analyses
    comparison.compare_runtimes()
    comparison.analyze_hotspots()
    
    if baseline_training and optimized_training:
        comparison.analyze_training_metrics()
        comparison.calculate_efficiency_metrics()
    
    # Generate plots
    plot_dir = os.path.join(output_dir, "plots")
    comparison.plot_comparisons(plot_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, report_name)
    comparison.generate_report(report_path)
    
    return report_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Python project performance profiles")
    parser.add_argument("--baseline", required=True, help="Path to baseline profile file (.prof)")
    parser.add_argument("--optimized", required=True, help="Path to optimized profile file (.prof)")
    parser.add_argument("--baseline-training", help="Path to baseline training metrics CSV/JSON")
    parser.add_argument("--optimized-training", help="Path to optimized training metrics CSV/JSON")
    parser.add_argument("--output", default="profile_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    report_path = analyze_profiles(
        baseline_profile=args.baseline,
        optimized_profile=args.optimized,
        baseline_training=args.baseline_training,
        optimized_training=args.optimized_training,
        output_dir=args.output
    )
    
    print(f"Analysis complete! Report generated at: {report_path}")