import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for professional plots
sns.set_theme(style="whitegrid")

class ThesisStatisticalAnalyzer:
	def __init__(self, metrics_data_path=None):
		"""
		Initializes the analyzer. 
		If metrics_data_path is provided, it loads the CSV.
		Otherwise, it waits for data via add_data methods.
		"""
		self.df = pd.DataFrame()
		if metrics_data_path and os.path.exists(metrics_data_path):
			self.df = pd.read_csv(metrics_data_path)
			
	def load_data_from_dict(self, stats_dict):
		"""
		Converts the dictionary structure from your 'analyzer.py' 
		into a Pandas DataFrame suitable for this statistical module.
		
		Expected input format (from analyzer.py):
		{
			'Classic': [{'run': 0, 'hv': 0.5, ...}, ...],
			'Hybrid': [{'run': 0, 'hv': 0.6, ...}, ...]
		}
		"""
		rows = []
		# Assuming both algos have same number of runs and order
		# We need to merge them by Run ID to create paired rows
		
		classic_data = stats_dict.get('Classic', [])
		hybrid_data = stats_dict.get('Hybrid', [])
		
		# Create a lookup map for easier merging
		hybrid_map = {item['run']: item for item in hybrid_data}
		
		for c_item in classic_data:
			run_id = c_item['run']
			if run_id in hybrid_map:
				h_item = hybrid_map[run_id]
				
				row = {'RunID': run_id}
				
				# Metrics to process
				metric_keys = ['igd_plus', 'gd_plus', 'hv', 'spacing']
				
				for m in metric_keys:
					if m in c_item:
						row[f'{m}_Classic'] = c_item[m]
					if m in h_item:
						row[f'{m}_Hybrid'] = h_item[m]
						
				rows.append(row)
				
		self.df = pd.DataFrame(rows)
		print(f"[StatAnalyzer] Data loaded. {len(self.df)} valid paired observations.")

	def _cliffs_delta(self, x, y):
		"""
		Calculates Cliff's Delta effect size (Non-parametric).
		Returns: delta value, interpretation string
		"""
		m, n = len(x), len(y)
		dom = 0
		for i in x:
			for j in y:
				if i > j: dom += 1
				elif i < j: dom -= 1
		delta = dom / (m * n)
		
		# Interpretation
		abs_d = abs(delta)
		if abs_d < 0.147: size = "Negligible"
		elif abs_d < 0.33: size = "Small"
		elif abs_d < 0.474: size = "Medium"
		else: size = "Large"
		
		return delta, size

	def _cohens_d(self, x, y):
		"""
		Calculates Cohen's d effect size (Parametric).
		"""
		nx = len(x)
		ny = len(y)
		dof = nx + ny - 2
		return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

	def perform_paired_analysis(self, alpha=0.05):
		"""
		Performs robust paired statistical analysis.
		Logic: Shapiro-Wilk on diff -> T-Test (if normal) OR Wilcoxon (if not).
		Adds Effect Size calculation.
		"""
		metrics = ['igd_plus', 'gd_plus', 'hv', 'spacing'] # Standardized names
		results = []

		print("\n--- Running Statistical Tests ---")

		for metric in metrics:
			col_classic = f'{metric}_Classic'
			col_hybrid = f'{metric}_Hybrid'

			if col_classic not in self.df.columns or col_hybrid not in self.df.columns:
				continue

			# Drop NaNs for robust handling
			clean_df = self.df[[col_classic, col_hybrid]].dropna()
			
			data_classic = clean_df[col_classic]
			data_hybrid = clean_df[col_hybrid]
			
			n = len(clean_df)
			if n < 8:
				print(f"Warning: Skipping {metric} due to insufficient data (n={n})")
				continue

			# 1. Calculate Differences
			differences = data_hybrid - data_classic

			# 2. Normality Test (Shapiro-Wilk) on DIFFERENCES
			# Note: For paired tests, we check normality of the *difference*, not the groups.
			stat_shapiro, p_shapiro = stats.shapiro(differences)
			is_normal = p_shapiro > alpha

			# 3. Choose Test & Effect Size
			if is_normal:
				test_name = "Paired T-Test"
				stat_test, p_value = stats.ttest_rel(data_hybrid, data_classic)
				
				# Effect Size: Cohen's d
				eff_size = self._cohens_d(data_hybrid, data_classic)
				eff_interp = "Small" if abs(eff_size) < 0.5 else "Medium" if abs(eff_size) < 0.8 else "Large"
				
			else:
				test_name = "Wilcoxon Signed-Rank"
				# alternative='two-sided' is default
				stat_test, p_value = stats.wilcoxon(data_hybrid, data_classic)
				
				# Effect Size: Cliff's Delta (Non-parametric)
				eff_size, eff_interp = self._cliffs_delta(data_hybrid, data_classic)

			# 4. Interpretation
			# Mean comparison to see WHO won
			mean_classic = data_classic.mean()
			mean_hybrid = data_hybrid.mean()
			
			winner = "Inconclusive"
			if p_value <= alpha:
				if metric in ['hv']: # Higher is better
					winner = "Hybrid" if mean_hybrid > mean_classic else "Classic"
				else: # Lower is better (IGD+, GD+, Spacing)
					winner = "Hybrid" if mean_hybrid < mean_classic else "Classic"
			else:
				winner = "No Signif. Diff"

			results.append({
				"Metric": metric,
				"N": n,
				"Normality (p)": round(p_shapiro, 4),
				"Test Used": test_name,
				"Test Stat": round(stat_test, 2),
				"P-Value": p_value, # Keep precision
				"Signif?": "YES" if p_value <= alpha else "NO",
				"Winner": winner,
				"Effect Size": round(eff_size, 3),
				"Effect Magnitude": eff_interp
			})

		return pd.DataFrame(results).set_index("Metric")

	def generate_boxplots(self, output_dir="plots"):
		"""
		Generates professional boxplots for the thesis.
		"""
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
			
		metrics = ['igd_plus', 'gd_plus', 'hv', 'spacing']
		metric_labels = {
			'igd_plus': 'IGD+ (Lower is Better)',
			'gd_plus': 'GD+ (Lower is Better)',
			'hv': 'Hypervolume (Higher is Better)',
			'spacing': 'Spacing (Lower is Better)'
		}

		# Melt dataframe for Seaborn
		# From wide (Classic, Hybrid columns) to long (Algorithm, Value columns)
		for metric in metrics:
			col_classic = f'{metric}_Classic'
			col_hybrid = f'{metric}_Hybrid'
			
			if col_classic not in self.df.columns: continue
			
			# Prepare long format data
			df_long = pd.melt(self.df, 
							  value_vars=[col_classic, col_hybrid],
							  var_name='Algorithm', 
							  value_name='Value')
			
			# Clean Algorithm names (remove _Classic/_Hybrid suffix)
			df_long['Algorithm'] = df_long['Algorithm'].apply(lambda x: x.split('_')[-1])
			
			plt.figure(figsize=(6, 5))
			sns.boxplot(x='Algorithm', y='Value', data=df_long, palette="Set2", width=0.5)
			sns.stripplot(x='Algorithm', y='Value', data=df_long, color=".3", size=4, alpha=0.6) # Add jitter points
			
			plt.title(f'Comparison of {metric.upper()}', fontsize=14)
			plt.ylabel(metric_labels[metric], fontsize=12)
			plt.xlabel('')
			
			# Save
			filename = f"{output_dir}/{metric}_boxplot.png"
			plt.savefig(filename, dpi=300, bbox_inches='tight')
			plt.close()
			print(f"[Plot] Saved {filename}")

# --- EXAMPLE USAGE INTEGRATION ---
if __name__ == "__main__":
	# 1. Load data (In reality, you pass 'stats' from analyzer.py)
	# Assuming 'stats' dictionary is available from the previous step
	# stats = analyzer.compute_metrics() 
	
	# DUMMY DATA FOR DEMO
	dummy_stats = {
		'Classic': [{'run': i, 'hv': 0.5 + np.random.normal(0, 0.05), 'igd_plus': 0.1 + np.random.normal(0, 0.01)} for i in range(30)],
		'Hybrid': [{'run': i, 'hv': 0.65 + np.random.normal(0, 0.05), 'igd_plus': 0.05 + np.random.normal(0, 0.01)} for i in range(30)]
	}
	
	stat_tool = ThesisStatisticalAnalyzer()
	stat_tool.load_data_from_dict(dummy_stats)
	
	# 2. Run Statistics
	results_table = stat_tool.perform_paired_analysis()
	
	print("\n=== FINAL STATISTICAL RESULTS ===")
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', 1000)
	print(results_table)
	
	# 3. Generate Plots
	stat_tool.generate_boxplots()