import os
import sys
import shutil
import pandas as pd
import numpy as np
# Import Modul dari Repo
from problem_generator import generateProblem
from problem import Problem
from lp_generator import create_VMP_MOMILP_File
from pamilo_runner import PaMILORunner
from analyzer import ExperimentAnalyzer
from nsga2_classic import NSGA2Classic
from nsga2_hybrid import NSGA2Hybrid
import gurobipy as gp

def setup_environment():
	# 1. Mount Drive
	from google.colab import drive
	drive.mount('/content/drive')
	
	# 2. Setup Path
	BASE_PATH = '/content/drive/MyDrive/Skripsi'
	# Pastikan folder ini ada
	DATASET_DIR = os.path.join(BASE_PATH, 'dataset')
	RESULTS_DIR = os.path.join(BASE_PATH, 'results')
	
	# Buat folder jika belum ada
	os.makedirs(DATASET_DIR, exist_ok=True)
	os.makedirs(os.path.join(RESULTS_DIR, 'lp_files'), exist_ok=True)
	os.makedirs(os.path.join(RESULTS_DIR, 'pamilo_sols'), exist_ok=True)
	os.makedirs(os.path.join(RESULTS_DIR, 'raw_fronts'), exist_ok=True) # Folder baru untuk CSV
	
	# 3. Setup License & Binary
	os.environ["GRB_LICENSE_FILE"] = os.path.join(BASE_PATH, 'gurobi.lic')
	
	# Pastikan binary executable
	pamilo_bin = os.path.join(BASE_PATH, 'bin', 'pamilo_cli')
	if os.path.exists(pamilo_bin):
		os.chmod(pamilo_bin, 0o755)
		
	return DATASET_DIR, RESULTS_DIR, pamilo_bin

def run_experiment_pipeline():
	DATASET_DIR, RESULTS_DIR, PAMILO_EXE = setup_environment()

	# Inisialisasi Tools
	pamilo_runner = PaMILORunner(PAMILO_EXE)
	analyzer = ExperimentAnalyzer()

	# --- FITUR RESUME: LOAD DATA LAMA DARI DRIVE ---
	RAW_FRONTS_DIR = os.path.join(RESULTS_DIR, 'raw_fronts')
	analyzer.loadResultsFromDirectory(RAW_FRONTS_DIR)

	# --- TAHAP 1: GENERATE PROBLEM ---
	if not os.listdir(DATASET_DIR):
		print("\n--- Generating Datasets ---")
		for i in range(1, 6):
			generateProblem(os.path.join(DATASET_DIR, f"small_{i}.json"), 'small', i)
			generateProblem(os.path.join(DATASET_DIR, f"large_{i}.json"), 'large', 100+i)

	# --- TAHAP 2: LOOP EKSPERIMEN ---
	problem_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".json")])

	# Prioritaskan Small
	small_files = [f for f in problem_files if 'small' in f]
	large_files = [f for f in problem_files if 'large' in f]
	sorted_files = small_files + large_files

	for p_file in sorted_files:
		scenario_name = p_file.replace(".json", "")
		full_path = os.path.join(DATASET_DIR, p_file)
		
		print(f"\n>>> SCENARIO: {scenario_name}")
		problem = Problem()
		problem.loadFromFile(full_path)

		# A. PaMILO (Small Only)
		if 'small' in scenario_name:
			lp_path = os.path.join(RESULTS_DIR, "lp_files", f"{scenario_name}.lp")
			# Base path tanpa ekstensi json (PaMILO nambah sendiri)
			sol_base = os.path.join(RESULTS_DIR, "pamilo_sols", scenario_name)
			expected_sol = sol_base + "_sol.json"

			if not os.path.exists(lp_path):
				create_VMP_MOMILP_File(problem, lp_path)
			
			if not os.path.exists(expected_sol):
				print(f"   [PaMILO] Solving...")
				success = pamilo_runner.solve(lp_path, sol_base, timeout_sec=3600)
			else:
				print("   [PaMILO] Found existing solution.")
				success = True
			
			if success: analyzer.loadPamiloReference(expected_sol)

		# B. NSGA-II (30 Runs)
		TOTAL_RUNS = 30
		
		# Siapkan folder untuk menyimpan CSV per run
		current_raw_dir = os.path.join(RAW_FRONTS_DIR, scenario_name)
		
		print(f"   [NSGA] Processing {TOTAL_RUNS} runs...")
		
		for run_id in range(TOTAL_RUNS):
			run_key = f"{scenario_name}_r{run_id}"
			
			# Cek apakah Run ini SUDAH ADA di analyzer (berarti sudah diload dari Drive)
			# Jika sudah ada dan datanya tidak kosong, Skip.
			if (run_key in analyzer.results['Classic'] and 
				len(analyzer.results['Classic'][run_key]) > 0 and
				run_key in analyzer.results['Hybrid'] and
				len(analyzer.results['Hybrid'][run_key]) > 0):
				
				# print(f"	 > Run {run_id} Skipped (Done)", end="\r")
				continue

			print(f"\r	 > Run {run_id+1}/{TOTAL_RUNS} Running...", end="")
			
			# Seed
			base_seed = int(''.join(filter(str.isdigit, scenario_name)) or 0)
			seed = 1000 + (base_seed * 100) + run_id

			# -- CLASSIC --
			# Cek file CSV individu dulu (double check)
			csv_classic = os.path.join(current_raw_dir, f"Classic_r{run_id}.csv")
			if not os.path.exists(csv_classic):
				algo_c = NSGA2Classic(problem, 100, 100, 0.9, 0.1)
				algo_c.setSeed(seed)
				algo_c.run()
				# SIMPAN KE DRIVE
				analyzer.addResult('Classic', run_key, algo_c.population, save_path=csv_classic)

			# -- HYBRID --
			csv_hybrid = os.path.join(current_raw_dir, f"Hybrid_r{run_id}.csv")
			if not os.path.exists(csv_hybrid):
				algo_h = NSGA2Hybrid(problem, 100, 100, 0.9, 0.1)
				algo_h.setSeed(seed)
				algo_h.run()
				# SIMPAN KE DRIVE
				analyzer.addResult('Hybrid', run_key, algo_h.population, save_path=csv_hybrid)
		
		print(" Done.")

	# --- TAHAP 3: METRICS & SUMMARY ---
	print("\n--- Calculating Final Metrics ---")
	final_stats = analyzer.computeMetrics()
	
	# Save to CSV (Summary)
	flat_data = []
	for algo in ['Classic', 'Hybrid']:
		if algo in final_stats:
			for m in final_stats[algo]:
				row = m.copy()
				row['Algorithm'] = algo
				flat_data.append(row)
	
	if flat_data:
		df_res = pd.DataFrame(flat_data)
		summary_path = os.path.join(RESULTS_DIR, 'final_metrics_summary.csv')
		df_res.to_csv(summary_path, index=False)
		print(f"[DONE] Summary saved to {summary_path}")

if __name__ == "__main__":
	run_experiment_pipeline()