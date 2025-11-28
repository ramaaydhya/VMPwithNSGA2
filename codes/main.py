import os
import sys
import shutil
import pandas as pd
import numpy as np
import time
import subprocess
import json
import glob
import gurobipy as gp

# --- IMPORT MODUL DARI REPO ---
if '/content/VMPwithNSGA2/codes' not in sys.path:
	sys.path.append('/content/VMPwithNSGA2/codes')

from problem_generator import generateProblem
from problem import Problem
from lp_generator import create_VMP_MOMILP_File
from experiment_analyzer import ExperimentAnalyzer
from nsga2_classic import NSGA2Classic
from nsga2_hybrid import NSGA2Hybrid

# --- KONFIGURASI PATH ---
REPO_ROOT = '/content/VMPwithNSGA2'
BIN_PATH = os.path.join(REPO_ROOT, 'bin', 'pamilo_cli')

# Path Lokal (Symlink ke Drive)
LOCAL_DATASET_DIR = os.path.join(REPO_ROOT, 'dataset')
LOCAL_RESULTS_DIR = os.path.join(REPO_ROOT, 'results')

# Path Asli di Google Drive
DRIVE_BASE = '/content/drive/MyDrive/Skripsi'
DRIVE_DATASET_DIR = os.path.join(DRIVE_BASE, 'dataset')
DRIVE_RESULTS_DIR = os.path.join(DRIVE_BASE, 'results')

# --- SWITCH KONTROL ---
ENABLE_PAMILO = False
ENABLE_NSGA   = True

# --- RUNNER HELPER ---
class PaMILORunnerFixed:
	def __init__(self, exec_path):
		self.exec_path = exec_path

	def solve(self, input_lp, output_base_arg, timeout_sec=900):
		if not os.path.exists(self.exec_path): return False, None
		os.makedirs(os.path.dirname(output_base_arg), exist_ok=True)
		
		cmd = [self.exec_path, input_lp, "-o", output_base_arg, "-t", str(os.cpu_count() or 2)]
		print(f"   [PaMILO] Executing (Timeout {timeout_sec}s)...")
		
		try:
			start = time.time()
			subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
			elapsed = time.time() - start
			
			# Cek output (PaMILO menambah _sol.json)
			expected_file = output_base_arg + "_sol.json"
			if os.path.exists(expected_file):
				print(f"   [PaMILO] Success ({elapsed:.1f}s). Saved to Drive.")
				return True, expected_file
			return False, None
		except subprocess.TimeoutExpired:
			print(f"   [PaMILO] Timeout reached ({timeout_sec}s).")
			return False, None
		except Exception as e:
			print(f"   [PaMILO] Error: {e}")
			return False, None

def setup_dependencies():
	print("\n--- 🔧 Setup Dependencies ---")
	# 1. Lisensi
	lic_paths = [os.path.join(REPO_ROOT, 'gurobi.lic'), os.path.join(DRIVE_BASE, 'gurobi.lic')]
	for lic in lic_paths:
		if os.path.exists(lic):
			os.environ["GRB_LICENSE_FILE"] = lic
			print(f"🔑 Lisensi: {lic}")
			break
	else:
		print("❌ FATAL: File 'gurobi.lic' tidak ditemukan!")
		return False

	# 2. Library Path
	import gurobipy
	lib_path = os.path.join(os.path.dirname(gurobipy.__file__), '.libs')
	if os.path.exists(lib_path):
		os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH','')}"
		print(f"🔧 Library Path: {lib_path}")

	# 3. Izin Binary
	if os.path.exists(BIN_PATH):
		os.chmod(BIN_PATH, 0o755)
	else:
		print(f"❌ Binary Missing: {BIN_PATH}")
		return False
	return True

def prepare_directories():
	print("\n--- 📂 Directory Sync (Colab <-> Drive) ---")
	os.makedirs(DRIVE_DATASET_DIR, exist_ok=True)
	os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)
	
	# Symlink Dataset
	if os.path.exists(LOCAL_DATASET_DIR):
		if os.path.islink(LOCAL_DATASET_DIR): os.unlink(LOCAL_DATASET_DIR)
		else: shutil.rmtree(LOCAL_DATASET_DIR)
	os.symlink(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR)
	
	# Symlink Results
	if os.path.exists(LOCAL_RESULTS_DIR):
		if os.path.islink(LOCAL_RESULTS_DIR): os.unlink(LOCAL_RESULTS_DIR)
		else: shutil.rmtree(LOCAL_RESULTS_DIR)
	os.symlink(DRIVE_RESULTS_DIR, LOCAL_RESULTS_DIR)
	
	# Subfolders
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'lp_files'), exist_ok=True)
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'pamilo_sols'), exist_ok=True)
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'raw_fronts'), exist_ok=True)
	print("✅ Drive Connected.")

def run_pipeline():
	prepare_directories()
	if not setup_dependencies(): return
	
	pamilo = PaMILORunnerFixed(BIN_PATH)
	analyzer = ExperimentAnalyzer()

	# ==========================================
	# TAHAP 1: CEK & GENERATE DATASET
	# ==========================================
	# Hapus file lama jika ingin regenerate dengan seed baru!
	# Code ini hanya generate jika file belum ada.
	
	required_files = [f"small_{i}.json" for i in range(1,6)] + [f"large_{i}.json" for i in range(1,6)]
	missing_files = [f for f in required_files if not os.path.exists(os.path.join(LOCAL_DATASET_DIR, f))]
	
	if missing_files:
		print(f"\n--- 🎲 Generating {len(missing_files)} Missing Datasets ---")
		for i in range(1, 6):
			s_path = os.path.join(LOCAL_DATASET_DIR, f"small_{i}.json")
			l_path = os.path.join(LOCAL_DATASET_DIR, f"large_{i}.json")
			
			if not os.path.exists(s_path):
				# SEED SMALL: 10 + i
				generateProblem(s_path, 'small', 10+i)
			
			if not os.path.exists(l_path):
				# SEED LARGE: 1000 + i
				generateProblem(l_path, 'large', 1000+i)
	else:
		print("\n[Info] All datasets found in Drive. Skipping generation.")

	# ==========================================
	# TAHAP 2: LOOP EKSPERIMEN
	# ==========================================
	all_f = [f for f in os.listdir(LOCAL_DATASET_DIR) if f.endswith(".json")]
	# Prioritas Small
	problem_files = sorted(all_f, key=lambda x: (0 if 'small' in x else 1, x))

	for p_file in problem_files:
		scen_name = p_file.replace(".json", "")
		print(f"\n>>> SCENARIO: {scen_name}")
		
		problem = Problem()
		problem.loadFromFile(os.path.join(LOCAL_DATASET_DIR, p_file))

		# --- A. PAMILO (Check Drive First) ---
		if ENABLE_PAMILO and 'small' in scen_name:
			lp_path = os.path.join(LOCAL_RESULTS_DIR, "lp_files", f"{scen_name}.lp")
			base_out = os.path.join(LOCAL_RESULTS_DIR, "pamilo_sols", scen_name)
			final_json = base_out + "_sol.json"

			if os.path.exists(final_json):
				print("   [PaMILO] Found in Drive. Loaded.")
				analyzer.loadPamiloReference(final_json)
			else:
				if not os.path.exists(lp_path):
					create_VMP_MOMILP_File(problem, lp_path)
				
				# REVISI: Timeout 900 detik (15 Menit)
				success, path = pamilo.solve(lp_path, base_out, timeout_sec=900)
				if success: analyzer.loadPamiloReference(path)

		# --- B. NSGA-II (Check Drive per Run) ---
		if ENABLE_NSGA:
			TOTAL_RUNS = 30
			raw_dir = os.path.join(LOCAL_RESULTS_DIR, "raw_fronts", scen_name)
			os.makedirs(raw_dir, exist_ok=True)
			
			print(f"   [NSGA] Checking {TOTAL_RUNS} runs...")
			
			for r in range(TOTAL_RUNS):
				csv_c = os.path.join(raw_dir, f"Classic_r{r}.csv")
				csv_h = os.path.join(raw_dir, f"Hybrid_r{r}.csv")
				
				if os.path.exists(csv_c) and os.path.exists(csv_h):
					continue 

				base_seed = int(''.join(filter(str.isdigit, scen_name)) or 0)
				seed = 1000 + (base_seed * 100) + r
				print(f"\r	 > Executing Run {r+1}/{TOTAL_RUNS}...", end="")

				if not os.path.exists(csv_c):
					ac = NSGA2Classic(problem, 100, 100, 0.9, 0.1)
					ac.setSeed(seed)
					ac.run(verbose=True)
					analyzer.addResult('Classic', f"{scen_name}_r{r}", ac.population, save_path=csv_c)
				
				if not os.path.exists(csv_h):
					ah = NSGA2Hybrid(problem, 100, 100, 0.9, 0.1)
					ah.setSeed(seed)
					ah.run(verbose=True)
					analyzer.addResult('Hybrid', f"{scen_name}_r{r}", ah.population, save_path=csv_h)
			print("\n	 > All runs synced to Drive.")

	# ==========================================
	# TAHAP 3: METRICS (Load All from Drive)
	# ==========================================
	if ENABLE_NSGA:
		print("\n--- 📊 Final Analysis ---")
		raw_root = os.path.join(LOCAL_RESULTS_DIR, "raw_fronts")
		analyzer.loadResultsFromDirectory(raw_root)

		final_stats = analyzer.computeMetrics()
		
		# Save Summary CSV
		flat = []
		for algo in ['Classic', 'Hybrid']:
			if final_stats[algo]:
				hv = np.mean([x['hv'] for x in final_stats[algo]])
				print(f"[{algo}] Avg HV: {hv:.4f}")
				for idx, m in enumerate(final_stats[algo]):
					m.update({'Algorithm': algo, 'RunID': idx})
					flat.append(m)
		
		if flat:
			pd.DataFrame(flat).to_csv(os.path.join(LOCAL_RESULTS_DIR, 'final_metrics.csv'), index=False)
			print("✅ Final Metrics saved to Drive.")

if __name__ == "__main__":
	run_pipeline()