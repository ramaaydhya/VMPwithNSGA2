import os
import sys
import shutil
import pandas as pd
import numpy as np
import time
import subprocess
import json
import gurobipy as gp
import glob

# Import Modul dari Repo
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

# Path Lokal (di dalam environment Colab/Repo)
LOCAL_DATASET_DIR = os.path.join(REPO_ROOT, 'dataset')
LOCAL_RESULTS_DIR = os.path.join(REPO_ROOT, 'results')

# Path Persisten (Google Drive)
DRIVE_BASE = '/content/drive/MyDrive/Skripsi'
DRIVE_DATASET_DIR = os.path.join(DRIVE_BASE, 'dataset')
DRIVE_RESULTS_DIR = os.path.join(DRIVE_BASE, 'results')

# --- SWITCH KONTROL ---
ENABLE_PAMILO = True
ENABLE_NSGA   = True

# --- CLASS RUNNER FIXED ---
class PaMILORunnerFixed:
	def __init__(self, exec_path):
		self.exec_path = exec_path

	def solve(self, input_lp, output_base_arg, timeout_sec=3600):
		if not os.path.exists(self.exec_path):
			print(f"   [Error] Binary not found: {self.exec_path}")
			return False, None

		os.makedirs(os.path.dirname(output_base_arg), exist_ok=True)
		threads = os.cpu_count() or 2
		cmd = [self.exec_path, input_lp, "-o", output_base_arg, "-t", str(threads)]
		
		print(f"   [PaMILO] Executing...")
		start = time.time()
		try:
			res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
			elapsed = time.time() - start
			expected_file = output_base_arg + "_sol.json"
			
			if res.returncode == 0 and os.path.exists(expected_file):
				print(f"   [PaMILO] Success ({elapsed:.1f}s). File: {os.path.basename(expected_file)}")
				return True, expected_file
			else:
				# Fallback stdout check
				content = res.stdout.strip()
				if content.startswith("{") and content.endswith("}"):
					with open(expected_file, 'w') as f: f.write(content)
					print(f"   [PaMILO] Recovered JSON from stdout.")
					return True, expected_file
				print(f"   [PaMILO] Failed. Code: {res.returncode}")
				return False, None
		except subprocess.TimeoutExpired:
			print(f"   [PaMILO] Timeout ({timeout_sec}s)")
			return False, None
		except Exception as e:
			print(f"   [PaMILO] Crash: {e}")
			return False, None

def setup_dependencies():
	print("\n--- 🔧 Setup Dependencies ---")
	# 1. Lisensi
	lic_paths = [os.path.join(REPO_ROOT, 'gurobi.lic'), os.path.join(DRIVE_BASE, 'gurobi.lic')]
	lic_found = False
	for lic in lic_paths:
		if os.path.exists(lic):
			os.environ["GRB_LICENSE_FILE"] = lic
			print(f"🔑 Lisensi: {lic}")
			lic_found = True
			break
	if not lic_found:
		print("❌ FATAL: File 'gurobi.lic' tidak ditemukan!")
		return False

	# 2. Library Path
	import gurobipy
	gurobi_home = os.path.dirname(gurobipy.__file__)
	lib_dirs = [os.path.join(gurobi_home, '.libs'), gurobi_home]
	for ld in lib_dirs:
		if os.path.exists(ld) and any("libgurobi" in f for f in os.listdir(ld)):
			os.environ['LD_LIBRARY_PATH'] = f"{ld}:{os.environ.get('LD_LIBRARY_PATH','')}"
			print(f"🔧 Library Path: {ld}")
			break
			
	# 3. Validasi
	try:
		gp.Model("check").optimize()
		print("✅ Gurobi Active!")
	except:
		print("❌ Gurobi Failed.")
		return False
		
	# 4. Binary
	if os.path.exists(BIN_PATH):
		os.chmod(BIN_PATH, 0o755)
	else:
		print(f"❌ Binary Missing: {BIN_PATH}")
		return False
	return True

def prepare_directories():
	"""
	Mengatur Symlink untuk Dataset dan Results.
	Sinkronisasi cerdas: GitHub -> Drive -> Colab.
	"""
	print("\n--- 📂 Directory Setup ---")
	
	# --- 1. SETUP DATASET FOLDER ---
	os.makedirs(DRIVE_DATASET_DIR, exist_ok=True)
	
	# Cek apakah Dataset ada di Repo Lokal (GitHub) tapi Drive kosong?
	# Jika ya, copy dari GitHub ke Drive (Initial Import)
	if os.path.exists(LOCAL_DATASET_DIR) and not os.path.islink(LOCAL_DATASET_DIR):
		repo_jsons = glob.glob(os.path.join(LOCAL_DATASET_DIR, "*.json"))
		drive_jsons = glob.glob(os.path.join(DRIVE_DATASET_DIR, "*.json"))
		
		if repo_jsons and not drive_jsons:
			print(f"📥 Mengimpor {len(repo_jsons)} dataset dari GitHub ke Drive...")
			for f in repo_jsons:
				shutil.copy2(f, DRIVE_DATASET_DIR)
	
	# Hapus folder lokal dataset (agar bisa diganti symlink)
	if os.path.exists(LOCAL_DATASET_DIR):
		if os.path.islink(LOCAL_DATASET_DIR): os.unlink(LOCAL_DATASET_DIR)
		else: shutil.rmtree(LOCAL_DATASET_DIR)
	
	# Buat Symlink Dataset: ./dataset -> Drive/Dataset
	os.symlink(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR)
	print(f"🔗 Dataset Link: {LOCAL_DATASET_DIR} -> {DRIVE_DATASET_DIR}")

	# --- 2. SETUP RESULTS FOLDER ---
	os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)
	
	if os.path.exists(LOCAL_RESULTS_DIR):
		if os.path.islink(LOCAL_RESULTS_DIR): os.unlink(LOCAL_RESULTS_DIR)
		else: shutil.rmtree(LOCAL_RESULTS_DIR)
			
	# Buat Symlink Results: ./results -> Drive/Results
	os.symlink(DRIVE_RESULTS_DIR, LOCAL_RESULTS_DIR)
	print(f"🔗 Results Link: {LOCAL_RESULTS_DIR} -> {DRIVE_RESULTS_DIR}")
	
	# Buat subfolder di Drive (via symlink)
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'lp_files'), exist_ok=True)
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'pamilo_sols'), exist_ok=True)
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'raw_fronts'), exist_ok=True)

def run_pipeline():
	prepare_directories()
	if not setup_dependencies(): return
	
	pamilo = PaMILORunnerFixed(BIN_PATH)
	analyzer = ExperimentAnalyzer()

	# --- A. GENERATE PROBLEM (Jika di Drive kosong) ---
	# Karena sudah di-link, cek ini mengecek isi Drive
	if not os.listdir(LOCAL_DATASET_DIR):
		print("\n--- 🎲 Generating Datasets (to Drive) ---")
		for i in range(1, 6):
			generateProblem(os.path.join(LOCAL_DATASET_DIR, f"small_{i}.json"), 'small', i)
			generateProblem(os.path.join(LOCAL_DATASET_DIR, f"large_{i}.json"), 'large', 100+i)
	
	# --- B. LOAD FILES (Small First) ---
	all_f = [f for f in os.listdir(LOCAL_DATASET_DIR) if f.endswith(".json")]
	problem_files = sorted([f for f in all_f if 'small' in f]) + sorted([f for f in all_f if 'large' in f])
	
	print(f"\n[Schedule] {len(problem_files)} Problems found in Drive.")

	# --- C. EKSEKUSI ---
	for p_file in problem_files:
		scen_name = p_file.replace(".json", "")
		full_path = os.path.join(LOCAL_DATASET_DIR, p_file)
		
		print(f"\n{'='*40}\n>>> SCENARIO: {scen_name}\n{'='*40}")
		problem = Problem()
		problem.loadFromFile(full_path)

		# 1. PaMILO
		if ENABLE_PAMILO and 'small' in scen_name:
			lp_path = os.path.join(LOCAL_RESULTS_DIR, "lp_files", f"{scen_name}.lp")
			base_out = os.path.join(LOCAL_RESULTS_DIR, "pamilo_sols", scen_name)
			exp_json = base_out + "_sol.json"

			if not os.path.exists(lp_path): create_VMP_MOMILP_File(problem, lp_path)
			
			if not os.path.exists(exp_json):
				success, final_path = pamilo.solve(lp_path, base_out)
			else:
				print("   [PaMILO] Exists in Drive.")
				success, final_path = True, exp_json
			
			if success: analyzer.loadPamiloReference(final_path)

		# 2. NSGA-II
		if ENABLE_NSGA:
			TOTAL_RUNS = 30
			print(f"   [NSGA] 30 Runs...")
			raw_dir = os.path.join(LOCAL_RESULTS_DIR, "raw_fronts", scen_name)
			os.makedirs(raw_dir, exist_ok=True)

			for r in range(TOTAL_RUNS):
				csv_c = os.path.join(raw_dir, f"Classic_r{r}.csv")
				csv_h = os.path.join(raw_dir, f"Hybrid_r{r}.csv")
				
				if os.path.exists(csv_c) and os.path.exists(csv_h):
					if (r+1)%5==0: print(f" [Skip {r+1}]", end="")
					continue # Resume logic

				base_seed = int(''.join(filter(str.isdigit, scen_name)) or 0)
				seed = 1000 + (base_seed * 100) + r
				print(f"\r   Run {r+1}/{TOTAL_RUNS}: ", end="")

				# Classic
				if not os.path.exists(csv_c):
					print("C", end="", flush=True)
					ac = NSGA2Classic(problem, 100, 100, 0.9, 0.1)
					ac.setSeed(seed)
					ac.run(verbose=False)
					analyzer.addResult('Classic', f"{scen_name}_r{r}", ac.population, save_path=csv_c)
				
				# Hybrid
				if not os.path.exists(csv_h):
					print(" | H", end="", flush=True)
					ah = NSGA2Hybrid(problem, 100, 100, 0.9, 0.1)
					ah.setSeed(seed)
					ah.run(verbose=False)
					analyzer.addResult('Hybrid', f"{scen_name}_r{r}", ah.population, save_path=csv_h)
			print(" Done.")

		# --- D. METRICS ---
		if ENABLE_NSGA:
			print("\n--- 📊 Computing Metrics ---")
			# Load ulang semua CSV dari Drive (Termasuk yg lama)
			raw_root = os.path.join(LOCAL_RESULTS_DIR, "raw_fronts")
			if os.path.exists(raw_root):
				analyzer.loadResultsFromDirectory(raw_root)

			final_stats = analyzer.computeMetrics()
			
			# Print & Save
			print("\n=== SUMMARY ===")
			flat = []
			for algo in ['Classic', 'Hybrid']:
				if final_stats[algo]:
					hv = np.mean([x['hv'] for x in final_stats[algo]])
					igd = np.mean([x['igd_plus'] for x in final_stats[algo]])
					print(f"[{algo}] HV={hv:.4f}, IGD+={igd:.4f}")
					
					for idx, m in enumerate(final_stats[algo]):
						m.update({'Algorithm': algo, 'RunID': idx})
						flat.append(m)
			
			if flat:
				pd.DataFrame(flat).to_csv(os.path.join(LOCAL_RESULTS_DIR, 'final_metrics.csv'), index=False)
				print("✅ Metrics saved to Drive.")

if __name__ == "__main__":
	run_pipeline()