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
# Pastikan Sel 1 (Clone) sudah dijalankan
if '/content/VMPwithNSGA2/codes' not in sys.path:
	sys.path.append('/content/VMPwithNSGA2/codes')

from problem_generator import generateProblem
from problem import Problem
from lp_generator import create_VMP_MOMILP_File
from analyzer import ExperimentAnalyzer
from nsga2_classic import NSGA2Classic
from nsga2_hybrid import NSGA2Hybrid

# --- KONFIGURASI PATH ---
REPO_ROOT = '/content/VMPwithNSGA2'
BIN_PATH = os.path.join(REPO_ROOT, 'bin', 'pamilo_cli')

LOCAL_DATASET_DIR = os.path.join(REPO_ROOT, 'dataset')
LOCAL_RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
DRIVE_RESULTS_DIR = '/content/drive/MyDrive/Skripsi/results'

# --- SWITCH KONTROL ---
ENABLE_PAMILO = True
ENABLE_NSGA   = True

# --- CLASS RUNNER FIXED (Override) ---
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
			
			# Prediksi nama file (PaMILO menambah _sol.json)
			expected_file = output_base_arg + "_sol.json"
			
			if res.returncode == 0 and os.path.exists(expected_file):
				print(f"   [PaMILO] Success ({elapsed:.1f}s). File: {os.path.basename(expected_file)}")
				return True, expected_file
			else:
				# Cek Stdout fallback
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
	
	# 1. Cek Lisensi
	lic_paths = [os.path.join(REPO_ROOT, 'gurobi.lic'), '/content/drive/MyDrive/Skripsi/gurobi.lic']
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

	# 2. Fix Library Path (LD_LIBRARY_PATH)
	import gurobipy
	gurobi_home = os.path.dirname(gurobipy.__file__)
	# Cari di folder instalasi dan subfolder .libs
	potential_libs = [os.path.join(gurobi_home, '.libs'), gurobi_home]
	
	lib_injected = False
	for lib_dir in potential_libs:
		if os.path.exists(lib_dir) and any("libgurobi" in f for f in os.listdir(lib_dir)):
			current_ld = os.environ.get('LD_LIBRARY_PATH', '')
			os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{current_ld}"
			print(f"🔧 Library Path Injected: {lib_dir}")
			lib_injected = True
			break
	
	if not lib_injected:
		print("⚠️ Warning: Library Gurobi (.so) tidak ditemukan otomatis.")

	# 3. Validasi Gurobi
	try:
		gp.Model("check").optimize()
		print("✅ Gurobi Active!")
	except:
		print("❌ Gurobi Activation Failed.")
		return False

	# 4. Izin Binary
	if os.path.exists(BIN_PATH):
		os.chmod(BIN_PATH, 0o755)
		print(f"✅ Binary Ready.")
	else:
		print(f"❌ Binary not found at {BIN_PATH}")
		return False

	return True

def prepare_directories():
	"""Symlink hasil ke Drive."""
	os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)
	if os.path.exists(LOCAL_RESULTS_DIR):
		if os.path.islink(LOCAL_RESULTS_DIR): os.unlink(LOCAL_RESULTS_DIR)
		else: shutil.rmtree(LOCAL_RESULTS_DIR)
	os.symlink(DRIVE_RESULTS_DIR, LOCAL_RESULTS_DIR)
	
	os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'lp_files'), exist_ok=True)
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'pamilo_sols'), exist_ok=True)
	# Folder baru untuk raw fronts
	os.makedirs(os.path.join(LOCAL_RESULTS_DIR, 'raw_fronts'), exist_ok=True)

def run_pipeline():
	prepare_directories()
	if not setup_dependencies(): return
	
	pamilo = PaMILORunnerFixed(BIN_PATH)
	analyzer = ExperimentAnalyzer()

	# --- A. GENERATE PROBLEM ---
	if not os.listdir(LOCAL_DATASET_DIR):
		print("\n--- Generating Datasets ---")
		for i in range(1, 6):
			generateProblem(os.path.join(LOCAL_DATASET_DIR, f"small_{i}.json"), 'small', i)
			generateProblem(os.path.join(LOCAL_DATASET_DIR, f"large_{i}.json"), 'large', 100+i)
	
	# --- B. SORT FILES (Small First) ---
	all_f = [f for f in os.listdir(LOCAL_DATASET_DIR) if f.endswith(".json")]
	problem_files = sorted([f for f in all_f if 'small' in f]) + sorted([f for f in all_f if 'large' in f])

	print(f"\n[Schedule] Processing {len(problem_files)} scenarios...")

	# --- C. LOOP EKSPERIMEN ---
	for p_file in problem_files:
		scen_name = p_file.replace(".json", "")
		full_path = os.path.join(LOCAL_DATASET_DIR, p_file)
		
		print(f"\n{'='*40}\n>>> SCENARIO: {scen_name}\n{'='*40}")
		
		problem = Problem()
		problem.loadFromFile(full_path)

		# 1. PaMILO (Hanya Small)
		if ENABLE_PAMILO and 'small' in scen_name:
			lp_path = os.path.join(LOCAL_RESULTS_DIR, "lp_files", f"{scen_name}.lp")
			pamilo_out_base = os.path.join(LOCAL_RESULTS_DIR, "pamilo_sols", scen_name)
			expected_json = pamilo_out_base + "_sol.json"

			if not os.path.exists(lp_path):
				create_VMP_MOMILP_File(problem, lp_path)
			
			if not os.path.exists(expected_json):
				success, final_path = pamilo.solve(lp_path, pamilo_out_base, timeout_sec=3600)
			else:
				print("   [PaMILO] Solution exists. Loading...")
				success, final_path = True, expected_json
			
			if success and final_path:
				analyzer.loadPamiloReference(final_path)
		
		# 2. NSGA-II
		if ENABLE_NSGA:
			TOTAL_RUNS = 30
			print(f"   [NSGA] 30 Runs...")
			
			# Siapkan folder simpan per skenario
			raw_dir = os.path.join(LOCAL_RESULTS_DIR, "raw_fronts", scen_name)
			os.makedirs(raw_dir, exist_ok=True)

			for r in range(TOTAL_RUNS):
				# Cek jika file sudah ada (Resume capability)
				csv_c = os.path.join(raw_dir, f"Classic_r{r}.csv")
				csv_h = os.path.join(raw_dir, f"Hybrid_r{r}.csv")
				
				if os.path.exists(csv_c) and os.path.exists(csv_h):
					if (r+1)%5==0: print(f" [Skip {r+1}]", end="")
					continue

				seed = 1000 + (int(''.join(filter(str.isdigit, scen_name)) or 0)*100) + r
				print(f"\r   Run {r+1}/{TOTAL_RUNS}: ", end="")

				# Classic
				if not os.path.exists(csv_c):
					print("C", end="", flush=True)
					alg_c = NSGA2Classic(problem, 100, 100, 0.9, 0.1)
					alg_c.setSeed(seed)
					alg_c.run(verbose=False)
					analyzer.addResult('Classic', f"{scen_name}_r{r}", alg_c.population, save_path=csv_c)
				
				# Hybrid
				if not os.path.exists(csv_h):
					print(" | H", end="", flush=True)
					alg_h = NSGA2Hybrid(problem, 100, 100, 0.9, 0.1)
					alg_h.setSeed(seed)
					alg_h.run(verbose=False)
					analyzer.addResult('Hybrid', f"{scen_name}_r{r}", alg_h.population, save_path=csv_h)
			print(" Done.")

	# --- D. METRICS ---
	if ENABLE_NSGA:
		print("\n--- 📊 Computing Metrics ---")
		
		# LOAD ULANG DARI DRIVE (Agar data lama terhitung)
		raw_root = os.path.join(LOCAL_RESULTS_DIR, "raw_fronts")
		if os.path.exists(raw_root):
			analyzer.loadResultsFromDirectory(raw_root)
			
		final_stats = analyzer.computeMetrics()
		
		# Save CSV Summary
		flat = []
		for algo in ['Classic', 'Hybrid']:
			for idx, m in enumerate(final_stats[algo]):
				m.update({'Algorithm': algo, 'RunID': idx})
				flat.append(m)
		
		if flat:
			pd.DataFrame(flat).to_csv(os.path.join(LOCAL_RESULTS_DIR, 'final_metrics.csv'), index=False)
			print("✅ Metrics saved to Drive.")

		# Summary Print
		print("\n=== SUMMARY RESULT ===")
		for algo in ['Classic', 'Hybrid']:
			if final_stats[algo]:
				hv = np.mean([x['hv'] for x in final_stats[algo]])
				igd = np.mean([x['igd_plus'] for x in final_stats[algo]])
				print(f"[{algo}] Avg HV={hv:.4f}, Avg IGD+={igd:.4f}")

if __name__ == "__main__":
	run_pipeline()