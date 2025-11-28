import json
import numpy as np
import pandas as pd
import os
import glob

from performance_metrics import PerformanceMetrics 
from population import Population

class ExperimentAnalyzer:
	def __init__(self):
		# Format: self.results[algo][run_id] = np.array([[obj1, obj2], ...])
		self.results = {
			'Classic': {},
			'Hybrid': {}
		}
		self.pamilo_solutions = [] 
		self.reference_front = None
		self.min_objectives = None
		self.max_objectives = None

	def addResult(self, algo_name, run_id, population: Population, save_path=None):
		"""
		Menambahkan hasil, ekstrak Pareto Front, dan (Opsional) SIMPAN ke file.
		"""
		front_data = []
		
		# Handle jika population adalah list atau object
		inds = population.individuals if hasattr(population, 'individuals') else population
		
		for individual in inds:
			if individual.frontRank == 0:
				obj_1 = individual.objectives['power_consumption']
				obj_2 = individual.objectives['net_communication']
				front_data.append([obj_1, obj_2])
		
		# Konversi ke Numpy Array
		front_arr = np.array(front_data) if front_data else np.empty((0, 2))
		
		# 1. Simpan ke Memory
		self.results[algo_name][run_id] = front_arr

		# 2. Simpan ke File (Jika path diberikan)
		if save_path:
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			df = pd.DataFrame(front_arr, columns=['Power', 'Net'])
			df.to_csv(save_path, index=False)

	def loadResultsFromDirectory(self, base_dir):
		"""
		[FITUR BARU] Memuat kembali hasil eksperimen dari file CSV di folder.
		Berguna untuk resume atau analisis ulang tanpa re-run.
		Struktur folder yang diharapkan: base_dir/{Scenario}/{Algorithm}_r{RunID}.csv
		"""
		print(f"[Analyzer] Reloading results from {base_dir}...")
		
		# Cari semua file CSV secara rekursif
		csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
		
		count = 0
		for fpath in csv_files:
			fname = os.path.basename(fpath)
			
			# Parsing nama file: Classic_r0.csv -> algo=Classic, run_id=..._r0
			# Asumsi format nama file dari main.py: "{Algorithm}_r{run_id}.csv"
			# Tapi kita butuh run_id unik string (misal "small_1_r0")
			
			# Deteksi Algo
			algo = None
			if "Classic" in fname: algo = 'Classic'
			elif "Hybrid" in fname: algo = 'Hybrid'
			
			if algo:
				try:
					df = pd.read_csv(fpath)
					if not df.empty:
						# Buat run_id unik berdasarkan nama folder (skenario) + nama file
						# Contoh: small_1/Classic_r0.csv -> run_id="small_1_r0"
						scenario = os.path.basename(os.path.dirname(fpath))
						run_suffix = fname.split('_')[-1].replace('.csv', '') # r0
						unique_run_id = f"{scenario}_{run_suffix}"
						
						self.results[algo][unique_run_id] = df.values
						count += 1
				except Exception as e:
					print(f"   [Warn] Failed to load {fname}: {e}")
					
		print(f"[Analyzer] Reloaded {count} runs from disk.")

	def loadPamiloReference(self, filepath):
		try:
			with open(filepath, 'r') as f:
				data = json.load(f)
			
			if "solutions" in data:
				for sol in data["solutions"]:
					vals = sol.get("values", [])
					if len(vals) >= 2:
						obj_pair = [vals[0], vals[1]] 
						self.pamilo_solutions.append(obj_pair)
			
		except Exception as e:
			print(f"[Analyzer] Error loading PaMILO: {e}")

	def buildGlobalReferenceFront(self):
		all_solutions = []
		
		# Gabung dari Memory (self.results)
		for algo in self.results:
			for run_id in self.results[algo]:
				if len(self.results[algo][run_id]) > 0:
					all_solutions.extend(self.results[algo][run_id])
					
		# Gabung PaMILO
		if self.pamilo_solutions:
			all_solutions.extend(self.pamilo_solutions)
			
		all_solutions = np.array(all_solutions)
		if len(all_solutions) == 0: return

		self.min_objectives = np.min(all_solutions, axis=0)
		self.max_objectives = np.max(all_solutions, axis=0)
		
		# Filter Non-Dominated
		is_dominated = np.zeros(len(all_solutions), dtype=bool)
		for i in range(len(all_solutions)):
			for j in range(len(all_solutions)):
				if i == j: continue
				if all(all_solutions[j] <= all_solutions[i]) and any(all_solutions[j] < all_solutions[i]):
					is_dominated[i] = True
					break
		
		self.reference_front = all_solutions[~is_dominated]
		self.reference_front = self.reference_front[np.argsort(self.reference_front[:, 0])]
		print(f"[Analyzer] Ref Front Built: {len(self.reference_front)} points.")

	def normalize(self, front):
		if self.min_objectives is None:
			self.buildGlobalReferenceFront()
		
		range_vals = self.max_objectives - self.min_objectives
		range_vals[range_vals == 0] = 1.0
		return (front - self.min_objectives) / range_vals

	def computeMetrics(self):
		if self.reference_front is None:
			self.buildGlobalReferenceFront()
			
		norm_ref_front = self.normalize(self.reference_front)
		hv_ref_point = np.array([1.1, 1.1])
		
		final_stats = {'Classic': [], 'Hybrid': []}
		
		for algo in ['Classic', 'Hybrid']:
			for run_id, raw_front in self.results[algo].items():
				if len(raw_front) == 0: continue
				
				norm_front = self.normalize(raw_front)
				
				metrics = {}
				metrics['run'] = run_id # Simpan ID agar bisa dilacak
				metrics['spacing']  = PerformanceMetrics.calculate_spacing(norm_front)
				metrics['hv']	   = PerformanceMetrics.calculate_hypervolume(norm_front, hv_ref_point)
				metrics['igd_plus'] = PerformanceMetrics.calculate_igd_plus(norm_front, norm_ref_front)
				metrics['gd_plus']  = PerformanceMetrics.calculate_gd_plus(norm_front, norm_ref_front)
				
				final_stats[algo].append(metrics)
				
		return final_stats