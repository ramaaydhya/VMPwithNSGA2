import json
import numpy as np
import pandas as pd
import os
import glob
import re

from performance_metrics import PerformanceMetrics 
from population import Population

class ExperimentAnalyzer:
	def __init__(self):
		# Format: self.results[algo][run_key] = np.array([[obj1, obj2], ...])
		self.results = {
			'Classic': {},
			'Hybrid': {}
		}
		self.pamilo_solutions = [] 
		self.reference_front = None
		self.min_objectives = None
		self.max_objectives = None

	def addResult(self, algo_name, run_key, population: Population, save_path=None):
		"""
		Menambahkan hasil dari populasi NSGA-II ke memori dan (opsional) menyimpannya ke CSV di Drive.
		"""
		front_data = []
		
		# Handle jika population adalah object wrapper atau list
		inds = population.individuals if hasattr(population, 'individuals') else population
		
		for individual in inds:
			# Ambil hanya solusi Rank 0 (Non-Dominated)
			if individual.frontRank == 0:
				obj_1 = individual.objectives['power_consumption']
				obj_2 = individual.objectives['net_communication']
				front_data.append([obj_1, obj_2])
		
		# Konversi ke Numpy Array
		front_arr = np.array(front_data) if front_data else np.empty((0, 2))
		
		# 1. Simpan ke Memory (RAM)
		self.results[algo_name][run_key] = front_arr

		# 2. Simpan ke File (Drive) - Agar aman jika crash
		if save_path:
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			df = pd.DataFrame(front_arr, columns=['Power', 'Net'])
			df.to_csv(save_path, index=False)
			# print(f"   [Saved] {os.path.basename(save_path)}")

	def loadResultsFromDirectory(self, base_dir):
		"""
		Memuat kembali hasil CSV dari Drive ke Memory.
		Digunakan untuk melanjutkan eksperimen yang terputus.
		"""
		if not os.path.exists(base_dir):
			return

		print(f"[Analyzer] Checking for existing results in {base_dir}...")
		# Cari semua file CSV secara rekursif
		csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
		
		loaded_count = 0
		for fpath in csv_files:
			try:
				fname = os.path.basename(fpath)
				# Format nama file diharapkan: "{Algorithm}_r{run_id}.csv" (misal: Classic_r0.csv)
				# Folder parent diharapkan nama skenario (misal: small_1)
				
				scenario_name = os.path.basename(os.path.dirname(fpath))
				
				algo = None
				if "Classic" in fname: algo = 'Classic'
				elif "Hybrid" in fname: algo = 'Hybrid'
				
				if algo:
					# Extract run_id dari nama file
					# Classic_r0.csv -> r0
					match = re.search(r'_r(\d+)', fname)
					if match:
						run_suffix = f"r{match.group(1)}"
						run_key = f"{scenario_name}_{run_suffix}" # Key unik: small_1_r0
						
						df = pd.read_csv(fpath)
						if not df.empty:
							self.results[algo][run_key] = df[['Power', 'Net']].values
							loaded_count += 1
			except Exception as e:
				print(f"   [Warn] Failed to load {fname}: {e}")
		
		if loaded_count > 0:
			print(f"[Analyzer] Successfully reloaded {loaded_count} past runs from Drive.")

	def loadPamiloReference(self, filepath):
		try:
			with open(filepath, 'r') as f:
				data = json.load(f)
			if "solutions" in data:
				for sol in data["solutions"]:
					vals = sol.get("values", [])
					if len(vals) >= 2:
						self.pamilo_solutions.append([vals[0], vals[1]])
			print(f"[Analyzer] Loaded PaMILO reference: {filepath}")
		except Exception as e:
			print(f"[Analyzer] Error loading PaMILO: {e}")

	def buildGlobalReferenceFront(self):
		all_solutions = []
		
		# Gabung dari Memory (NSGA)
		for algo in self.results:
			for run_key in self.results[algo]:
				if len(self.results[algo][run_key]) > 0:
					all_solutions.extend(self.results[algo][run_key])
		
		# Gabung PaMILO
		if self.pamilo_solutions:
			all_solutions.extend(self.pamilo_solutions)
			
		all_solutions = np.array(all_solutions)
		if len(all_solutions) == 0: return

		self.min_objectives = np.min(all_solutions, axis=0)
		self.max_objectives = np.max(all_solutions, axis=0)
		
		# Filter Non-Dominated (Pareto)
		is_dominated = np.zeros(len(all_solutions), dtype=bool)
		for i in range(len(all_solutions)):
			for j in range(len(all_solutions)):
				if i == j: continue
				if all(all_solutions[j] <= all_solutions[i]) and any(all_solutions[j] < all_solutions[i]):
					is_dominated[i] = True
					break
		
		self.reference_front = all_solutions[~is_dominated]
		# Sort agar rapi
		self.reference_front = self.reference_front[np.argsort(self.reference_front[:, 0])]

	def normalize(self, front):
		if self.min_objectives is None: self.buildGlobalReferenceFront()
		range_vals = self.max_objectives - self.min_objectives
		range_vals[range_vals == 0] = 1.0
		return (front - self.min_objectives) / range_vals

	def computeMetrics(self):
		if self.reference_front is None: self.buildGlobalReferenceFront()
		
		norm_ref = self.normalize(self.reference_front)
		hv_ref_point = np.array([1.1, 1.1])
		
		final_stats = {'Classic': [], 'Hybrid': []}
		
		for algo in ['Classic', 'Hybrid']:
			# Sort keys agar urutan run rapi (0, 1, 2...)
			sorted_keys = sorted(self.results[algo].keys(), key=lambda x: int(x.split('_r')[-1]))
			
			for run_key in sorted_keys:
				raw_front = self.results[algo][run_key]
				if len(raw_front) == 0: continue
				
				norm_front = self.normalize(raw_front)
				
				metrics = {
					'run_key': run_key,
					'spacing': PerformanceMetrics.calculate_spacing(norm_front),
					'hv': PerformanceMetrics.calculate_hypervolume(norm_front, hv_ref_point),
					'igd_plus': PerformanceMetrics.calculate_igd_plus(norm_front, norm_ref),
					'gd_plus': PerformanceMetrics.calculate_gd_plus(norm_front, norm_ref)
				}
				final_stats[algo].append(metrics)
				
		return final_stats