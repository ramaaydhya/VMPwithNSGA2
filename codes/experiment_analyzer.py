import json
import numpy as np
import copy

from performance_metrics import PerformanceMetrics # Asumsi file metrics.py sudah ada
from population import Population


class ExperimentAnalyzer:
	def __init__(self):
		# Format: self.results[algo][run_id] = np.array([[obj1, obj2], ...])
		# algo is either 'Classic' or 'Hybrid' 
		self.results = {
			'Classic': {},
			'Hybrid': {}
		}
		
		# Store solution from PaMILO
		self.pamilo_solutions = [] 
		
		# Reference Front = (Non-dominated solutions from NSGA + PaMILO)
		self.reference_front = None
		self.min_objectives = None
		self.max_objectives = None

	def addResult(self, algo_name, run_id, population):
		"""
		Menambahkan hasil akhir populasi dari NSGA-II.
		Hanya mengambil solusi Rank 0 (Non-dominated) dari run tersebut.
		"""
		front_data = []
		# Population is iterable  
		for individual in population:
			# Extract nondominated solutions from the population
			if individual.frontRank == 0:
				obj_1 = individual.objectives['power_consumption']
				obj_2 = individual.objectives['net_communication']
				front_data.append([obj_1, obj_2])
		
		# Save as np.array
		if front_data:
			self.results[algo_name][run_id] = np.array(front_data)
		else:
			self.results[algo_name][run_id] = np.empty((0, 2))

	def loadPamiloReference(self, filepath):
		"""
		Extract objective vectors from PaMILO output(JSON file)
		"""
		try:
			with open(filepath, 'r') as f:
				data = json.load(f)
			
			count = 0
			if "solutions" in data:
				for sol in data["solutions"]:
					vals = sol.get("values", [])
					if len(vals) >= 2:
						obj_pair = [vals[0], vals[1]] 
						self.pamilo_solutions.append(obj_pair)
						count += 1
			
			print(f"[Analyzer] Loaded {count} solutions from PaMILO ({filepath})")
			
		except FileNotFoundError:
			print(f"[Analyzer] WARNING: JSON file not found: {filepath}")
		except Exception as e:
			print(f"[Analyzer] Error loading file: {e}")

	def buildGlobalReferenceFront(self):
		"""
		Build reference front from:
		1. All best solutions from 30 runs of NSGA2Hybrid and NSGA2Classic.
		2. All PaMILO solutions.
		"""
		all_solutions = []
		
		# 1. Collect nondominated solutions from NSGA2Hybrid + NSGA2Classic
		for algo in self.results:
			for run_id in self.results[algo]:
				if len(self.results[algo][run_id]) > 0:
					all_solutions.extend(self.results[algo][run_id])
		
		# 2. Collect nondominated solutions from PaMILO
		if self.pamilo_solutions:
			all_solutions.extend(self.pamilo_solutions)
			
		all_solutions = np.array(all_solutions)
		
		if len(all_solutions) == 0:
			print("[Analyzer] Error: No solutions found to build reference front.")
			return

		# 3. Compute global min and max (from all_solutions) for normalization
		self.min_objectives = np.min(all_solutions, axis=0)
		self.max_objectives = np.max(all_solutions, axis=0)
		
		# 4. Filter nondominated solutions from all_solutions (Reference Pareto Front) 
		# Naive O(N^2) 
		is_dominated = np.zeros(len(all_solutions), dtype=bool)
		for i in range(len(all_solutions)):
			for j in range(len(all_solutions)):
				if i == j: continue
				# Jika solusi j mendominasi solusi i
				if all(all_solutions[j] <= all_solutions[i]) and any(all_solutions[j] < all_solutions[i]):
					is_dominated[i] = True
					break
		
		self.reference_front = all_solutions[~is_dominated]
		
		# Tidying (sort by first objective)
		self.reference_front = self.reference_front[np.argsort(self.reference_front[:, 0])]
		
		print(f"[Analyzer] Reference Front built. Size: {len(self.reference_front)}")
		print(f"		   (Includes PaMILO: {len(self.pamilo_solutions) > 0})")

	def normalize(self, front):
		"""Normalize into [0, 1] using global min/max."""
		if self.min_objectives is None:
			raise ValueError("Build reference front first!")
		
		range_vals = self.max_objectives - self.min_objectives
		range_vals[range_vals == 0] = 1.0 # No division by zero
		
		return (front - self.min_objectives) / range_vals

	def computeMetrics(self):
		"""
		Evaluate IGD+, GD+, HV, Spacing for each run NSGA-II.
		"""
		if self.reference_front is None:
			self.buildGlobalReferenceFront()
			
		# Normalisasi Reference Front
		norm_ref_front = self.normalize(self.reference_front)
		
		# Titik Referensi Hypervolume (sedikit di atas 1.0)
		hv_ref_point = np.array([1.1, 1.1])
		
		final_stats = {'Classic': [], 'Hybrid': []}
		
		for algo in ['Classic', 'Hybrid']:
			for run_id, raw_front in self.results[algo].items():
				if len(raw_front) == 0: continue
				
				# Normalize front from this run
				norm_front = self.normalize(raw_front)
				
				# Evaluate each metric
				metrics = {}
				metrics['spacing']  = PerformanceMetrics.calculate_spacing(norm_front)
				metrics['hv']	   = PerformanceMetrics.calculate_hypervolume(norm_front, hv_ref_point)
				metrics['igd_plus'] = PerformanceMetrics.calculate_igd_plus(norm_front, norm_ref_front)
				metrics['gd_plus']  = PerformanceMetrics.calculate_gd_plus(norm_front, norm_ref_front)
				
				final_stats[algo].append(metrics)
				
		return final_stats