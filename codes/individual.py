import numpy as np
from abc import ABC, abstractmethod

class Individual(ABC):
	def __init__(self, algorithm, problem):
		self.algorithm = algorithm
		self.problem = problem

		# NSGA attributes
		self.frontRank: int = -1
		self.crowdingDistance: float = float('nan')
		self.dominationCount: int = -1
		self.dominatedSolutions: List[Individual] = []

		# Optimization attributes
		self.objectives = {
			'power_consumption': 0,
			'net_communication': 0,
		}
		self.constraintViolations = {
			'cpu': np.zeros(problem.N_P),
			'mem': np.zeros(problem.N_P),
			'net': np.zeros(problem.N_P),
		}
		self.totalViolation: float = 0.0
		self.isConstraintViolated: bool = False

		# Solution representation
		self.chromosome_list: List[int] = []
		self.server_map: Dict[int, List[int]] = {}

		# Cache attributes
		self.total_cpu_per_server = np.zeros(problem.N_P)
		self.total_mem_per_server = np.zeros(problem.N_P)
		self.total_net_per_server = np.zeros(problem.N_P)
		self.v_net_per_vm = np.zeros(problem.N_V) 
		
	def dominates(self, other):
		# Logika Constrained Dominance (Deb et al.)
		if not self.isConstraintViolated and other.isConstraintViolated:
			return True
		elif self.isConstraintViolated and not other.isConstraintViolated:
			return False
		elif self.isConstraintViolated and other.isConstraintViolated:
			# Jika keduanya melanggar, pilih yang pelanggarannya lebih kecil
			return self.totalViolation < other.totalViolation
		else:
			# Jika keduanya feasible, cek Pareto Dominance pada objektif
			isEqual = True
			for key in self.objectives.keys():
				if self.objectives[key] > other.objectives[key]: # Minimasi
					return False
				elif self.objectives[key] < other.objectives[key]:
					isEqual = False
			return not isEqual

	def evaluateFull(self):
		"""Menghitung ulang semua constraint dan objektif dari nol."""
		self.calculateConstraint_CPU_Mem()
		self.calculateConstraint_Net()

		self.calculateObjective_Power()
		self.calculateObjective_Net()

		self.updateConstraintStatus()

	def updateConstraintStatus(self):
		"""Mengupdate status boolean isConstraintViolated."""
		self.totalViolation = 0
		for _, violation in self.constraintViolations.items():
			self.totalViolation += np.sum(violation)
		
		self.isConstraintViolated = self.totalViolation > 0

	def calculateConstraint_CPU_Mem(self):
		"""
		Menghitung penggunaan CPU dan Memori per server.
		Optimasi: Loop server map + Numpy Sum slicing.
		"""
		self.total_cpu_per_server.fill(0)
		self.total_mem_per_server.fill(0)

		for server_idx, vm_list in self.server_map.items():
			if not vm_list: continue
			
			# Menggunakan numpy indexing untuk penjumlahan cepat
			self.total_cpu_per_server[server_idx] = np.sum(self.problem.v_cpu[vm_list])
			self.total_mem_per_server[server_idx] = np.sum(self.problem.v_mem[vm_list])

		self.constraintViolations["cpu"] = np.maximum(0, self.total_cpu_per_server - self.problem.p_cpu)	
		self.constraintViolations["mem"] = np.maximum(0, self.total_mem_per_server - self.problem.p_mem)

	def calculateConstraint_Net(self):
		"""
		Menghitung beban jaringan per server (Ingress + Egress).
		Optimasi: Vektorisasi NumPy (No Loop for VM).
		"""
		chrom_arr = np.array(self.chromosome_list) # Size: (N_V,)
		
		# 1. Buat Mask Matriks: True jika VM i dan j berada di server yang BERBEDA
		# Broadcasting: (N_V, 1) != (1, N_V) -> Matriks (N_V, N_V)
		is_diff_server = chrom_arr[:, np.newaxis] != chrom_arr[np.newaxis, :]
		
		# 2. Hitung total trafik antar-server keluar dari setiap VM
		# Kalikan Traffic Matrix dengan Mask (hanya ambil trafik yang lintas server)
		inter_server_traffic = np.sum(self.problem.T_matrix * is_diff_server, axis=1)
		
		# 3. Total Traffic per VM = Trafik External + Trafik Inter-Server
		self.v_net_per_vm = self.problem.e_vector + inter_server_traffic
		
		# 4. Agregasi ke Server (Group By Server ID)
		self.total_net_per_server.fill(0)
		# np.add.at melakukan penjumlahan in-place pada indeks yang ditentukan
		np.add.at(self.total_net_per_server, chrom_arr, self.v_net_per_vm)

		self.constraintViolations["net"] = np.maximum(0, self.total_net_per_server - self.problem.p_net)	

	def calculateObjective_Power(self):
		"""
		Menghitung total konsumsi daya.
		"""
		pc_sum = 0
		for server_idx in range(self.problem.N_P):
			if self.total_cpu_per_server[server_idx] > 0:
				pc_sum += self._get_power_for_server(server_idx)
		self.objectives["power_consumption"] = pc_sum

	def calculateObjective_Net(self):
		"""
		Menghitung Total Network Communication Cost.
		Optimasi: Vektorisasi NumPy (Matriks Operation).
		"""
		chrom_arr = np.array(self.chromosome_list)
		
		# 1. Construct Matriks Cost antar VM berdasarkan Server tempat mereka berada
		# Advanced Indexing: C_mapped[i, j] = C_matrix[server_i, server_j]
		C_mapped = self.problem.C_matrix[chrom_arr[:, np.newaxis], chrom_arr[np.newaxis, :]]
		
		# 2. Hitung Term 1: Inter-VM Traffic Cost
		# Element-wise multiplication lalu sum semua
		weighted_traffic = np.sum(self.problem.T_matrix * C_mapped)
		t_sum_1 = weighted_traffic * 0.5 # Karena matriks simetris
		
		# 3. Hitung Term 2: External Traffic Cost
		# Ambil cost gateway untuk server masing-masing VM
		gateway_costs = self.problem.g_vector[chrom_arr]
		t_sum_2 = np.sum(self.problem.e_vector * gateway_costs)

		self.objectives["net_communication"] = t_sum_1 + t_sum_2

	def _get_power_for_server(self, server_idx):
		"""Helper menghitung power satu server."""
		cpu_usage = self.total_cpu_per_server[server_idx]
		if cpu_usage <= 0: return 0.0

		p_cpu_j = self.problem.p_cpu[server_idx]
		PC_max_j = self.problem.PC_max[server_idx]
		PC_idle_j = self.problem.PC_idle[server_idx]

		if p_cpu_j == 0: return 0.0 # Safety

		U_j_cpu = cpu_usage / p_cpu_j
		return (PC_max_j - PC_idle_j) * U_j_cpu + PC_idle_j

	def deltaUpdate_CPU_Mem_Power(self, vm_idx, src_server_idx, dst_server_idx):
		"""
		Update inkremental untuk CPU, Memori, dan Power.
		Dipanggil oleh Mutasi dan Repair.
		"""
		v_cpu_i = self.problem.v_cpu[vm_idx]
		v_mem_i = self.problem.v_mem[vm_idx]

		# Simpan power lama
		pc_src_before = self._get_power_for_server(src_server_idx)
		pc_dst_before = self._get_power_for_server(dst_server_idx)

		# Pindahkan Resource
		self.total_cpu_per_server[src_server_idx] -= v_cpu_i
		self.total_cpu_per_server[dst_server_idx] += v_cpu_i
		self.total_mem_per_server[src_server_idx] -= v_mem_i
		self.total_mem_per_server[dst_server_idx] += v_mem_i

		# Update Constraint Violation (Hanya untuk 2 server terkait)
		self.constraintViolations["cpu"][src_server_idx] = np.maximum(0, self.total_cpu_per_server[src_server_idx] - self.problem.p_cpu[src_server_idx])
		self.constraintViolations["cpu"][dst_server_idx] = np.maximum(0, self.total_cpu_per_server[dst_server_idx] - self.problem.p_cpu[dst_server_idx])
		self.constraintViolations["mem"][src_server_idx] = np.maximum(0, self.total_mem_per_server[src_server_idx] - self.problem.p_mem[src_server_idx])
		self.constraintViolations["mem"][dst_server_idx] = np.maximum(0, self.total_mem_per_server[dst_server_idx] - self.problem.p_mem[dst_server_idx])
		
		# Hitung Power Baru
		pc_src_after = self._get_power_for_server(src_server_idx)
		pc_dst_after = self._get_power_for_server(dst_server_idx)
		
		# Update Objektif Power dengan Delta
		power_delta = (pc_src_after + pc_dst_after) - (pc_src_before + pc_dst_before)
		self.objectives["power_consumption"] += power_delta

	@abstractmethod
	def getChromosome(self):
		pass

	@abstractmethod
	def syncRepresentations(self):
		pass

	@abstractmethod
	def evaluateDelta(self, vm_idx, new_server_idx):
		pass