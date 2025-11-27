import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Union

from problem import Problem

class Individual(ABC):
	def __init__(self, algorithm, problem: Problem):
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

		# cache
		self.total_cpu_per_server = np.zeros(problem.N_P)
		self.total_mem_per_server = np.zeros(problem.N_P)
		self.total_net_per_server = np.zeros(problem.N_P)
		self.v_net_per_vm = np.zeros(problem.N_V) 
    	
	def dominates(self, other: Individual) -> bool:
		if not self.isConstraintViolated and other.isConstraintViolated:
			return True
		elif self.isConstraintViolated and not other.isConstraintViolated:
			return False
		elif self.isConstraintViolated and other.isConstraintViolated:
			return self.totalViolation < other.totalViolation
		else:
			isEqual = True
			for key in self.objectives.keys():
				if self.objectives[key] > other.objectives[key]:
					return False
				elif self.objectives[key] < other.objectives[key]:
					isEqual = False
			return not isEqual

	def evaluateFull(self):
		self.calculateConstraint_CPU_Mem()
		self.calculateConstraint_Net()

		self.calculateObjective_Power()
		self.calculateObjective_Net()

		self.updateConstraintStatus()

	def updateConstraintStatus(self):
		self.totalViolation = 0
		for _, violation in self.constraintViolations.items():
			self.totalViolation += np.sum(violation)
		
		self.isConstraintViolated = self.totalViolation > 0

	def calculateConstraint_CPU_Mem(self):
		self.total_cpu_per_server.fill(0)
		self.total_mem_per_server.fill(0)

		for server_idx, vm_list in self.server_map.items():
			if not vm_list:
				continue
			self.total_cpu_per_server[server_idx] = sum(self.problem.v_cpu[vm] for vm in vm_list)
			self.total_mem_per_server[server_idx] = sum(self.problem.v_mem[vm] for vm in vm_list)

		self.constraintViolations["cpu"] = np.maximum(0, self.total_cpu_per_server - self.problem.p_cpu)	
		self.constraintViolations["mem"] = np.maximum(0, self.total_mem_per_server - self.problem.p_mem)

	def calculateConstraint_Net(self):
		self.v_net_per_vm.fill(0)
		self.total_net_per_server.fill(0)

		for vm_1 in range(self.problem.N_V):
			server_1 = self.chromosome_list[vm_1]
			inter_server_traffic = 0
			for vm_2 in range(self.problem.N_V):
				if vm_1 == vm_2:
					continue
				server_2 = self.chromosome_list[vm_2]
				is_different_server = 1 if server_1 != server_2 else 0
				inter_server_traffic += is_different_server * self.problem.T_matrix[vm_1, vm_2]
			self.v_net_per_vm[vm_1] = self.problem.e_vector[vm_1] + inter_server_traffic

		for server_idx, vm_list in self.server_map.items():
			if not vm_list: continue
			self.total_net_per_server[server_idx] = sum(self.v_net_per_vm[vm] for vm in vm_list)

		self.constraintViolations["net"] = np.maximum(0, self.total_net_per_server - self.problem.p_net)	

	def calculateObjective_Power(self):
		pc_sum = 0
		for server_idx in range(self.problem.N_P):
			if self.total_cpu_per_server[server_idx] > 0:
				pc_sum += self._get_power_for_server(server_idx)
		self.objectives["power_consumption"] = pc_sum

	def calculateObjective_Net(self):
		t_sum_1 = 0
		for vm_1 in range(self.problem.N_V):
			server_1 = self.chromosome_list[vm_1]
			for vm_2 in range(self.problem.N_V):
				server_2 = self.chromosome_list[vm_2]
				t_sum_1 += self.problem.T_matrix[vm_1, vm_2] * self.problem.C_matrix[server_1, server_2]
		t_sum_1 *= 0.5 

		t_sum_2 = 0
		for vm_idx in range(self.problem.N_V):
			server_idx = self.chromosome_list[vm_idx]
			t_sum_2 += self.problem.e_vector[vm_idx] * self.problem.g_vector[server_idx]

		self.objectives["net_communication"] = t_sum_1 + t_sum_2

	def _get_power_for_server(self, server_idx: int) -> float:
		cpu_usage = self.total_cpu_per_server[server_idx]
		if cpu_usage <= 0:
			return 0.0

		p_cpu_j = self.problem.p_cpu[server_idx]
		PC_max_j = self.problem.PC_max[server_idx]
		PC_idle_j = self.problem.PC_idle[server_idx]

		epsilon = 1e-6
		U_j_cpu = cpu_usage / (p_cpu_j + epsilon)
		return (PC_max_j - PC_idle_j) * U_j_cpu + PC_idle_j

	def deltaUpdate_CPU_Mem_Power(self, vm_idx: int, src_server_idx: int, dst_server_idx: int):
		v_cpu_i = self.problem.v_cpu[vm_idx]
		v_mem_i = self.problem.v_mem[vm_idx]

		pc_src_before = self._get_power_for_server(src_server_idx)
		pc_dst_before = self._get_power_for_server(dst_server_idx)

		self.total_cpu_per_server[src_server_idx] -= v_cpu_i
		self.total_cpu_per_server[dst_server_idx] += v_cpu_i
		self.total_mem_per_server[src_server_idx] -= v_mem_i
		self.total_mem_per_server[dst_server_idx] += v_mem_i

		self.constraintViolations["cpu"][src_server_idx] = np.maximum(0, self.total_cpu_per_server[src_server_idx] - self.problem.p_cpu[src_server_idx])
		self.constraintViolations["cpu"][dst_server_idx] = np.maximum(0, self.total_cpu_per_server[dst_server_idx] - self.problem.p_cpu[dst_server_idx])
		self.constraintViolations["mem"][src_server_idx] = np.maximum(0, self.total_mem_per_server[src_server_idx] - self.problem.p_mem[src_server_idx])
		self.constraintViolations["mem"][dst_server_idx] = np.maximum(0, self.total_mem_per_server[dst_server_idx] - self.problem.p_mem[dst_server_idx])
        
		pc_src_after = self._get_power_for_server(src_server_idx)
		pc_dst_after = self._get_power_for_server(dst_server_idx)
        
		power_delta = (pc_src_after + pc_dst_after) - (pc_src_before + pc_dst_before)
		self.objectives["power_consumption"] += power_delta

	@abstractmethod
	def getChromosome(self) -> Union[List[int], Dict[int, List[int]]]:
		pass

	@abstractmethod
	def syncRepresentations(self):
		pass

	@abstractmethod
	def evaluateDelta(self, vm_idx: int, new_server_idx: int):
		pass		