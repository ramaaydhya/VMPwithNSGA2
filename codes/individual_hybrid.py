from typing import Dict, List

from individual import Individual

class IndividualHybrid(Individual):
	def __init__(self, algorithm: NSGA2, problem: Problem, server_map: Dict[int, List[int]]):
		super().__init__(algorithm, problem)
		self.server_map = server_map
		self.vm_to_server_map: List[int] = [0] * self.problem.N_V
		self.syncRepresentations()

	def getChromosome(self) -> Dict[int, List[int]]:
		return self.server_map

	def syncRepresentations(self):
		self.chromosome_list = [0] * self.problem.N_V
		for server_idx, vm_list in self.server_map.items():
			for vm_idx in vm_list:
				self.chromosome_list[vm_idx] = server_idx
				self.vm_to_server_map[vm_idx] = server_idx

	def evaluateDelta(self, vm_idx: int, new_server_idx: int):
		old_server_idx = self.vm_to_server_map[vm_idx] 
		if old_server_idx == new_server_idx:
			return

		self.deltaUpdate_CPU_Mem_Power(vm_idx, old_server_idx, new_server_idx)
		
		self.server_map[old_server_idx].remove(vm_idx)
		if new_server_idx not in self.server_map:
			self.server_map[new_server_idx] = []
		self.server_map[new_server_idx].append(vm_idx)

		self.chromosome_list[vm_idx] = new_server_idx

		self.vm_to_server_map[vm_idx] = new_server_idx
		
		self.calculateConstraints_Net()
		self.calculateObjectives_Net()
		
		self.updateConstraintStatus()