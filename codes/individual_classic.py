from typing import List

from individual import Individual

class IndividualClassic(Individual):

	def __init__(self, algorithm: NSGA2, problem: Problem, chromosome_list: List[int]):
		super().__init__(algorithm, problem)
		self.chromosome_list = chromosome_list
		self.syncRepresentations()

	def getChromosome(self) -> List[int]:
		return self.chromosome_list

	def syncRepresentations(self):
		self.server_map = {}
		for vm_idx, server_idx in enumerate(self.chromosome_list):
			if server_idx not in self.server_map:
				self.server_map[server_idx] = []
			self.server_map[server_idx].append(vm_idx)

	def evaluateDelta(self, vm_idx: int, new_server_idx: int):
		old_server_idx = self.chromosome_list[vm_idx]
		if old_server_idx == new_server_idx:
			return

		self.deltaUpdate_CPU_Mem_Server(vm_idx, old_server_idx, new_server_idx)
        
		self.chromosome_list[vm_idx] = new_server_idx
		self.server_map[old_server_idx].remove(vm_idx)
		if new_server_idx not in self.server_map:
			self.server_map[new_server_idx] = []
		self.server_map[new_server_idx].append(vm_idx)
        
		self.calculateConstraint_Net()
		self.calculateObjective_Net()
        
		self.updateConstraintStatus()						