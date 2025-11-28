from individual import Individual

class IndividualHybrid(Individual):
	def __init__(self, problem, server_map):
		super().__init__(problem)
		self.server_map = server_map
		self.vm_to_server_map: List[int] = [0] * self.problem.N_V
		self.syncRepresentations()

	def getChromosome(self):
		return self.server_map

	def syncRepresentations(self):
		self.chromosome_list = [0] * self.problem.N_V
		for server_idx, vm_list in self.server_map.items():
			for vm_idx in vm_list:
				self.chromosome_list[vm_idx] = server_idx
				self.vm_to_server_map[vm_idx] = server_idx

	def evaluateDelta(self, vm_idx, new_server_idx):
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
		
		self.calculateConstraint_Net()
		self.calculateObjective_Net()
		
		self.updateConstraintStatus()