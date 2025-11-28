from individual import Individual

class IndividualClassic(Individual):

	def __init__(self, problem, chromosome_list):
		super().__init__(problem)
		self.chromosome_list = chromosome_list
		self.syncRepresentations()

	def getChromosome(self):
		return self.chromosome_list

	def syncRepresentations(self):
		self.server_map = {}
		for vm_idx, server_idx in enumerate(self.chromosome_list):
			if server_idx not in self.server_map:
				self.server_map[server_idx] = []
			self.server_map[server_idx].append(vm_idx)

	def evaluateDelta(self, vm_idx, new_server_idx):
		old_server_idx = self.chromosome_list[vm_idx]
		if old_server_idx == new_server_idx:
			return

		self.deltaUpdate_CPU_Mem_Power(vm_idx, old_server_idx, new_server_idx)

		self.chromosome_list[vm_idx] = new_server_idx
		self.server_map[old_server_idx].remove(vm_idx)
		if new_server_idx not in self.server_map:
			self.server_map[new_server_idx] = []
		self.server_map[new_server_idx].append(vm_idx)

		self.calculateConstraint_Net()
		self.calculateObjective_Net()

		self.updateConstraintStatus()						