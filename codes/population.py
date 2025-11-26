from individual import Individual

class Population:
	def __init__(self, individuals: list[Individual] = None):
		self.individuals = individuals if individuals is not None else []
		self.fronts: list[list[Individual]] = []

	def __len__(self):
		return len(self.individuals)

	def __iter__(self):
		return iter(self.individuals)

	def __getitem__(self, index):
		return self.individuals[index]	

	def extend(self, newIndividuals: list[Individual]):
		self.individuals.extend(newIndividuals)

	def append(self, newIndividual: Individual):
		self.individuals.append(newIndividual)		