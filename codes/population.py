from individual import Individual

class Population:
	def __init__(self, individuals = None):
		self.individuals = individuals if individuals is not None else []
		self.fronts = []

	def __len__(self):
		return len(self.individuals)

	def __iter__(self):
		return iter(self.individuals)

	def __getitem__(self, index):
		return self.individuals[index]	

	def extend(self, newIndividuals):
		self.individuals.extend(newIndividuals)

	def append(self, newIndividual):
		self.individuals.append(newIndividual)		