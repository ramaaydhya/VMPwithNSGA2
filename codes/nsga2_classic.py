import random
import copy

from individual_classic import IndividualClassic
from nsga2 import NSGA2
from population import Population
from problem import Problem

class NSGA2Classic(NSGA2):
	def __init__(self, problem, populationSize=100, maxGeneration=100, crossoverProbability=0.9, mutationProbability=0.1):
		super().__init__(problem, populationSize, maxGeneration, crossoverProbability, mutationProbability)

	# Biased Uniform Crossover
	# each gene are picked randomly biased towards fitter parent
	def crossover(self, parent1, parent2):
		BIAS_RATIO = 0.7

		better_parent, worse_parent = parent1, parent2

		if parent1.frontRank > parent2.frontRank:
			better_parent, worse_parent = parent2, parent1
		elif parent1.frontRank == parent2.frontRank:
			if parent1.crowdingDistance < parent2.crowdingDistance:
				better_parent, worse_parent = parent2, parent1

		chromosome_1 = [-1] * self.problem.N_V
		chromosome_2 = [-1] * self.problem.N_V

		for vm_idx in range(self.problem.N_V):
			if random.random() <= BIAS_RATIO:
				chromosome_1[vm_idx] = better_parent.chromosome_list[vm_idx]
			else:
				chromosome_1[vm_idx] = worse_parent.chromosome_list[vm_idx] 

			if random.random() <= BIAS_RATIO:	
				chromosome_2[vm_idx] = better_parent.chromosome_list[vm_idx]
			else:	
				chromosome_2[vm_idx] = worse_parent.chromosome_list[vm_idx]

		offspring_1 = IndividualClassic(self, self.problem, chromosome_1)
		offspring_2 = IndividualClassic(self, self.problem, chromosome_2)

		offspring_1.evaluateFull()
		offspring_2.evaluateFull()

		return offspring_1, offspring_2

	# Random Mutation
	def mutate(self, individual):
		is_mutated = False

		prob_per_gene = 1.0 / self.problem.N_V

		current_server = None
		new_server = None
		
		for vm_idx in range(self.problem.N_V):
			if random.random() <= prob_per_gene:
				current_server = individual.chromosome_list[vm_idx]
				new_server = random.randint(0, self.problem.N_P - 1)

			while new_server == current_server:
				new_server = random.randint(0, self.problem.N_P - 1)

			individual.evaluateDelta(vm_idx, new_server)
			is_mutated = True

		if is_mutated and individual.isConstraintViolated:
			self.repair(individual)			

	def _create_individual_from_list(self, chromosome_list):
		return IndividualClassic(self, self.problem, chromosome_list)