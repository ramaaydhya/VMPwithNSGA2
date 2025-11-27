import random
import copy
import numpy as np
from abc import ABC, abstractmethod

from individual import Individual
from population import Population
from problem import Problem

class NSGA2(ABC):
	def __init__(self, problem, populationSize=100, maxGeneration=100, crossoverProbability=0.9, mutationProbability=0.1):
		self.problem = problem
		self.populationSize: int = populationSize
		self.maxGeneration: int = maxGeneration
		self.crossoverProbability: float = crossoverProbability
		self.mutationProbability: float = mutationProbability

		self.population: Population = None
	
	def setSeed(self, seed):
		random.seed(seed)
		np.random.seed(seed)

	def run(self):			
		# First ever generation
		self.generatePopulation()
		self.fastNonDominatedSort(self.population)
		for front in self.population.fronts:
			self.calculateCrowdingDistance(front)
		offspring = self.createOffspring(self.population)

		for _ in range(self.maxGeneration):
			self.population.extend(offspring)
			self.fastNonDominatedSort(self.population)

			new_individuals = []
			front_idx = 0

			# Add fronts from the highest rank before exceeding population size limit
			while len(new_individuals) + len(self.population.fronts[front_idx]) <= self.populationSize:
				self.calculateCrowdingDistance(self.population.fronts[front_idx])
				new_individuals.extend(self.population.fronts[front_idx])
				front_idx += 1
				if front_idx >= len(self.population.fronts):
					break

			# At this point, the added individuals don't fulfill the population size yet 
			# However, adding one more front will exceed population size limit
			# Therefore, we select some individuals with high crowding distance from this new front 		
			if len(new_individuals) < self.populationSize:
				last_front = self.population.fronts[front_idx]
				self.calculateCrowdingDistance(last_front)
				last_front.sort(key=lambda x: x.crowdingDistance, reverse=True)
				fill_count = self.populationSize - len(new_individuals)
				new_individuals.extend(last_front[:fill_count])

			self.population = Population(new_individuals)
			
			# One more sorting and crowding distance calculation for selecting parents 
			self.fastNonDominatedSort(self.population)
			for front in self.population.fronts:
				self.calculateCrowdingDistance(front)
			offspring = self.createOffspring(self.population)			

	def fastNonDominatedSort(self, population):
		population.fronts = [[]]
  		for individual in population:
			individual.dominationCount = 0
			individual.dominatedSolutions = []
			for other in population:
	  			if individual.dominates(other):
					individual.dominatedSolutions.append(other)
	  			elif other.dominates(individual):
					individual.dominationCount += 1
			if individual.dominationCount == 0:
	  			individual.frontRank = 0
	  			population.fronts[0].append(individual)
  		i = 0
  		while len(population.fronts[i]) > 0:
			temp = []
			for individual in population.fronts[i]:
	  			for other in individual.dominatedSolutions:
					other.dominationCount -= 1
					if other.dominationCount == 0:
		  				other.frontRank = i + 1
		  				temp.append(other)
			i += 1
			population.fronts.append(temp)

	def calculateCrowdingDistance(self, front):
		if len(front) > 0:
			individualCount = len(front)
			for individual in front:
	  			individual.crowdingDistance = 0

			for key in front[0].objectives.keys():
	  			front.sort(key=lambda x: x.objectives[key])
	  			front[0].crowdingDistance = float('inf')
	  			front[individualCount - 1].crowdingDistance = float('inf')
	  			objectiveValues = [individual.objectives[key] for individual in front]
	  			scale = max(objectiveValues) - min(objectiveValues)
	  			scale = scale if scale != 0 else 1
	  			for i in range(1, individualCount - 1):
					front[i].crowdingDistance += (front[i + 1].objectives[key] - front[i - 1].objectives[key]) / scale

	def createOffspring(self, population):
		offspringList = []

		while len(offspringList) < self.populationSize:
			parent1 = self.tournament(population)
			parent2 = parent1
			
			while parent1 == parent2:
				parent2 = self.tournament(population)
			
			if random.random() <= self.crossoverProbability:
				offspring1, offspring2 = self.crossover(parent1, parent2)
				offspring1.evaluateFull()
				offspring2.evaluateFull()
			else:
				offspring1 = copy.deepcopy(parent1)
				offspring2 = copy.deepcopy(parent2)

			if random.random() <= self.mutationProbability:
				self.mutate(offspring1)
			if random.random() <= self.mutationProbability:
				self.mutate(offspring2)

			offspringList.append(offspring1)
			if len(offspringList) < self.populationSize:
				offspringList.append(offspring2)

		return offspringList

	def tournament(self, population):
		selectedIndividuals = random.sample(population.individuals, 2)
		if selectedIndividuals[0].frontRank < selectedIndividuals[1].frontRank:
			return selectedIndividuals[0]
		elif selectedIndividuals[0].frontRank == selectedIndividuals[1].frontRank:
			if selectedIndividuals[0].crowdingDistance > selectedIndividuals[1].crowdingDistance:
				return selectedIndividuals[0]
			else:	
				return selectedIndividuals[1]
		else:
			return selectedIndividuals[1]		

	def generatePopulation(self):
		self.population = Population()

		for _ in range(self.populationSize):
			chromosome_list = self._generate_chromosome_random_first_fit()
			ind = self._create_individual_from_list(chromosome_list)
			self.population.append(ind)

	# Helper function for NSGA2.generatePopulation()		
	def _generate_chromosome_random_first_fit(self):
		chromosome = [-1] * self.problem.N_V

		remaining_cpu = np.copy(self.problem.p_cpu)
		remaining_mem = np.copy(self.problem.p_mem)

		vm_queue = list(range(self.problem.N_V))
		random.shuffle(vm_queue)
		
		for vm_idx in vm_queue:
			req_cpu = self.problem.v_cpu[vm_idx]
			req_mem = self.problem.v_mem[vm_idx]

			is_placed = False

			for server_idx in range(self.problem.N_P):
				if req_cpu <= remaining_cpu[server_idx] and req_mem <= remaining_mem[server_idx]:
					chromosome[vm_idx] = server_idx

					remaining_cpu[server_idx] -= req_cpu
					remaining_mem[server_idx] -= req_mem

					is_placed = True
					break

			# Fallback if nothing fits
			if not is_placed:
				chromosome[vm_idx] = random.randint(0, self.problem.N_P - 1)

		return chromosome
	
	def repair(self, individual):
		MAX_ATTEMPTS = 50 
		attempt = 0
		
		p_cpu = self.problem.p_cpu
		p_mem = self.problem.p_mem
		
		overloaded_servers = set()
		for server_idx in range(self.problem.N_P):
			if (individual.total_cpu_per_server[server_idx] > p_cpu[server_idx] or 
				individual.total_mem_per_server[server_idx] > p_mem[server_idx]):
				overloaded_servers.add(server_idx)
		
		while len(overloaded_servers) > 0 and attempt < MAX_ATTEMPTS:
			attempt += 1
			
			source_server_idx = random.choice(list(overloaded_servers))
			
			vms_on_server = individual.server_map[source_server_idx]
			
			# Edge case 
			if not vms_on_server:
				overloaded_servers.remove(source_server_idx)
				continue

			vm_to_move = random.choice(vms_on_server)
			req_cpu = self.problem.v_cpu[vm_to_move]
			req_mem = self.problem.v_mem[vm_to_move]
			
			found_target = False
			target_candidates = list(range(self.problem.N_P))
			random.shuffle(target_candidates)
			
			for target_server_idx in target_candidates:
				if target_server_idx == source_server_idx: continue
				
				avl_cpu = p_cpu[target_server_idx] - individual.total_cpu_per_server[target_server_idx]
				avl_mem = p_mem[target_server_idx] - individual.total_mem_per_server[target_server_idx]
				
				if avl_cpu >= req_cpu and avl_mem >= req_mem:
					
					individual.evaluateDelta(vm_to_move, target_server_idx)
					found_target = True
					
					new_usage_cpu = individual.total_cpu_per_server[source_server_idx]
					new_usage_mem = individual.total_mem_per_server[source_server_idx]
					
					if (new_usage_cpu <= p_cpu[source_server_idx] and 
						new_usage_mem <= p_mem[source_server_idx]):
						overloaded_servers.remove(source_server_idx)
					
					break
			
		individual.updateConstraintStatus()

	@abstractmethod
	def _create_individual_from_list(self, chromosome_list):
		pass

	@abstractmethod
	def crossover(self, parent1, parent2):
		pass

	@abstractmethod
	def mutate(self, offspring):
		pass
