import random
import copy
import numpy as np
from abc import ABC, abstractmethod
import sys

# Asumsi import kelas lain
# from individual import Individual
from population import Population
# from problem import Problem

class NSGA2(ABC):
	def __init__(self, problem, populationSize=100, maxGeneration=100, 
				 crossoverProbability=0.9, mutationProbability=0.1):
		self.problem = problem
		self.populationSize = populationSize
		self.maxGeneration = maxGeneration
		self.crossoverProbability = crossoverProbability
		self.mutationProbability = mutationProbability
		
		self.population = None

	def setSeed(self, seed):
		"""Mengatur seed random untuk reproduktibilitas."""
		random.seed(seed)
		np.random.seed(seed)

	def log(self, message, verbose):
		"""Helper untuk print jika verbose aktif."""
		if verbose:
			print(message, flush=True)

	def run(self, verbose=True):			
		self.log(f"\n[NSGA-II] Initializing Population ({self.populationSize} individuals)...", verbose)
		
		# Inisialisasi Populasi Awal
		self.generatePopulation()
		
		self.log("[NSGA-II] Initial Rank & Crowding Distance Calculation...", verbose)
		self.fastNonDominatedSort(self.population)
		for front in self.population.fronts:
			self.calculateCrowdingDistance(front)
			
		self.log("[NSGA-II] Creating First Generation Offspring...", verbose)
		offspring = self.createOffspring(self.population, verbose)

		for gen in range(self.maxGeneration):
			if verbose:
				print(f"\n=== GENERATION {gen+1}/{self.maxGeneration} ===", flush=True)

			# 1. Combine Parent + Offspring
			# self.population adalah object Population, offspring adalah list
			current_size = len(self.population.individuals)
			self.log(f"  > Merging Population (Size: {current_size} + {len(offspring)})", verbose)
			self.population.extend(offspring)
			
			# 2. Environmental Selection (Sorting)
			self.log("  > Environmental Selection: Fast Non-Dominated Sort", verbose)
			self.fastNonDominatedSort(self.population)
			
			# 3. Truncation (Memilih N terbaik)
			self.log("  > Environmental Selection: Truncating to Population Size", verbose)
			new_pop_list = []
			front_idx = 0
			
			# Ambil front demi front sampai batas
			while len(new_pop_list) + len(self.population.fronts[front_idx]) <= self.populationSize:
				self.calculateCrowdingDistance(self.population.fronts[front_idx]) 
				new_pop_list.extend(self.population.fronts[front_idx])
				front_idx += 1
				if front_idx >= len(self.population.fronts): break

			# Potong front terakhir jika perlu
			if len(new_pop_list) < self.populationSize:
				last_front = self.population.fronts[front_idx]
				self.calculateCrowdingDistance(last_front)
				# Sort descending by Crowding Distance
				last_front.sort(key=lambda x: x.crowdingDistance, reverse=True)
				
				fill_count = self.populationSize - len(new_pop_list)
				new_pop_list.extend(last_front[:fill_count])
				self.log(f"	- Filled remaining {fill_count} slots from Front {front_idx}", verbose)

			# Update Populasi (Bungkus list ke Object Population)
			self.population = Population(new_pop_list)
			
			# 4. Mating Preparation
			self.log("  > Mating Prep: Sorting & CD Calculation for Parents", verbose)
			self.fastNonDominatedSort(self.population)
			for front in self.population.fronts:
				self.calculateCrowdingDistance(front)
			
			# 5. Reproduction
			self.log("  > Reproduction: Tournament -> Crossover -> Mutation", verbose)
			offspring = self.createOffspring(self.population, verbose)

		self.log("\n[NSGA-II] Optimization Finished.", verbose)

	def fastNonDominatedSort(self, population):
		# Reset fronts
		population.fronts = [[]]
		
		# Iterasi via properti individuals jika population adalah object wrapper
		ind_list = population.individuals if hasattr(population, 'individuals') else population
		
		for individual in ind_list:
			individual.dominationCount = 0
			individual.dominatedSolutions = []
			for other in ind_list:
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

	def calculateCrowdingDistance(self, front: list):
		if len(front) > 0:
			individualCount = len(front)
			for individual in front:
				individual.crowdingDistance = 0

			for key in front[0].objectives.keys():
				front.sort(key=lambda x: x.objectives[key])
				front[0].crowdingDistance = float('inf')
				front[individualCount - 1].crowdingDistance = float('inf')
				
				objectiveValues = [ind.objectives[key] for ind in front]
				scale = max(objectiveValues) - min(objectiveValues)
				if scale == 0: scale = 1.0
					
				for i in range(1, individualCount - 1):
					front[i].crowdingDistance += (objectiveValues[i+1] - objectiveValues[i-1]) / scale

	def createOffspring(self, population, verbose=False) -> list:
		offspringList = []
		
		# Statistik untuk log
		stats = {'crossover': 0, 'mutation': 0, 'clones': 0}

		while len(offspringList) < self.populationSize:
			# 1. Selection
			parent1 = self.tournament(population)
			parent2 = parent1
			while parent1 == parent2:
				parent2 = self.tournament(population)
			
			# 2. Crossover
			if random.random() <= self.crossoverProbability:
				offspring1, offspring2 = self.crossover(parent1, parent2)
				# WAJIB: Hitung nilai objektif & constraint untuk anak baru
				# Karena proses crossover merusak struktur kromosom, nilai lama tidak valid
				offspring1.evaluateFull()
				offspring2.evaluateFull()
				stats['crossover'] += 1
			else:
				# Clone parent jika tidak crossover
				offspring1 = copy.deepcopy(parent1)
				offspring2 = copy.deepcopy(parent2)
				stats['clones'] += 1

			# 3. Mutation
			if random.random() <= self.mutationProbability:
				self.mutate(offspring1)
				stats['mutation'] += 1
			
			if random.random() <= self.mutationProbability:
				self.mutate(offspring2)
				stats['mutation'] += 1

			offspringList.append(offspring1)
			if len(offspringList) < self.populationSize:
				offspringList.append(offspring2)

		# Print Statistik Reproduksi
		if verbose:
			print(f"	[Stats] Crossover Pairs: {stats['crossover']} | "
				  f"Mutations: {stats['mutation']} | "
				  f"Clones Pairs: {stats['clones']}", flush=True)
			
		return offspringList

	def tournament(self, population) -> object:
		# Handle wrapper
		candidates = population.individuals if hasattr(population, 'individuals') else population
		participants = random.sample(candidates, 2)
		
		p1, p2 = participants[0], participants[1]
		
		# Crowding Comparison Operator
		if p1.frontRank < p2.frontRank:
			return p1
		elif p1.frontRank > p2.frontRank:
			return p2
		else:
			if p1.crowdingDistance > p2.crowdingDistance:
				return p1
			else:
				return p2

	def generatePopulation(self):
		# Gunakan list kosong [] agar kompatibel dengan __init__ Population
		self.population = Population([]) 
		
		for _ in range(self.populationSize):
			chromosome_list = self._generate_chromosome_random_first_fit()
			ind = self._create_individual_from_list(chromosome_list)
			self.population.append(ind)

	def _generate_chromosome_random_first_fit(self) -> list:
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

			if not is_placed:
				chromosome[vm_idx] = random.randint(0, self.problem.N_P - 1)

		return chromosome
	
	def repair(self, individual):
		# Implementasi default (kosong)
		# Subclass atau mixin bisa meng-override ini
		pass 

	@abstractmethod
	def _create_individual_from_list(self, chromosome_list: list):
		pass

	@abstractmethod
	def crossover(self, parent1, parent2):
		pass

	@abstractmethod
	def mutate(self, offspring):
		pass