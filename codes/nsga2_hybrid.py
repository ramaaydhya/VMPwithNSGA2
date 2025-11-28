import random
import copy

from individual_hybrid import IndividualHybrid
from nsga2 import NSGA2
from population import Population
from problem import Problem

class NSGA2Hybrid(NSGA2):
	def __init__(self, problem, populationSize=100, maxGeneration=100, crossoverProbability=0.9, mutationProbability=0.1):
		super().__init__(problem, populationSize, maxGeneration, crossoverProbability, mutationProbability)

	def crossover(self, parent1, parent2):
		offspring_1 = self._gga_crossover(parent1, parent2)
		offspring_2 = self._gga_crossover(parent2, parent1)

		return offspring_1, offspring_2

	def _gga_crossover(self, donor, receiver):
		active_servers_donor = [s for s, vms in donor.server_map.items() if vms]

		# Fallbackk if donor is empty (very unlikely to happen) 
		if not active_servers_donor:
			return copy.deepcopy(receiver)

		# Select random servers to inject with random sample size (<=50%) 
		num_inject = random.randint(1, max(1, len(active_servers_donor) // 2))	
		servers_to_inject = random.sample(active_servers_donor, num_inject)

		offspring_server_map = {}
		injected_vms = set()

		# ==== 1. Injection from donor ====
		for server_idx in servers_to_inject:
			vm_list = list(donor.server_map[server_idx])
			offspring_server_map[server_idx] = vm_list
			injected_vms.update(vm_list)

		# ==== 2. Inheritance from receiver ====
		for server_idx, vm_list in receiver.server_map.items():
			# Skip servers that have been already injected in the offspring chromosome
			if server_idx in offspring_server_map:
				continue
			# Collect all VMs that have not been placed yet		
			remaining_vms = [vm for vm in vm_list if vm not in injected_vms]
			# Inject those
			if remaining_vms:
				offspring_server_map[server_idx] = remaining_vms		

		# ==== 3. Reinsert ====

		# Identify all VMs that have been already placed
		current_placed_vms = set()
		for vm_list in offspring_server_map.values():
			current_placed_vms.update(vm_list)

		# Collect all unplaced VMs
		unplaced_vms = [vm for vm in range(self.problem.N_V) if vm not in current_placed_vms]
		
		# Reinsert those unplaced VMs randomly
		random.shuffle(unplaced_vms)
		self._reinsert_vms(offspring_server_map, unplaced_vms)

		# Construct offspring
		offspring = IndividualHybrid(self.problem, offspring_server_map)	
		offspring.evaluateFull()
		return offspring

	def mutate(self, individual):
		# Identify all active servers
		active_servers = [s for s, vms in individual.server_map.items() if vms]
		if not active_servers: return

		# Pick random server to kill
		server_to_kill = random.choice(active_servers)
		# Collect every VMs inside it
		vms_to_move = list(individual.server_map[server_to_kill])
		# Empty the server
		individual.server_map[server_to_kill] = []

		# Migrate these VMs:
		# Pick random VM
		random.shuffle(vms_to_move)
		for vm_idx in vms_to_move:
			is_target_found = False

			# Pick random candidate target
			candidates = [s for s in range(self.problem.N_P) if s != server_to_kill]
			random.shuffle(candidates)

			# Check if the candidate can fit this VM
			for target_server in candidates:
				p_cpu = self.problem.p_cpu[target_server]
				p_mem = self.problem.p_mem[target_server]
				cur_cpu = individual.total_cpu_per_server[target_server]
				cur_mem = individual.total_mem_per_server[target_server]

				req_cpu = self.problem.v_cpu[vm_idx]
				req_mem = self.problem.v_mem[vm_idx]

				# If the VM can fit into this server, place the VM there
				if (cur_cpu + req_cpu <= p_cpu) and (cur_mem + req_mem <= p_mem):
					individual.evaluateDelta(vm_idx, target_server)
					is_target_found = True
					break
				# Otherwise, pick another random candidate and try again
			
			# If it fits into nowhere, then the solution has already been invalid in the first place
			# Pick arbitrary server
			if not is_target_found:
				fallback = random.choice(candidates)
				individual.evaluateDelta(vm_idx, fallback)

		# If the mutated individual is invalid, repair it
		if individual.isConstraintViolated:
			self.repair(individual)

	def _create_individual_from_list(self, chromosome_list):
		server_map = {}
		for vm_idx, server_idx in enumerate(chromosome_list):
			if server_idx not in server_map:
				server_map[server_idx] = []
			server_map[server_idx].append(vm_idx)	

		return IndividualHybrid(self.problem, server_map)

	def _reinsert_vms(self, server_map, unplaced_vms):		
		current_cpu = {s: 0 for s in range(self.problem.N_P)}
		current_mem = {s: 0 for s in range(self.problem.N_P)}
		
		# Calculate resource usage for every server
		for server_idx, vm_list in server_map.items():
			for vm_idx in vm_list:
				current_cpu[server_idx] += self.problem.v_cpu[vm_idx]
				current_mem[server_idx] += self.problem.v_mem[vm_idx]
				
		# Reinsert unplaced VMs
		for vm_idx in unplaced_vms:
			req_cpu = self.problem.v_cpu[vm_idx]
			req_mem = self.problem.v_mem[vm_idx]
			is_placed = False
			
			# Pick one random active server
			active_servers = list(server_map.keys())
			random.shuffle(active_servers) # Shuffle
			
			for server_idx in active_servers:
				# If the VM fits into this server, placed the VM there 
				if (current_cpu[server_idx] + req_cpu <= self.problem.p_cpu[server_idx] and 
					current_mem[server_idx] + req_mem <= self.problem.p_mem[server_idx]):
					
					server_map[server_idx].append(vm_idx)
					current_cpu[server_idx] += req_cpu
					current_mem[server_idx] += req_mem
					is_placed = True
					# Proceed to the next VM
					break
				# Otherwise, try another active server
			
			# If it doesn't fit into any active server, activate first idle server 
			if not is_placed:
				for server_idx in range(self.problem.N_P):
					# Skip active server (already checked)
					if server_idx in server_map and server_map[server_idx]:
						continue
					if (req_cpu <= self.problem.p_cpu[server_idx] and 
					    req_mem <= self.problem.p_mem[server_idx]):
						
						if server_idx not in server_map:
							server_map[server_idx] = []
						server_map[server_idx].append(vm_idx)
						
						current_cpu[server_idx] += req_cpu
						current_mem[server_idx] += req_mem
						is_placed = True
						# Proceed to the next VM
						break

			# Fallback (Random)
			if not is_placed:
				random_server = random.randint(0, self.problem.N_P - 1)
				if random_server not in server_map: 
					server_map[random_server] = []
				server_map[random_server].append(vm_idx)	