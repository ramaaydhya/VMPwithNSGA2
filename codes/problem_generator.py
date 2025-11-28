import json
import random
import numpy as np
import os

# ==========================================
# 1. DATABASE SPESIFIKASI 
# ==========================================

# Spesifikasi Server
# Diambil dari Uhlig et al. (2025)
# 
SERVER_TYPES_SPEC = [
    {'id': 1, 'cpu': 64, 'mem': 512, 'pc_idle': 300, 'pc_max': 700, 'bw': 50},
    {'id': 2, 'cpu': 64, 'mem': 512, 'pc_idle': 300, 'pc_max': 700, 'bw': 50},
	{'id': 3, 'cpu': 96, 'mem': 1024, 'pc_idle': 400, 'pc_max': 900, 'bw': 100},
    {'id': 4, 'cpu': 96, 'mem': 1024, 'pc_idle': 400, 'pc_max': 900, 'bw': 100},
    {'id': 5, 'cpu': 96, 'mem': 1024, 'pc_idle': 400, 'pc_max': 900, 'bw': 100},
	{'id': 6, 'cpu': 128, 'mem': 2048, 'pc_idle': 500, 'pc_max': 1200, 'bw': 200},
    {'id': 7, 'cpu': 128, 'mem': 2048, 'pc_idle': 500, 'pc_max': 1200, 'bw': 200},
	{'id': 8, 'cpu': 224, 'mem': 4096, 'pc_idle': 600, 'pc_max': 2000, 'bw': 200},
    {'id': 9, 'cpu': 224, 'mem': 4096, 'pc_idle': 600, 'pc_max': 2000, 'bw': 200},
    {'id': 10, 'cpu': 224, 'mem': 4096, 'pc_idle': 600, 'pc_max': 2000, 'bw': 200},
]

# These VM types are AWS EC2 instance types
VM_TYPES_SPEC = [
	# c5 series: Compute-optimized
	{'name': 'c5.large', 'cpu': 1, 'mem': 4, 'bw': 10.0},
	{'name': 'c5.xlarge', 'cpu': 2, 'mem': 8, 'bw': 10.0},
	{'name': 'c5.2xlarge', 'cpu': 4, 'mem': 16, 'bw': 10.0},
	{'name': 'c5.4xlarge', 'cpu': 8, 'mem': 32, 'bw': 10.0},
	# r5 series: Memory-optimized
	{'name': 'r5.large', 'cpu': 1, 'mem': 16, 'bw': 10.0},
	{'name': 'r5.xlarge', 'cpu': 2, 'mem': 32, 'bw': 10.0},
	{'name': 'r5.2xlarge', 'cpu': 4, 'mem': 64, 'bw': 10.0},
	{'name': 'r5.4xlarge', 'cpu': 8, 'mem': 128, 'bw': 10.0},
	# t3 series: General purpose
	{'name': 't3.micro', 'cpu': 1, 'mem': 1, 'bw': 5.0},
	{'name': 't3.small', 'cpu': 1, 'mem': 2, 'bw': 5.0},
	{'name': 't3.medium', 'cpu': 1, 'mem': 4, 'bw': 5.0},
	{'name': 't3.large', 'cpu': 1, 'mem': 8, 'bw': 5.0},
	{'name': 't3.xlarge', 'cpu': 2, 'mem': 16, 'bw': 5.0},
	{'name': 't3.2xlarge', 'cpu': 4, 'mem': 32, 'bw': 5.0},
]

SCENARIO_SPEC = {
	'small': {
		'num_servers': 20,
		'num_vms': 50,
		'num_vm_types': 4,
		'num_clusters': 5
	},
	'large': {
		'num_servers': 100,
		'num_vms': 300,
		'num_vm_types': 10,
		'num_clusters': 50
	},
}

# ==========================================
# 2. LOGIKA FAT-TREE & COST
# ==========================================


def _calculate_fattree_topology(num_servers):
	"""
	Menghitung parameter k untuk k-ary Fat-Tree.
	Kapasitas total = (k^3) / 4 host.
	"""
	k = 2
	while True:
		capacity = (k ** 3) / 4
		if capacity >= num_servers:
			break
		k += 2  # k harus genap

	servers_per_rack = k // 2
	racks_per_pod = k // 2

	return k, servers_per_rack, racks_per_pod


def getFatTreeCost(server_1, server_2, servers_per_rack, racks_per_pod):
	if server_1 == server_2:
		return 0

	rack_1 = server_1 // servers_per_rack
	rack_2 = server_2 // servers_per_rack

	if rack_1 == rack_2:
		return 1  # Intra-Rack

	pod_1 = rack_1 // racks_per_pod
	pod_2 = rack_2 // racks_per_pod

	if pod_1 == pod_2:
		return 3  # Intra-Pod

	return 5  # Inter-Pod

# ==========================================
# 3. GENERATOR UTAMA
# ==========================================


def generateProblem(filename, scenario_name, seed_value):
	random.seed(seed_value)
	np.random.seed(seed_value)

	# ==== 1. Generate Topology ====
	num_servers = SCENARIO_SPEC[scenario_name]['num_servers']
	num_vms = SCENARIO_SPEC[scenario_name]['num_vms']
	num_vm_types = SCENARIO_SPEC[scenario_name]['num_vm_types']
	num_clusters = SCENARIO_SPEC[scenario_name]['num_clusters']

	k, servers_per_rack, racks_per_pod = _calculate_fattree_topology(
		num_servers)

	print(f"[{scenario_name}] Generating {filename} (Seed {seed_value})")
	print(f"  Topology: k={k}-ary Fat-Tree (Max {int((k**3)/4)} servers)")

	# ==== 2. Generate Servers ====
	servers = []
	for i in range(num_servers):
		spec = random.choice(SERVER_TYPES_SPEC)
		servers.append({
			'id': i,
			'type': spec['id'],
			'p_cpu': spec['cpu'],
			'p_mem': spec['mem'],
			'p_net': spec['bw'],
			'pc_idle': spec['pc_idle'],
			'pc_max': spec['pc_max']
		})

	# ==== 3. Generate VMs and Cluster Assignment ====
	selected_vm_types_spec = random.sample(VM_TYPES_SPEC, num_vm_types)

	vms = []
	vm_cluster_map = []

	# Pastikan setiap cluster memiliki minimal 1 VM
	cluster_assignments = [c % num_clusters for c in range(num_vms)]
	random.shuffle(cluster_assignments)

	for i in range(num_vms):
		spec = random.choice(selected_vm_types_spec)
		cluster_id = cluster_assignments[i]

		vms.append({
			'id': i,
			'type': spec['name'],
			'v_cpu': spec['cpu'],
			'v_mem': spec['mem'],
			'v_net': spec['bw'],
			'cluster_id': cluster_id
		})
		vm_cluster_map.append(cluster_id)

	# ==== 4. Generate VM Communication Traffic ====

	# JUSTIFIKASI EMPIRIS (Polak et al. 2024):
	# Berdasarkan analisis histogram dataset real-world:
	# 1. Mean Load Factor = 0.3159
	# 2. Median Load Factor = 0.0344 (Sangat rendah, mayoritas idle)
	# 3. Elephants (High Load) = 33% dari VM

	e_vector = np.zeros(num_vms)
	T_out_list = np.zeros(num_vms)
	T_in_list = np.zeros(num_vms)

	for vm_idx in range(num_vms):
		v_net = vms[vm_idx]['v_net']

		rand_val = random.random()

		# [DATA DRIVEN LOGIC]
		if rand_val < 0.33:
			# 33% Elephants (High Load: 0.6 - 1.0)
			# Sesuai spike di histogram pada 0.65, 0.8, 1.0
			load_factor = random.uniform(0.6, 1.0)
		elif rand_val < 0.43:
			# 10% Medium Load (Gap di histogram: 0.2 - 0.6)
			load_factor = random.uniform(0.2, 0.6)
		else:
			# 57% Mice (Low Load: 0.0 - 0.06)
			# Agar Median mendekati 0.0344
			load_factor = random.uniform(0.001, 0.06)

		total_throughput = v_net * load_factor

		# Rasio External (Beta Dist)
		external_ratio = random.betavariate(1, 4)

		external_val = total_throughput * external_ratio
		internal_val = total_throughput * (1 - external_ratio)

		e_vector[vm_idx] = external_val
		T_out_list[vm_idx] = internal_val
		T_in_list[vm_idx] = min(v_net, internal_val * random.uniform(0.8, 1.2))

	# Grouping VM Indices by Cluster
	clusters = {}
	for vm_idx, cluster_id in enumerate(vm_cluster_map):
		if cluster_id not in clusters:
			clusters[cluster_id] = []
		clusters[cluster_id].append(vm_idx)

	# Matriks Sementara (Directed)
	T_matrix_directed = np.zeros((num_vms, num_vms))

	for cluster_id, vm_list in clusters.items():
		sum_T_in = sum(T_in_list[vm_idx] for vm_idx in vm_list)
		if sum_T_in == 0:
			continue

		for vm_1 in vm_list:
			for vm_2 in vm_list:
				if vm_1 == vm_2:
					continue
				# Gravity Model
				gravity_val = (T_out_list[vm_1] * T_in_list[vm_2]) / sum_T_in
				T_matrix_directed[vm_1][vm_2] = gravity_val

	# Final Symmetric Matrix
	T_matrix = np.zeros((num_vms, num_vms))

	for vm_1 in range(num_vms):
		for vm_2 in range(vm_1 + 1, num_vms):
			# Symmetrization
			total_traffic = T_matrix_directed[vm_1][vm_2] + \
				T_matrix_directed[vm_2][vm_1]

			# Inter-Cluster Noise (Sangat Kecil)
			if vm_cluster_map[vm_1] != vm_cluster_map[vm_2]:
				if random.random() < 0.005:
					noise_val = random.uniform(0.0001, 0.001)
					total_traffic += noise_val

			T_matrix[vm_1][vm_2] = total_traffic
			T_matrix[vm_2][vm_1] = total_traffic

	# ==== 5. Generate Server Communication Cost ====
	C_matrix = np.zeros((num_servers, num_servers))
	g_vector = np.zeros(num_servers)

	GATEWAY_COST_CONSTANT = 4.0

	for server_1 in range(num_servers):
		g_vector[server_1] = GATEWAY_COST_CONSTANT
		for server_2 in range(num_servers):
			C_matrix[server_1][server_2] = getFatTreeCost(
				server_1, server_2, servers_per_rack, racks_per_pod)

	# ==== 6. Save Problem Parameters to JSON ====
	problem_data = {
		'meta': {
			'scenario': scenario_name,
			'seed': seed_value,
			'num_servers': num_servers,
			'num_vms': num_vms,
			'fattree_k': k,
			'topology': 'Fat-Tree'
		},
		'servers': servers,
		'vms': vms,
		'T_matrix': T_matrix.tolist(),
		'C_matrix': C_matrix.tolist(),
		'e_vector': e_vector.tolist(),
		'g_vector': g_vector.tolist()
	}

	os.makedirs(os.path.dirname(filename), exist_ok=True)

	with open(filename, 'w') as f:
		json.dump(problem_data, f)

	print(f"  [OK] Saved to {filename}")
