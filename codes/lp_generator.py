import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os

from problem import Problem

def create_VMP_MOMILP_File(problem, output_filename=None):
	N_V = problem.N_V
	N_P = problem.N_P
	v_cpu = problem.v_cpu
	v_mem = problem.v_mem
	p_cpu = problem.p_cpu
	p_mem = problem.p_mem
	p_net = problem.p_net
	PC_max = problem.PC_max
	PC_idle = problem.PC_idle
	T_matrix = problem.T_matrix
	C_matrix = problem.C_matrix
	e_vector = problem.e_vector
	g_vector = problem.g_vector

	vm_indices = range(N_V)
	server_indices = range(N_P)

	# Define constant P_{ij}
	P_const = np.ndarray((N_V, N_P))
	for i in vm_indices:
		for j in server_indices:
			cap = p_cpu[j] if p_cpu[j] > 0 else 1.0
			P_const[i,j] = v_cpu[i] * (PC_max[j] - PC_idle[j]) / cap

	# Define constant B_{ij}		
	B_const = np.ndarray((N_V, N_P))
	for i in vm_indices:
		for j in server_indices:
			B_const[i,j] = e_vector[i] * g_vector[j]

	# Define constant A_{ijkl}
	# A_const = np.ndarray((N_V, N_P, N_V, N_P))
	# for i in vm_indices:
	# 	for j in server_indices:
	# 		for k in vm_indices:
	# 			for l in server_indices:
	# 				A_const[i,j,k,l] = 0.5 * T_matrix[i,k] * C_matrix[j,l]

	try:
		# Construct a model
		model = gp.Model("VMP_Linearized")

		# Define its decision variables: x_{ij}, y_{j}, and w_{ijkl}
		x_vars = model.addVars(vm_indices, server_indices, 
			vtype=GRB.BINARY, name="x")
		y_vars = model.addVars(server_indices, 
			vtype=GRB.BINARY, name="y")

		w_indices = []
		for i in vm_indices:
			for k in vm_indices:
				if i < k and T_matrix[i,k] > 0: # Upper triangle & Non-zero traffic
					for j in server_indices:
						for l in server_indices:
							w_indices.append((i,j,k,l))

		w_vars = model.addVars(w_indices, 
			vtype=GRB.BINARY, name="w")

		# Define the first objective (O1'): total power consumption
		obj_pow = gp.quicksum(P_const[i,j] * x_vars[i,j] 
			for i in vm_indices 
			for j in server_indices)
		obj_pow += gp.quicksum(PC_idle[j] * y_vars[j] for j in server_indices)

		# Define the second objective (O2'): total network communication cost
		obj_net = gp.quicksum(T_matrix[i,k] * C_matrix[j,l] * w_vars[i,j,k,l] 
			for (i,j,k,l) in w_indices)

		obj_net += gp.quicksum(B_const[i,j] * x_vars[i,j] 
			for i in vm_indices 
			for j in server_indices)

		# Set as multiobjective optimization model with two objectives
		model.NumObj = 2

		# priority, weight, abstol & reltol are left to be in default values
		# these parameters will be overwritten by PaMILO
		model.setObjectiveN(obj_pow, index=0, priority=1, name='PowerConsumption')
		model.setObjectiveN(obj_net, index=1, priority=0, name='CommunicationCost')

		# All objectives are to be minimized
		model.ModelSense = GRB.MINIMIZE

		# Define constraint (V1')
		model.addConstrs(
			(x_vars.sum(i, '*') == 1 for i in vm_indices), 
			name="V1_OneVMOneServer")

		# Define constraint (V2')
		model.addConstrs(
			(gp.quicksum(v_cpu[i] * x_vars[i,j] for i in vm_indices) <= p_cpu[j] * y_vars[j]
				for j in server_indices), 
			name="V2_CpuCapacity")
		# Define constraint (V3')
		model.addConstrs(
			(gp.quicksum(v_mem[i] * x_vars[i,j] for i in vm_indices) <= p_mem[j] * y_vars[j]
				for j in server_indices), 
			name="V3_MemCapacity")

		# Define constraint (V4')
		# Step A: Prepare Expression Buckets per Server
		# Kita kumpulkan dulu term-nya ke list biar quicksum-nya efisien
		server_traffic_expr = {j: [] for j in server_indices}

		# Step B: External Traffic (VM -> Internet)
		# Langsung tambahkan e_vector[i] * x[i,j] ke bucket server j
		for j in server_indices:
			for i in vm_indices:
				if e_vector[i] > 0:
					server_traffic_expr[j].append(e_vector[i] * x[i,j])

		# Step C: Internal Traffic (VM <-> VM)
		# Kita manfaatkan w_indices yang sudah sparse
		for (i, j, k, l) in w_indices:
			# Jika server j != l, berarti ada trafik fisik lewat NIC
			if j != l:
				traffic_val = T_matrix[i,k]
				# Trafik membebani Server j (karena VM i ada di situ)
				server_traffic_expr[j].append(traffic_val * w[i,j,k,l])
				# Trafik membebani Server l (karena VM k ada di situ)
				server_traffic_expr[l].append(traffic_val * w[i,j,k,l])

		# Step D: Add Constraint to Model
		for j in server_indices:
			if server_traffic_expr[j]: # Hanya jika ada trafik
				model.addConstr(
					gp.quicksum(server_traffic_expr[j]) <= p_net[j] * y[j],
					name=f"V4_NetCap_{j}")

		# Define constraint (V5')
		model.addConstrs((x_vars[i,j] <= y_vars[j] 
					  for i in vm_indices for j in server_indices), name="V5_ActiveServer")

		# Define constraint (V6')
		model.addConstrs((w_vars[i,j,k,l] <= x_vars[i,j]
					  for (i,j,k,l) in w_indices), name="V6_wx") 
		# Define constraint (V7')
		model.addConstrs((w_vars[i,j,k,l] <= x_vars[k,l] 
					  for (i,j,k,l) in w_indices), name="V7_wx")
		# Define constraint (V8')
		model.addConstrs((w_vars[i,j,k,l] >= x_vars[i,j] + x_vars[k,l] - 1 
					  for (i,j,k,l) in w_indices), name="V8_wx")

		# Save model as .lp file
		# output_filename = "vmp_momilp_model.lp"

		os.makedirs(os.path.dirname(output_filename) or '.', exist_ok=True)

		model.write(output_filename)
		
		print(f"\nThe model has been written to: {output_filename}")
		print("Ready to be used as input for PaMILO.")

	except gp.GurobiError as e:
		print(f"Error Gurobi (code {e.errno}): {e.message}")
	except Exception as e:
		print(f"Error occured: {e}")

# # --- Jalankan fungsi pembuat file ---
# if __name__ == "__main__":
#	 create_vmp_momilp_file(problem, output_filename)		