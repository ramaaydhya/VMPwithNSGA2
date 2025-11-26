import numpy as np
import json

class Problem:
	def __init__(self):
		N_V: int = 0
		N_P: int = 0

		v_cpu: np.ndarray = None
		v_mem: np.ndarray = None
		
		p_cpu: np.ndarray = None
		p_mem: np.ndarray = None
		p_net: np.ndarray = None
		
		PC_max: np.ndarray = None
		PC_idle: np.ndarray = None
		
		T_matrix: np.ndarray = None
		C_matrix: np.ndarray = None
		e_vector: np.ndarray = None
		g_vector: np.ndarray = None

	def loadFromFile(self, filepath):
		with open(filepath, 'r') as f:
			data = json.load(f)

		self.N_P = data['meta']['num_servers']
		self.N_V = data['meta']['num_vms']

		self.p_cpu = np.array([server['p_cpu'] for server in data['servers']])
		self.p_mem = np.array([server['p_mem'] for server in data['servers']])
		self.p_net = np.array([server['p_net'] for server in data['servers']])

		self.PC_idle = np.array([server['pc_idle'] for server in data['servers']])
		self.PC_max = np.array([server['pc_max'] for server in data['servers']])

		self.v_cpu = np.array([vm['v_cpu'] for vm in data['vms']])
		self.v_mem = np.array([vm['v_mem'] for vm in data['vms']])

		self.T_matrix = np.array(data['traffic_matrix'])
		self.C_matrix = np.array(data['cost_matrix'])

		self.e_vector = np.array(data['e_vector'])
		self.g_vector = np.array(data['g_vector'])