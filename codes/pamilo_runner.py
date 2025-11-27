import subprocess
import os
import time
import multiprocessing

class PaMILORunner:
	def __init__(self, pamilo_executable_path):
		self.exec_path = pamilo_executable_path

	def solve(self, input_lp_path, output_json_path, timeout_sec=3600):
		if not os.path.exists(self.exec_path):
			print(f"[PaMILO] Error: Executable not found at {self.exec_path}")
			return False

		os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
		
		num_threads = os.cpu_count()

		if num_threads is None or num_threads < 1:
			num_threads = 2

		print(f"[PaMILO] Detected {num_threads} CPU cores. Setting -t {num_threads}")

		cmd = [
			self.exec_path, 
			input_lp_path,
			"-o", output_json_path,	 # Flag output
			"-t", str(num_threads)	  # Flag thread limit
		]

		print(f"[PaMILO] Running command: {' '.join(cmd)}")
		start_time = time.time()

		try:
			result = subprocess.run(
				cmd,
				capture_output=True,
				text=True,
				timeout=timeout_sec
				)

			elapsed = time.time() - start_time

			if result.returncode == 0:
				print(f"[PaMILO] Success! Time: {elapsed:.2f}s")
				if os.path.exists(output_json_path):
					print(f"[PaMILO] Output saved to: {output_json_path}")
					return True
				else:
					print("[PaMILO] Success reported, but output file missing.")
					return False
			else:
				print(f"[PaMILO] Failed with return code {result.returncode}")
				print("STDERR:", result.stderr)
				return False
		except subprocess.TimeoutExpired:
			print(f"[PaMILO] Timeout reached after {timeout_sec} seconds (Python kill).")
			return False
		except Exception as e:
			print(f"[PaMILO] Execution error: {e}")
			return False					