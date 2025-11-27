import numpy as np

class PerformanceMetrics:
	
	@staticmethod
	def calculate_spacing(front):
		
		if len(front) < 2:
			return 0.0

		# Urutkan berdasarkan objektif pertama untuk perhitungan jarak tetangga di 2D
		# (Untuk >2D, butuh nearest neighbor search, tapi VMP biasanya 2D)
		sorted_front = front[np.argsort(front[:, 0])]
		
		# Hitung jarak Euclidean antar solusi bertetangga
		distances = []
		for i in range(len(sorted_front) - 1):
			d = np.linalg.norm(sorted_front[i] - sorted_front[i+1])
			distances.append(d)
			
		d_mean = np.mean(distances)
		sum_sq_diff = sum((d - d_mean) ** 2 for d in distances)
		
		return np.sqrt(sum_sq_diff / (len(front) - 1))

	@staticmethod
	def calculate_hypervolume(front, ref_point):
		"""
		Menghitung Hypervolume untuk 2 Objektif.
		front: Array (N, 2) yang SUDAH DINORMALISASI.
		ref_point: Biasanya [1.1, 1.1] jika dinormalisasi.
		"""
		# Filter solusi yang melebihi reference point (didominasi ref point)
		valid_points = []
		for point in front:
			if point[0] <= ref_point[0] and point[1] <= ref_point[1]:
				valid_points.append(point)
		
		if not valid_points:
			return 0.0

		valid_points = np.array(valid_points)
		sorted_front = valid_points[np.argsort(valid_points[:, 0])]

		area = 0.0

		for i in range(len(sorted_front)):
			current_x = sorted_front[i][0]
			current_y = sorted_front[i][1]

			if i < len(sorted_front) - 1:
				next_x = sorted_front[i+1][0]
			else:
				next_x = ref_point[0]
			width = next_x - current_x
			height = ref_point[1] - current_y

			area += width * height

		return area			

	@staticmethod
	def _distance_plus(point_a, point_z):
		"""Modified Euclidean Distance untuk IGD+/GD+ (hanya menghitung degradasi)"""
		# d+(a, z) = sqrt( sum( max(a_i - z_i, 0)^2 ) )
		# Jika a lebih baik dari z (lebih kecil), jaraknya 0.
		diff = np.maximum(point_a - point_z, 0)
		return np.linalg.norm(diff)

	@staticmethod
	def calculate_gd_plus(front, ref_front):
		"""
		Generational Distance Plus (GD+).
		Mengukur seberapa dekat Front kita ke Reference Front.
		Lebih kecil = Lebih baik (Konvergensi).
		"""
		sum_dist = 0.0
		for solution in front:
			# Cari jarak terdekat ke salah satu solusi di Reference Front
			min_dist = float('inf')
			for ref_sol in ref_front:
				dist = PerformanceMetrics._distance_plus(solution, ref_sol)
				if dist < min_dist:
					min_dist = dist
			sum_dist += min_dist**2 # GD biasanya dikuadratkan dulu
			
		return np.sqrt(sum_dist) / len(front)

	@staticmethod
	def calculate_igd_plus(front, ref_front):
		"""
		Inverted Generational Distance Plus (IGD+).
		Mengukur konvergensi DAN diversity.
		Loop dari Reference Front ke Front kita.
		"""
		sum_dist = 0.0
		for ref_sol in ref_front:
			# Cari solusi terdekat di front kita untuk titik referensi ini
			min_dist = float('inf')
			for solution in front:
				# Perhatikan urutan parameter untuk distance plus!
				# d+(ref, sol) -> kita mau tahu seberapa jauh sol dari ref
				# Tapi IGD+ mendefinisikan d+(z, a) = max(a - z, 0)
				# dimana z element Z (Ref), a element A (Approx)
				dist = PerformanceMetrics._distance_plus(solution, ref_sol)
				if dist < min_dist:
					min_dist = dist
			sum_dist += min_dist # IGD+ biasanya tidak di-akar rata-rata, tapi rata-rata langsung
			
		return sum_dist / len(ref_front)