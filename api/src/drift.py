import numpy as np
import prometheus_client

# we assume all samples will be tif images with 3 dimensions (bands, height, width)

dimension_gauges = {
	'bands': prometheus_client.Gauge(
		'input_bands',
		'Number of bands in the input TIF image',
		labelnames=["model"]
	),
	'height': prometheus_client.Gauge(
		'input_height',
		'Height of the input TIF image',
		labelnames=["model"]
	),
	'width': prometheus_client.Gauge(
		'input_width',
		'Width of the input TIF image',
		labelnames=["model"]
	)
}

class DriftDetector:
	def __init__(self, model, window_size=10):
		self.model = model
		self.window_size = window_size
		self.current_window = []
		self.drift_metrics = {}
		self.dimension_thresholds = {
			'bands': 100,
			'height': 10000,  # Adjust these thresholds based on your needs
			'width': 10000
		}

	def add_sample(self, sample):
		self.current_window.append(sample)
		if len(self.current_window) > self.window_size:
			self.compute_drift()	
			self.report_drift()
			self.alerts()
			self.current_window = []
					 
	def compute_drift(self):
		# Initialize lists for each dimension
		dimensions = {
			'bands': [],
			'height': [],
			'width': []
		}
		
		# Collect dimensions from all samples in the window
		for sample in self.current_window:
			shape = np.array(sample).shape
			dimensions['bands'].append(shape[0])    # First dimension for bands
			dimensions['height'].append(shape[1])   # Second dimension for height
			dimensions['width'].append(shape[2])    # Third dimension for width
		
		# Store computed means in drift metrics
		self.drift_metrics = {
			dim_name: np.mean(sizes) 
			for dim_name, sizes in dimensions.items()
		}

	def report_drift(self):
		# Report dimension metrics to Prometheus
		for dim_name, mean_size in self.drift_metrics.items():
			dimension_gauges[dim_name].labels(model=self.model).set(mean_size)

	def alerts(self):
		for dim_name, mean_size in self.drift_metrics.items():
			threshold = self.dimension_thresholds[dim_name]
			if mean_size > threshold:
				print(f"⚠️ WARNING: {dim_name} mean size ({mean_size:.1f}) "
					  f"exceeds threshold of {threshold} ❗")
