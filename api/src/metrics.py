import prometheus_client

model_counter = prometheus_client.Counter(
    "model_counter",
    "Number of models requested",
    labelnames=["model"]
)

model_error_counter = prometheus_client.Counter(
    "model_inference_errors_total",
    "Number of model inference errors",
    labelnames=["model", "error_type"]
)

model_inference_duration = prometheus_client.Histogram(
    "model_inference_duration_seconds",
    "Time spent processing inference requests",
    labelnames=["model"]
)

model_inference_batch_size = prometheus_client.Gauge(
    "model_inference_batch_size",
    "Number of images in the batch",
    labelnames=["model"]
)

model_inference_timeout = prometheus_client.Counter(
    "model_inference_timeout_total",
    "Number of inference requests that timed out",
    labelnames=["model"]
)
