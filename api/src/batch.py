import asyncio
import numpy as np

from .metrics import model_inference_duration, model_inference_batch_size, model_inference_timeout

class BatchProcessor:
	def __init__(
		self,
		model,
		batch_size: int = 16,
		timeout: float = 0.2,  # 200ms
		drift_detector = None
	):
		self.model = model
		self.batch_size = batch_size
		self.timeout = timeout
		self.processing = False
		self.batches = []
		self.batch_timer = None

		
	async def add_item(self, item, callback):
		# add item to batch
		request_data = {
			"data": item,
			"callback": callback,
		}
		last_batch = self.batches[-1] if self.batches else None
		if last_batch and len(last_batch) < self.batch_size:
			last_batch.append(request_data)
			# print(f"Added item to last batch. Current size: {len(last_batch)}/{self.batch_size}", flush=True)
		else:
			self.batches.append([request_data])
			if self.batch_timer is None:
				self.batch_timer = asyncio.create_task(self._timeout_handler())
			# print(f"Started new batch. Current size: {len(self.batches)}/{self.batch_size}", flush=True)
		# check if we need to process the batch
		first_batch = self.batches[0]
		if len(first_batch) == self.batch_size:
			model_inference_batch_size.labels(model=self.model.model_name).set(self.batch_size)
			if self.batch_timer:
				self.batch_timer.cancel()
				self.batch_timer = None
			await self.process_batch()

	async def _timeout_handler(self):
		await asyncio.sleep(self.timeout)
		if self.batches:  # Only process if there are batches waiting
			# print(f"Timeout reached, processing incomplete batch of size {len(self.batches[0])}", flush=True)
			model_inference_timeout.labels(model=self.model.model_name).inc()
			await self.process_batch()
		self.batch_timer = None
		
	async def process_batch(self):
		current_batch = self.batches.pop(0)
		# Prepare batch data
		batch_data = np.concatenate([item["data"] for item in current_batch], axis=0)
		with model_inference_duration.labels(model=self.model.model_name).time():
			batch_results = self.model.predict(batch_data)
		# Distribute results
		for idx, item in enumerate(current_batch):
			await item["callback"](batch_results[idx])
		