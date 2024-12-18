from typing import List, Dict, Any, Callable
import asyncio
import time
import numpy as np

class BatchProcessor:
	def __init__(
		self,
		model,
		batch_size: int = 16,
		timeout: float = 0.2,  # 200ms
	):
		self.model = model
		self.batch_size = batch_size
		self.timeout = timeout
		self.pending_batch: List[Dict[str, Any]] = []
		self.batch_lock = asyncio.Lock()
		self.timeout_task = None
		self.processing = False
		
	async def add_item(self, item: np.ndarray, callback: Callable):
		"""Add an item to the pending batch."""
		print(">>> Starting add_item method")
		
		done_event = asyncio.Event()
		request_data = {
			"data": item,
			"callback": callback,
			"event": done_event
		}

		async with self.batch_lock:
			self.pending_batch.append(request_data)
			current_size = len(self.pending_batch)
			print(f"Added item to batch. Current size: {current_size}/{self.batch_size}")

			# Start timeout task if there isn't one already running
			if not self.timeout_task:
				print(">>> Creating new timeout task...")
				# Create and schedule the timeout task
				self.timeout_task = asyncio.ensure_future(self._timeout_handler())
				print(">>> Created new timeout task")

			should_process = current_size >= self.batch_size and not self.processing

		if should_process:
			if self.timeout_task:
				self.timeout_task.cancel()
				self.timeout_task = None
			print(f"🎯 Processing batch of size {len(self.pending_batch)} due to BATCH FULL")
			await self.process_batch()

		print(">>> Waiting for done_event")
		await done_event.wait()
		print(">>> Done waiting")

	async def _timeout_handler(self):
		"""Handle timeout for incomplete batches."""
		print(">>> Timeout handler started")
		try:
			print(f"⏳ Starting timeout of {self.timeout}s for batch of size {len(self.pending_batch)}")
			await asyncio.sleep(self.timeout)
			print(">>> Timeout sleep completed")
			
			async with self.batch_lock:
				if self.pending_batch and not self.processing:  # If there are still pending items
					print(f"⏰ Processing batch of size {len(self.pending_batch)} due to TIMEOUT")
					await self.process_batch()
		except asyncio.CancelledError:
			print("❌ Timeout task was cancelled")
		except Exception as e:
			print(f"❌ Timeout handler error: {str(e)}")
			raise

	async def process_batch(self):
		"""Process all items in the current batch."""
		if not self.pending_batch:
			return
		self.processing = True
		try:
			async with self.batch_lock:
				if not self.pending_batch:
					return

				# Get current batch and clear pending
				current_batch = self.pending_batch
				self.pending_batch = []
				
				# Reset timeout task since we're processing the batch
				if self.timeout_task:
					self.timeout_task.cancel()
					self.timeout_task = None

			# Prepare batch data
			batch_data = np.concatenate([item["data"] for item in current_batch], axis=0)
			batch_results = self.model.predict(batch_data)
			
			# Distribute results
			for idx, item in enumerate(current_batch):
				await item["callback"](batch_results[idx])
				item["event"].set()  # Signal completion
				
		except Exception as e:
			print(f"Batch processing error: {str(e)}")
			for item in current_batch:
				await item["callback"](Exception(f"Batch processing failed: {str(e)}"))
				item["event"].set()
		finally:
			self.processing = False
			# Start a new timeout task if there are pending items
			async with self.batch_lock:
				if self.pending_batch and not self.timeout_task:
					self.timeout_task = asyncio.create_task(self._timeout_handler())
					print(f"Started new timeout task for remaining {len(self.pending_batch)} items")