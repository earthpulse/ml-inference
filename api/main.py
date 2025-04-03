from fastapi import FastAPI, File, UploadFile, status, Form, Depends, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.responses import StreamingResponse
import rasterio as rio
import io
import os
from PIL import Image
from skimage.transform import resize
from typing import Dict, Optional
import asyncio
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import traceback
import json
from api.src.eotdl_wrapper import ModelWrapper
from api.src.batch import BatchProcessor
from api.src.drift import DriftDetector
from api.src.metrics import model_counter, model_error_counter

__version__ = "2025.02.26"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
	title="ml-inference",
	version=__version__,
	description="API to perform inference on models hosted on EOTDL.",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# API Key authentication setup
API_KEY = os.getenv("API_KEY", None)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
	if API_KEY:
		if not api_key:
			logger.error("API Key is required")
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail="API Key is required"
			)
		if api_key != API_KEY:
			logger.error("Invalid API Key")
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail="Invalid API Key"
			)
	return api_key

@app.get("/")
async def hello():
	return {
		"message": "Hello from the ml-inference API!",
		"version": __version__,
		"auth_required": API_KEY is not None
	}

DOWNLOAD_PATH = os.getenv("EOTDL_DOWNLOAD_PATH", "/tmp")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 1))
DRIFT_DETECTION = os.getenv("DRIFT_DETECTION", "false")
RATE_LIMIT = os.getenv("RATE_LIMIT", None) 

batch_processors: Dict[str, BatchProcessor] = {}
drift_detector: Dict[str, DriftDetector] = {}

@app.post("/{model}")
@limiter.limit(RATE_LIMIT)
async def inference(
	request: Request,  
	model: str, 
	image: UploadFile = File(...), 
	version: int = Form(None),
	api_key: str = Depends(verify_api_key)
):
	# Implementation
	try:
		model_counter.labels(model=model).inc()
		# download model from ETODL
		# download_path, stac_df = download_model(model, DOWNLOAD_PATH, version)
		model_wrapper = ModelWrapper(model, path=DOWNLOAD_PATH, version=version)
		# initialize batch processor if not already initialized
		if model not in batch_processors:
			batch_processors[model] = BatchProcessor(
				model=model_wrapper,
				batch_size=BATCH_SIZE,
				timeout=BATCH_TIMEOUT,
			)
		# load image in memory as numpy array
		with rio.open(io.BytesIO(image.file.read())) as src:
			image = src.read()
		if DRIFT_DETECTION == 'true':
			if model not in drift_detector:
				drift_detector[model] = DriftDetector(model)
			drift_detector[model].add_sample(image)
		assert image.ndim == 3, "Image must have 3 dimensions (bands, height, width)"
		# pre-process and validate input
		image = model_wrapper.process_inputs(image)
		# normalization
		image = (
			image / 255.0
		)  # this should be defined in the model metadata (this is only for RGB will not work with satellite images)
		# TODO: resize if necessary
		# execute model
		# outputs = model.predict(image)
		outputs = await process_in_batch(
			image,
			batch_processors[model]
		)
		# return outputs
		if model_wrapper.props["mlm:output"]["tasks"] == ["classification"]:
			return outputs.tolist()
		elif model_wrapper.props["mlm:output"]["tasks"] == ["segmentation"]:  # only returns first output as image
			# return mask
			# image = sigmoid(outputs) > 0.5  # this should be defined in the model metadata
			if outputs.ndim == 3:  # get first band
				outputs = outputs[0]
			outputs = resize(outputs, model_wrapper.original_size, preserve_range=True)
			# outputs = outputs.astype(np.uint8)
			img = Image.fromarray(outputs, mode="F")  # only returns binary mask
			buf = io.BytesIO()
			img.save(buf, "tiff")
			buf.seek(0)
			return StreamingResponse(buf, media_type="image/tiff") # TODO: should be COG
		else:
			raise Exception(
				"Output task not supported", model.props["mlm:output"]["tasks"]
			)
	except Exception as e:
		logger.error(f"Error in inference: {e}")
		traceback.print_exc()
		model_error_counter.labels(model=model, error_type="inference").inc()
		raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

async def process_in_batch(image, processor: BatchProcessor):
	# Create an event to wait for the result
	result_event = asyncio.Event()
	result_container = {"result": None}
	# Callback function to be called when batch processing is done
	async def on_result(result):
		result_container["result"] = result
		result_event.set()
	# Add to batch processor
	await processor.add_item(image, on_result)
	# Wait for result
	await result_event.wait()
	return result_container["result"]

@app.get("/{model}")
async def retrieve_model_metadata(
	model: str, 
	version: int = None,
	api_key: str = Depends(verify_api_key)
):
	try:
		model = ModelWrapper(model, path=DOWNLOAD_PATH, version=version)
		return model.items()
	except Exception as e:
		logger.error(f"Error in retrieve_model_metadata: {e}")
		traceback.print_exc()
		raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
