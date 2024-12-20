from fastapi import FastAPI, File, UploadFile, status, Form
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import rasterio as rio
import io
import os
from PIL import Image
from skimage.transform import resize
from functools import partial
from typing import Dict
import asyncio
from prometheus_fastapi_instrumentator import Instrumentator
import prometheus_client

from src.eotdl_wrapper import ModelWrapper
from src.batch import BatchProcessor

__version__ = "2024.12.18"

app = FastAPI(
    title="ml-inference",
    version=__version__,
    description="API to perform inference on models hosted on EOTDL.",
)

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def hello():
    return {
        "message": "Hello from the ml-inference API!",
        "version": __version__,
    }


DOWNLOAD_PATH = os.getenv("EOTDL_DOWNLOAD_PATH", "/tmp")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 1))

batch_processors: Dict[str, BatchProcessor] = {}

model_counter = prometheus_client.Counter(
    "model_counter",
    "Number of models processed",
    labelnames=["model"]
)

@app.post("/{model}")
async def inference(
    model: str, 
    image: UploadFile = File(...), 
    version: int = Form(None)
):
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
                timeout=BATCH_TIMEOUT
            )
        # load image in memory as numpy array
        with rio.open(io.BytesIO(image.file.read())) as src:
            image = src.read()
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
            return outputs
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
        print("ERROR", "inference")
        print(e)
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
def retrieve_model_metadata(model: str, version: int = None):
    try:
        model = ModelWrapper(model, path=DOWNLOAD_PATH, version=version)
        return model.gdf.to_json()
    except Exception as e:
        print("ERROR", "retrieve_model_metadata")
        print(e)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
