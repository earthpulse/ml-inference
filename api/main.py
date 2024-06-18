from fastapi import FastAPI, File, UploadFile, status, Form
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import rasterio as rio
import onnxruntime
import io
import os
import numpy as np

from src.utils import download_model
from src.onnx import get_onnx_session

__version__ = "2024.06.18"

app = FastAPI(
    title="ml-inference",
    version=__version__,
    description="API to perform inference on models hosted on EOTDL.",
)

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

# TODO: normalization, processing, ...


@app.post("/{model}")
async def inference(
    model: str, image: UploadFile = File(...), version: int = Form(None)
):
    try:
        # download model from ETODL
        download_path, stac_df = download_model(model, DOWNLOAD_PATH, version)

        # validate model metadata
        items = stac_df[stac_df["type"] == "Feature"]
        assert (
            len(items) == 1
        ), "Only one model is expected (multiple items found in STAC metadata)"
        model_props = items.properties.values[0]
        # print(model_props)
        assert (
            model_props["mlm:framework"] == "ONNX"
        ), "Only ONNX models are supported (mlm:framework)"
        model_name = model_props["mlm:name"]
        model_path = f"{download_path}/assets/{model_name}"
        # print(model_path)

        # load model
        ort_session = get_onnx_session(model_path)
        # print(ort_session.get_provider_options())

        # load image in memory as numpy array
        with rio.open(io.BytesIO(image.file.read())) as src:
            image = src.read()
        assert image.ndim == 3, "Image must have 3 dimensions (bands, height, width)"

        # pre-process and validate input
        input = model_props["mlm:input"]

        # input data type
        dtype = input["input"]["data_type"]
        image = image.astype(dtype)

        # input shape
        input_shape = input["input"]["shape"]
        ndims = len(input_shape)
        if ndims != image.ndim:
            if ndims == 4:
                image = np.expand_dims(image, axis=0).astype(np.float32)
            else:
                raise Exception("Input shape not valid", input_shape, image.ndim)
        for i, dim in enumerate(input_shape):
            if dim != -1:
                assert dim == image.shape[i], "Input dimension not valid"
        # TODO: resize if necessary

        # execute model
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: image}
        ort_outs = ort_session.run(None, ort_inputs)

        output_nodes = ort_session.get_outputs()
        output_names = [node.name for node in output_nodes]

        # return outputs
        return {
            "model": model,
            **{output: ort_outs[i].tolist() for i, output in enumerate(output_names)},
        }

    except Exception as e:
        print("ERROR")
        print(e)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
