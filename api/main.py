from fastapi import FastAPI, File, UploadFile, status, Form
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import rasterio as rio
import io
import os
import numpy as np
from PIL import Image

from src.utils import download_model, format_outputs, sigmoid
from src.onnx import get_onnx_session

__version__ = "2024.08.19"

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

        # normalization
        image = (
            image / 255.0
        )  # this should be defined in the model metadata (this is only for RGB will not work with satellite images)

        # TODO: resize if necessary

        # execute model
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: image}
        ort_outs = ort_session.run(None, ort_inputs)

        output_nodes = ort_session.get_outputs()
        output_names = [node.name for node in output_nodes]

        outputs = format_outputs(ort_outs, output_names, image, model_props)

        # return outputs
        if model_props["mlm:output"]["tasks"] == ["classification"]:
            return {
                "model": model,
                **outputs,
            }
        elif model_props["mlm:output"]["tasks"] == [
            "segmentation"
        ]:  # only returns first output as image
            # return mask
            batch = outputs[output_names[0]]
            image = batch[0]
            image = sigmoid(image) > 0.5  # this should be defined in the model metadata
            if image.ndim == 3:  # get first band
                image = image[0]
            img = Image.fromarray(image, mode="L")  # only returns binary mask
            buf = io.BytesIO()
            img.save(buf, "tiff")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/tiff")
        else:
            raise Exception(
                "Output task not supported", model_props["mlm:output"]["tasks"]
            )

    except Exception as e:
        print("ERROR", "inference")
        print(e)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@app.get("/{model}")
def retrieve_model_metadata(model: str, version: int = None):
    try:
        _, stac_df = download_model(model, DOWNLOAD_PATH, version, download=False)
        return stac_df.to_json()
    except Exception as e:
        print("ERROR", "retrieve_model_metadata")
        print(e)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
