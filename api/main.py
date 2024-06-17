from fastapi import FastAPI, File, UploadFile, status, Form
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import rasterio as rio
import onnxruntime
from glob import glob
import json
import io
import os
import numpy as np

from src.utils import download_model


app = FastAPI(
    title="ml-inference",
    version="2024.06.17",
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
    return {"message": "Hello from the ml-inference API!"}


DOWNLOAD_PATH = os.getenv("EOTDL_DOWNLOAD_PATH", "/tmp")


@app.post("/{model}")
async def inference(
    model: str, image: UploadFile = File(...), version: int = Form(None)
):
    # download model from ETODL
    download_path, stac_df = download_model(model, DOWNLOAD_PATH, version)

    # validate model metadata
    items = stac_df[stac_df["type"] == "Feature"]
    assert (
        len(items) == 1
    ), "Only one model is expected (multiple items found in STAC metadata)"
    model_props = items.properties.values[0]
    print(model_props)
    assert (
        model_props["mlm:framework"] == "ONNX"
    ), "Only ONNX models are supported (mlm:framework)"
    model_name = model_props["mlm:name"]
    model_path = f"{download_path}/assets/{model_name}"
    print(model_path)

    # load model
    ort_session = onnxruntime.InferenceSession(model_path)
    # input_name = ort_session.get_inputs()[0].name
    # ort_inputs = {input_name: image}
    # ort_outs = ort_session.run(None, ort_inputs)

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

    return {"message": f"{model} model executed"}


#     try:
#         # Read the data from the file in memory
#         contents = await img_file.read()

#         # Use MemoryFile from rasterio to open the file directly from memory
#         with MemoryFile(contents) as memfile:
#             with memfile.open() as dataset:
#                 # Convert the file content to a numpy array
#                 img_array = dataset.read()

#         download_path = "tmp/"

#         # check if *.onnx and *.json files are in the path
#         if not glob(f"{download_path}/*.onnx") or not glob(f"{download_path}/*.json"):
#             download_path = download_model(
#                 model, force=True, path=download_path
#             )  # it should point to STAC of the model

#         stac_path = glob(f"{download_path}/*.json")[0]
#         with open(stac_path) as file:
#             string = file.read()
#             stac = json.loads(string)

#         # Check the specifications of the array
#         img_shape = list(img_array.shape)
#         # TODO: check dim_order
#         img_dtype = img_array.dtype
#         # Check the STAC of the model to check the `input` requirements
#         input = stac["properties"]["mlm:input"][0]
#         input_shape = input["input_array"]["shape"]
#         input_dims = input["input_array"]["dim_order"]
#         input_dtype = input["input_array"]["data_type"]
#         # Compare the array with the `input` requirements
#         # & Do the preprocessing of the image if necessary
#         batch_size = input_shape[0]
#         channels = input_shape[1]

#         if len(img_shape) != len(input_shape):
#             if len(img_shape) == 3:
#                 # Batch size is missing
#                 img_array = np.array(batch_size * [img_array])
#                 # TODO: It seems that the model was not exported correctly to accept a random batch size (it only works with a batch size of 25)

#             elif len(img_shape) == 2:
#                 # Batch size and channel are missing
#                 img_array = batch_size * [channels * [img_array]]

#         if str(img_dtype) != input_dtype:
#             img_array = img_array.astype(np.dtype(input_dtype))

#         if input["norm_by_channel"]:
#             img_array = img_array / input["norm_with_clip_values"][0]

#         # Check the STAC of the model to check the `runtime` requirements and apply
#         runtime = stac["properties"]["mlm:runtime"][0]

#         model_path = glob(f"{download_path}/*.onnx")[0]

#         # Execute model
#         ort_session = onnxruntime.InferenceSession(model_path)
#         input_name = ort_session.get_inputs()[0].name
#         ort_inputs = {input_name: img_array}
#         ort_outs = ort_session.run(None, ort_inputs)

#         # Check the specifications of the outputs
#         # Check the STAC of the model to check the `output` requirements
#         # Compare the outputs with the `output` requirements
#         # & Do the postprocessing of the outputs if necessary
#         output = stac["properties"]["mlm:output"][0]

#         labels = []
#         for clas in output["classification_classes"]:
#             labels.append(clas["name"])

#         if output["post_processing_function"]:
#             func_name = output[
#                 "post_processing_function"
#             ]  # TODO: from str to call a function
#             probas = globals()[func_name](
#                 ort_outs[0][0]
#             )  # this should be defined in metadata

#         return {
#             "message": f"{model} model executed",
#             "probas": probas.tolist(),
#             "class": labels[np.argmax(probas)],
#         }
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
