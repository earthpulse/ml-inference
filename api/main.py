from fastapi import FastAPI, File, UploadFile, status
from fastapi.exceptions import HTTPException
import numpy as np
from rasterio.io import MemoryFile
from eotdl.models import download_model
import onnxruntime
from glob import glob

app = FastAPI()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# this is an example, only works with https://www.eotdl.com/models/EuroSAT-RGB-BiDS23
@app.post("/{model}")
async def inference(model: str, img_file: UploadFile = File(...)):
    try:
        # Read the data from the file in memory
        contents = await img_file.read()

        # Use MemoryFile from rasterio to open the file directly from memory
        with MemoryFile(contents) as memfile:
            with memfile.open() as dataset:
                # Convert the file content to a numpy array
                img_array = dataset.read()

        download_path = download_model(
            model, force=True
        )  # it should point to STAC of the model

        # TODO: Check the specifications of the array
        # TODO: Check the STAC of the model to check the `input` requirements
        # TODO: Compare the array with the `input` requirements
        # TODO: Do the preprocessing of the image if necessary

        # TODO: It seems that the model was not exported correctly to accept a random batch size (it only works with a batch size of 25)
        batch = (
            np.array(25 * [img_array.astype(np.float32)]) / 255
        )  # this should be defined in metadata

        # TODO: Check the STAC of the model to check the `runtime` requirements and apply
        file_path = glob(f"{download_path}/*.onnx")[0]

        lables = [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ]  # this should be defined in metadata

        # Execute model
        ort_session = onnxruntime.InferenceSession(file_path)
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: batch}
        ort_outs = ort_session.run(None, ort_inputs)
        probas = softmax(ort_outs[0][0])

        # TODO: Check the STAC of the model to check the `output` requirements
        # TODO: Check the specifications of the outputs
        # TODO: Compare the outputs with the `output` requirements
        # TODO: Do the postprocessing of the outputs if necessary

        return {
            "message": f"{model} model executed",
            "probas": probas.tolist(),
            "class": lables[np.argmax(probas)],
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
