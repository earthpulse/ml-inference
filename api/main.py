from fastapi import FastAPI, File, UploadFile, status
from fastapi.exceptions import HTTPException
import numpy as np
from rasterio.io import MemoryFile
from eotdl.models import download_model
import os
import onnxruntime

app = FastAPI()

@app.post("/{model}")
async def inference(model: str, img_file: UploadFile = File(...)):
    
    # Read the data from the file in memory
    contents = await img_file.read()

    # Use MemoryFile from rasterio to open the file directly from memory
    with MemoryFile(contents) as memfile:
        with memfile.open() as dataset:
            # Convert the file content to a numpy array
            img_array = dataset.read()
    
    try:
        download_path = download_model(model, force=True) # it should point to STAC of the model
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    
    # TODO: Check the specifications of the array 
    # TODO: Check the STAC of the model to check the `input` requirements
    # TODO: Compare the array with the `input` requirements        
    # TODO: Do the preprocessing of the image if necessary

    # TODO: It seems that the model was not exported correctly to accept a random batch size (it only works with a batch size of 25)
    batch = np.array(25*[img_array.astype(np.float32)])

    # TODO: Check the STAC of the model to check the `runtime` requirements and apply
    file_path = os.path.join(download_path, 'model.onnx')
    
    # Execute model
    ort_session = onnxruntime.InferenceSession(file_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: batch}
    ort_outs = ort_session.run(None, ort_inputs)

    # TODO: Check the STAC of the model to check the `output` requirements
    # TODO: Check the specifications of the outputs
    # TODO: Compare the outputs with the `output` requirements  
    # TODO: Do the postprocessing of the outputs if necessary

    return {"message": f"{model} model executed", "outputs": f"{ort_outs[0][0]}"}