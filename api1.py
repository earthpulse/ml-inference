from fastapi import FastAPI, File, UploadFile
import numpy as np
from rasterio.io import MemoryFile
from eotdl.models import download_model
import os
import onnxruntime
import glob

app = FastAPI()

@app.post("/")
async def inference(model_name: str, img_file: UploadFile = File(...)):
    # Read the data from the file in memory
    contents = await img_file.read()
    
    # Use MemoryFile from rasterio to open the file directly from memory
    with MemoryFile(contents) as memfile:
        with memfile.open() as dataset:
            # Convert the file content to a numpy array
            img_array = dataset.read()
    
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "models")
    model_path = os.path.join(cache_dir, model_name)

    # Check if the model doesn't exist in the cache directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        download_model(model_name, path = model_path)
    
    # Set the base directory where to search for the .onnx file
    base_path = os.path.expanduser(f'~/.cache/models/{model_name}')

    # Use glob to find directories containing the 'model.onnx' file
    # The pattern '**' matches any number of directories including nested ones,
    # and 'model.onnx' is the filename we are looking for
    pattern = os.path.join(base_path, '**', 'v1', 'model.onnx')

    # Perform the search, 'recursive=True' allows '**' to match any directory depth
    onnx_path = glob.glob(pattern, recursive=True)
    print(img_array.dtype)

    # Check if we found any models
    if onnx_path:
        # If there are multiple models, decide how to select the one you want,
        # for example, taking the first one found:
        onnx_path = onnx_path[0]

    # TODO: It seems that the model was not exported correctly to accept a random batch size (it only works with a batch size of 25)
    batch = np.array(25*[img_array.astype(np.float32)])
    print(batch.shape)
 
    # Execute model
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: batch}
    ort_outs = ort_session.run(None, ort_inputs)

    print('OUTPUT', ort_outs[0][0])
    
    return {"message": f"{model_name} model executed in local", "array_shape": img_array.shape, "model_path": model_path, "model_output": ort_outs[0].shape}
