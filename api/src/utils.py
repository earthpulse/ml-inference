import os
import requests
import geopandas as gpd
import numpy as np
import io

EOTDL_API_URL = "https://api.eotdl.com/"
EOTDL_API_KEY = os.getenv("EOTDL_API_KEY")


def format_response(response):
    if response.status_code == 200:
        return response.json(), None
    return None, response.json()["detail"]


def retrieve_model(name):
    response = requests.get(EOTDL_API_URL + "models?name=" + name)
    return format_response(response)


def retrieve_model_catalog(model_id, version):
    url = f"{EOTDL_API_URL}models/{model_id}/stage/catalog.v{version}.parquet"
    headers = {"X-API-Key": EOTDL_API_KEY}
    response = requests.get(url, headers=headers)
    data, error = format_response(response)
    if error:
        return None, error
    presigned_url = data["presigned_url"]
    # get parquet file from presigned url and load as geopandas dataframe
    response = requests.get(presigned_url)
    if response.status_code != 200:
        return None, f"Failed to download parquet file: {response.status_code}"
    # Load parquet from memory buffer
    parquet_buffer = io.BytesIO(response.content)
    gdf = gpd.read_parquet(parquet_buffer)
    return gdf, None


def download_file_url(url, path):
    file_name = url.split("/stage/")[-1] 
    file_path = f"{path}/{file_name}"
    for i in range(1, len(file_path.split("/")) - 1):
        os.makedirs("/".join(file_path.split("/")[: i + 1]), exist_ok=True)
    headers = {"X-API-Key": EOTDL_API_KEY}
    response = requests.get(url, headers=headers)
    data, error = format_response(response)
    if error:
        return None, error
    presigned_url = data["presigned_url"]
    response = requests.get(presigned_url)
    response.raise_for_status()  # This will raise an HTTPError for 4XX and 5XX status codes
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return file_path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def reshape(x, input, output_shape, input_shape):
#     _output_shape = output_shape
#     _output_shape[0] = input.shape[0]  # batch size (maybe not always)
#     for i in range(1, len(output_shape)):
#         dim = output_shape[i]
#         if dim == -1:
#             j = (
#                 i + 1 if len(input_shape) > len(output_shape) else i
#             )  # skip channel dimension if input has one more dimension (assuming channles first which may not be always the case)
#             if input_shape[j] == -1:
#                 _output_shape[i] = input.shape[j]  # take first input shape as reference
#             else:
#                 _output_shape[i] = input_shape[j]
#         else:
#             _output_shape[i] = dim
#     # it may require a resize as well...
#     return x.reshape(_output_shape)