import os
import requests
import geopandas as gpd
import numpy as np

from .dataframe import STACDataFrame

EOTDL_API_URL = "https://api.eotdl.com/"
EOTDL_API_KEY = os.getenv("EOTDL_API_KEY")


def download_model(model_name, dst_path, version, force=False, download=True):
    # check if model already downloaded
    version = 1 if version is None else version
    download_path = dst_path + "/" + model_name + "/v" + str(version)
    if os.path.exists(download_path) and not force:
        df = STACDataFrame.from_stac_file(download_path + f"/{model_name}/catalog.json")
        return download_path, df
    # check model exists
    model, error = retrieve_model(model_name)
    if error:
        raise Exception(error)
    if model["quality"] < 2:
        raise Exception("Only Q2+ models are supported")
    # check version exist
    assert version in [
        v["version_id"] for v in model["versions"]
    ], f"Version {version} not found"
    # download model files
    gdf, error = retrieve_model_stac(model["id"], version)
    if error:
        raise Exception(error)
    df = STACDataFrame(gdf)
    if not download:
        return download_path, df
    os.makedirs(download_path, exist_ok=True)
    df.to_stac(download_path)
    df = df.dropna(subset=["assets"])
    for row in df.iterrows():
        for k, v in row[1]["assets"].items():
            href = v["href"]
            _, filename = href.split("/download/")
            download_file_url(href, filename, f"{download_path}/assets")
    return download_path, df


def format_response(response):
    if response.status_code == 200:
        return response.json(), None
    return None, response.json()["detail"]


def retrieve_model(name):
    response = requests.get(EOTDL_API_URL + "models?name=" + name)
    return format_response(response)


def retrieve_model_stac(model_id, version):
    url = f"{EOTDL_API_URL}models/{model_id}/download"
    headers = {"X-API-Key": EOTDL_API_KEY}
    response = requests.get(url, headers=headers)
    data, error = format_response(response)
    if data:
        return gpd.GeoDataFrame.from_features(response.json()["features"]), None
    return data, error


def download_file_url(url, filename, path):
    headers = {"X-API-Key": EOTDL_API_KEY}
    path = f"{path}/{filename}"
    for i in range(1, len(path.split("/")) - 1):
        os.makedirs("/".join(path.split("/")[: i + 1]), exist_ok=True)
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024 * 1024 * 10
        with open(path, "wb") as f:
            for chunk in r.iter_content(block_size):
                if chunk:
                    f.write(chunk)
        return path


def format_outputs(ort_outputs, output_names, input, model_props):
    if model_props["mlm:output"]["tasks"] == ["classification"]:
        return {
            output: ort_outputs[i].tolist() for i, output in enumerate(output_names)
        }
    elif model_props["mlm:output"]["tasks"] == ["segmentation"]:
        return {output: ort_outputs[i] for i, output in enumerate(output_names)}
        # return {
        #     output: reshape(
        #         ort_outputs[i],
        #         input,
        #         model_props["mlm:output"]["result"]["shape"],
        #         model_props["mlm:input"]["input"]["shape"],
        #     )
        #     for i, output in enumerate(output_names)
        # }
    else:
        raise Exception("Output task not supported", model_props["mlm:output"]["tasks"])


def reshape(x, input, output_shape, input_shape):
    _output_shape = output_shape
    _output_shape[0] = input.shape[0]  # batch size (maybe not always)
    for i in range(1, len(output_shape)):
        dim = output_shape[i]
        if dim == -1:
            j = (
                i + 1 if len(input_shape) > len(output_shape) else i
            )  # skip channel dimension if input has one more dimension (assuming channles first which may not be always the case)
            if input_shape[j] == -1:
                _output_shape[i] = input.shape[j]  # take first input shape as reference
            else:
                _output_shape[i] = input_shape[j]
        else:
            _output_shape[i] = dim
    # it may require a resize as well...
    return x.reshape(_output_shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
