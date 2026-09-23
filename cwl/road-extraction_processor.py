"""DPH entrypoint for AI Road Extraction. Same shape as flood-risk-fixed-value.

argv: image_url model url
`url` is injected by the hub from the process link rel="service".

PENDING TBD:
- service URL of the inference API MEEO operates (POST /{model}, file field "image")
- if that API requires X-API-Key, set API_KEY in the DPH environment
- the hub only persists output.json; mask.tif is not returned to the client
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

image_url, model, service_url = sys.argv[1:4]
urllib.request.urlretrieve(image_url, "input.tif")

boundary = "----road-extraction"
body = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="image"; filename="input.tif"\r\n'
    "Content-Type: image/tiff\r\n\r\n"
).encode() + Path("input.tif").read_bytes() + f"\r\n--{boundary}--\r\n".encode()

request = urllib.request.Request(
    f"{service_url.rstrip('/')}/{model}",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
if os.environ.get("API_KEY"):
    request.add_header("X-API-Key", os.environ["API_KEY"])

with urllib.request.urlopen(request, timeout=3600) as response:
    payload = response.read()
    media_type = response.headers.get("Content-Type", "")

if "json" in media_type:
    result = json.loads(payload)
else:
    Path("mask.tif").write_bytes(payload)
    result = {"mediaType": media_type.split(";")[0], "filename": "mask.tif", "bytes": len(payload)}

print(json.dumps(result))
