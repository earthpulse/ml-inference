"""DPH entrypoint for AI Road Extraction.

argv: staged_dir model url

staged_dir is the platform stage-in Directory (catalog.json + local assets).
This process does not download the STAC catalogue and does not upload results.

PENDING MEEO:
- stage-in must pass that Directory from a GDA STAC item link
- set ROAD_ASSET_KEY if the RGB asset key is not the first local image/tiff
- url comes from the process link rel="service" (POST /{model}, field "image")
- if the API needs a key, set API_KEY in the DPH environment
- stage-out publishes the Directory stac_catalog (asset key "road-mask")
- the hub still json-loads output "result"; see road-extraction.cwl
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

TIFF = (".tif", ".tiff")


def load_items(staged: Path):
    found = []
    for path in staged.rglob("*.json"):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        assets = doc.get("assets") if isinstance(doc, dict) else None
        if doc.get("type") == "Feature" and isinstance(assets, dict):
            found.append((path.parent, doc))
    return found


def find_raster(staged: Path):
    # PENDING MEEO: asset key of the 3-band RGB GeoTIFF. Empty = first local image/tiff.
    key = os.environ.get("ROAD_ASSET_KEY", "")
    remote = False
    for base, doc in load_items(staged):
        names = [key] if key else list(doc["assets"])
        for name in names:
            asset = doc["assets"].get(name) or {}
            href = asset.get("href") or ""
            media = asset.get("type") or ""
            if not (key or media.startswith("image/tiff") or href.lower().endswith(TIFF)):
                continue
            if "://" in href:
                remote = True  # PENDING MEEO: stage-in must materialise assets; we do not fetch hrefs.
                continue
            path = (base / href).resolve()
            if path.is_file():
                return path, doc
    if not key:
        for path in staged.rglob("*"):
            if path.is_file() and path.suffix.lower() in TIFF:
                return path, {}
    hint = "remote href left by stage-in" if remote else "no local GeoTIFF"
    raise SystemExit(f"PENDING MEEO: {hint} in {staged} (ROAD_ASSET_KEY={key!r})")


def write_catalog(src: dict, mask: bytes, media_type: str) -> str:
    # PENDING MEEO: confirm collection id, license, and asset key "road-mask".
    item_id = f"{src.get('id') or 'input'}-roads"
    props = src.get("properties") or {}
    # PENDING MEEO: datetime is copied from the source item when present.
    when = props.get("datetime") or props.get("start_datetime") or "1970-01-01T00:00:00Z"
    item_dir = Path("stac_catalog") / item_id
    item_dir.mkdir(parents=True)
    (item_dir / "road-mask.tif").write_bytes(mask)
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": src.get("geometry"),
        "bbox": src.get("bbox"),
        "properties": {"datetime": when},
        "assets": {
            "road-mask": {
                "href": "road-mask.tif",
                "type": media_type or "image/tiff; application=geotiff",
                "roles": ["data"],
            }
        },
        "links": [{"rel": "root", "href": "../catalog.json", "type": "application/json"}],
    }
    (item_dir / f"{item_id}.json").write_text(json.dumps(item))
    catalog = {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": "road-extraction",
        "description": "AI Road Extraction mask",
        "links": [
            {"rel": "root", "href": "./catalog.json", "type": "application/json"},
            {"rel": "item", "href": f"./{item_id}/{item_id}.json", "type": "application/json"},
        ],
    }
    (Path("stac_catalog") / "catalog.json").write_text(json.dumps(catalog))
    return item_id


def main():
    staged, model, service_url = sys.argv[1:4]
    raster, src = find_raster(Path(staged))
    boundary = "----road-extraction"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{raster.name}"\r\n'
        "Content-Type: image/tiff\r\n\r\n"
    ).encode() + raster.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
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
        media_type = (response.headers.get("Content-Type") or "").split(";")[0]
    if "json" in media_type or payload[:1] in (b"{", b"["):
        # PENDING MEEO: the inference API must return the mask GeoTIFF, not JSON.
        raise SystemExit("PENDING MEEO: inference API returned JSON, expected a GeoTIFF mask")
    item_id = write_catalog(src, payload, media_type)
    print(json.dumps({
        "stac_catalog": "stac_catalog",
        "item": f"{item_id}/{item_id}.json",
        "asset": "road-mask",
        "bytes": len(payload),
    }))


if __name__ == "__main__":
    main()
