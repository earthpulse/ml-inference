import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import Affine


def _has_georef(crs, transform) -> bool:
	"""True when input has real georeferencing (not PNG/JPG with identity transform)."""
	if crs is None or transform is None:
		return False
	return transform != Affine.identity()


def to_binary_mask(array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
	"""Convert model output to uint8 binary mask (0 background, 1 road)."""
	values = np.asarray(array, dtype=np.float32)
	if values.min() >= 0 and values.max() <= 1:
		probas = values
	else:
		probas = 1 / (1 + np.exp(-values))
	return (probas > threshold).astype(np.uint8)


def array_to_cog_bytes(
	array: np.ndarray,
	crs,
	transform,
	dtype: str = "uint8",
) -> bytes:
	"""Write a 2D array as an in-memory Cloud Optimized GeoTIFF (GDAL COG driver)."""
	if array.ndim != 2:
		raise ValueError(f"Expected 2D array, got shape {array.shape}")

	data = np.asarray(array, dtype=dtype)
	profile = {
		"driver": "COG",
		"dtype": data.dtype,
		"count": 1,
		"height": data.shape[0],
		"width": data.shape[1],
		"compress": "DEFLATE",
	}
	if _has_georef(crs, transform):
		profile["crs"] = crs
		profile["transform"] = transform
	with MemoryFile() as memfile:
		with memfile.open(**profile) as dst:
			dst.write(data, 1)
		return memfile.read()
