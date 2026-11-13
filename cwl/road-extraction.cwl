cwlVersion: v1.2
class: CommandLineTool

label: AI Road Extraction
doc: >
  Road mask for a 3-band RGB GeoTIFF already staged by the platform.
  Process id on the DPH must be "road-extraction" (the hub loads
  {processID}.cwl from its cwl_dir).
  EarthPulse does not download the STAC catalogue and does not publish
  results. MEEO stage-in fills `staged`; MEEO stage-out publishes
  `stac_catalog`.

baseCommand: python3
arguments: [road-extraction_processor.py]

requirements:
  NetworkAccess:
    networkAccess: true
  InitialWorkDirRequirement:
    listing:
      - class: File
        location: road-extraction_processor.py

inputs:
  staged:
    type: Directory
    label: Staged STAC item
    doc: >
      PENDING MEEO (stage-in): Directory produced from a GDA STAC item
      link. Must contain catalog.json and the RGB GeoTIFF as a local
      file. Do not pass image_url. Optional env ROAD_ASSET_KEY selects
      the asset key; empty means the first local image/tiff.
    inputBinding:
      position: 1
  model:
    type: string
    default: MassachusettsRoadsS2Model
    inputBinding:
      position: 2
  url:
    type: string
    label: Inference API base URL
    doc: >
      PENDING MEEO: not a user input. The hub injects this from the
      process link rel="service" (base URL, without the model name).
      The process POSTs the GeoTIFF to {url}/{model}, file field "image".
      If that API requires a key, set API_KEY in the DPH environment
      (sent as X-API-Key).
    inputBinding:
      position: 3

outputs:
  stac_catalog:
    type: Directory
    outputBinding:
      glob: stac_catalog
    label: Results STAC catalog
    doc: >
      PENDING MEEO (stage-out): local catalog ready to publish.
      Layout is catalog.json, one item, asset key "road-mask"
      (road-mask.tif). Confirm that key, the collection id and the
      license before wiring stage-out.
  result:
    type: File
    outputBinding:
      glob: output.json
    doc: >
      PENDING MEEO: the DPH runner still json-loads the output named
      "result" (lib/cwl_runner.py read_cwl_file) and then deletes the
      file. This JSON is only a short pointer (item id, asset key,
      byte size) so that runner does not break. Stage-out must use
      stac_catalog. Drop this output once the hub accepts a Directory.

stdout: output.json
