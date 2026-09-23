cwlVersion: v1.2
class: CommandLineTool

label: AI Road Extraction
doc: >
  Road mask for a 3-band RGB GeoTIFF.
  Process id on the DPH must be "road-extraction" (the hub loads
  {processID}.cwl from its cwl_dir).
  `url` is not a user input: the hub injects it from the process
  link rel="service", the inference API MEEO already operates.
  PENDING TBD: that service URL, and how the COG is returned.
  The hub json-loads `result` and drops the file, so mask.tif is not
  in the job until output transmission is agreed.

baseCommand: python3
arguments: [road-extraction_processor.py]

requirements:
  InitialWorkDirRequirement:
    listing:
      - class: File
        location: road-extraction_processor.py

inputs:
  image_url:
    type: string
    doc: URL of a 3-band RGB GeoTIFF. Execution accepts JSON only.
    inputBinding:
      position: 1
  model:
    type: string
    default: MassachusettsRoadsS2Model
    inputBinding:
      position: 2
  url:
    type: string
    inputBinding:
      position: 3

outputs:
  result:
    type: File
    outputBinding:
      glob: output.json
  mask:
    type: File?
    outputBinding:
      glob: mask.tif

stdout: output.json
