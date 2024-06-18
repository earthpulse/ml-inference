import onnxruntime as ort


def get_onnx_session(model):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model, providers=providers)
    except Exception as e:
        raise RuntimeError(f"Error loading ONNX model: {str(e)}")
    return session
