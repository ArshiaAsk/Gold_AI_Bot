from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    import onnxruntime
except Exception:  # pragma: no cover - optional dependency
    onnxruntime = None


class OptimizedPredictor:
    """ONNX runtime predictor with in-process LRU caching."""

    def __init__(self, onnx_model_path: str, input_name: str | None = None):
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for OptimizedPredictor. Install it via requirements.")

        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            onnx_model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = input_name or self.session.get_inputs()[0].name

    @lru_cache(maxsize=1024)
    def _predict_cached(self, shape: tuple[int, ...], buffer: bytes) -> np.ndarray:
        features = np.frombuffer(buffer, dtype=np.float32).reshape(shape)
        return self.session.run(None, {self.input_name: features})[0]

    def predict(self, features) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        arr = np.ascontiguousarray(arr)
        return self._predict_cached(arr.shape, arr.tobytes())


# Backward compatibility for old typo-based import.
OpimizedPredictor = OptimizedPredictor
