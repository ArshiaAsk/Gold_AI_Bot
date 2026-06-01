from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def export_keras_to_onnx(
    keras_model_path: str,
    onnx_output_path: str,
    *,
    input_shape: Tuple[int, int, int] = (1, 30, 15),
    opset: int = 13,
) -> str:
    """
    Convert a saved Keras model to ONNX for OptimizedPredictor.

    Returns the output path on success.
    """
    import keras
    import tf2onnx
    import tensorflow as tf

    keras_path = Path(keras_model_path)
    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_model_path}")

    output_path = Path(onnx_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(str(keras_path), compile=False)
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
    output_path.write_bytes(model_proto.SerializeToString())
    logger.info("Exported ONNX model to %s", output_path)
    return str(output_path)


def ensure_onnx_artifact(
    keras_model_path: str,
    onnx_output_path: Optional[str] = None,
) -> Optional[str]:
    """Export ONNX only when missing or older than the Keras artifact."""
    keras_path = Path(keras_model_path)
    if not keras_path.exists():
        return None

    onnx_path = Path(onnx_output_path or keras_path.with_suffix(".onnx"))
    if onnx_path.exists() and onnx_path.stat().st_mtime >= keras_path.stat().st_mtime:
        return str(onnx_path)

    try:
        return export_keras_to_onnx(str(keras_path), str(onnx_path))
    except Exception as exc:
        logger.warning("ONNX export skipped: %s", exc)
        return None
