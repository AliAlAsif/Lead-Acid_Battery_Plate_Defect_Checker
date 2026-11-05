"""
Export the trained 4-class plate classifier (plate_classifier.pt) to ONNX,
then quantize it for fast in-browser inference.

Outputs
-------
- model_export/plate_classifier.onnx   full float32 (reference / regenerable)
- docs/model.onnx                      int8 quantized (what the website loads)

Quantization uses STATIC QDQ (QuantizeLinear/DequantizeLinear) format, which
produces QLinearConv / QLinearMatMul kernels. Those are supported by the
ONNX Runtime WebAssembly backend used in the browser. (Dynamic quantization
emits ConvInteger/MatMulInteger, which the WASM backend does not implement.)

Run from the project root:

    .venv\\Scripts\\python model_export\\export_onnx.py

Preprocessing must match the notebook exactly:
    Resize((224, 224)) -> ToTensor [0,1] -> Normalize(mean, std)
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "notebooks", "plate_classifier.pt")
FLOAT32_ONNX = os.path.join(ROOT, "model_export", "plate_classifier.onnx")
INT8_ONNX = os.path.join(ROOT, "docs", "model.onnx")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

CLASSES = ["good", "surface_error", "crack", "bend"]


def build_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def export_float32():
    ckpt = torch.load(CKPT, map_location="cpu")
    assert ckpt["classes"] == CLASSES, f"unexpected classes: {ckpt['classes']}"
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    os.makedirs(os.path.dirname(FLOAT32_ONNX), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        FLOAT32_ONNX,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print("Exported float32 ONNX ->", FLOAT32_ONNX)


def preprocess(path):
    img = Image.open(path).convert("RGB")
    x = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])(img)
    return x.unsqueeze(0).numpy()


class CalibrationReader:
    """Feeds preprocessed images to the static quantizer for activation ranges."""

    def __init__(self, arrays):
        self._arrs = list(arrays)
        self._i = 0

    def get_next(self):
        if self._i >= len(self._arrs):
            return None
        item = {"input": self._arrs[self._i]}
        self._i += 1
        return item


def quantize_int8():
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    # Calibrate on every available image so activation ranges are representative.
    calib = []
    for cls in CLASSES:
        for p in sorted(glob.glob(os.path.join(ROOT, "data", cls, "*.*"))):
            calib.append(preprocess(p))
    print(f"Calibrating on {len(calib)} images.")

    os.makedirs(os.path.dirname(INT8_ONNX), exist_ok=True)
    quantize_static(
        model_input=FLOAT32_ONNX,
        model_output=INT8_ONNX,
        calibration_data_reader=CalibrationReader(calib),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print("Exported int8 ONNX ->", INT8_ONNX)


def verify():
    """Check the quantized model agrees with the float32 model on sample images."""
    import onnxruntime as ort

    sess_f32 = ort.InferenceSession(FLOAT32_ONNX, providers=["CPUExecutionProvider"])
    sess_i8 = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])

    samples = []
    for cls in CLASSES:
        samples += sorted(glob.glob(os.path.join(ROOT, "data", cls, "*.*")))[:4]

    mismatches = 0
    max_gap = 0.0
    for p in samples:
        x = preprocess(p)
        logits_f32 = sess_f32.run(None, {"input": x})[0][0]
        logits_i8 = sess_i8.run(None, {"input": x})[0][0]
        p_f32 = torch.softmax(torch.tensor(logits_f32), dim=0).numpy()
        p_i8 = torch.softmax(torch.tensor(logits_i8), dim=0).numpy()
        a_f32, a_i8 = int(p_f32.argmax()), int(p_i8.argmax())
        gap = float(np.abs(p_f32 - p_i8).max())
        max_gap = max(max_gap, gap)
        if a_f32 != a_i8:
            mismatches += 1
        print(f"{'OK ' if a_f32 == a_i8 else 'MISMATCH'} {os.path.basename(p):30s} "
              f"f32={CLASSES[a_f32]} i8={CLASSES[a_i8]}  max|dP|={gap:.4f}")

    print(f"\n{len(samples)} images checked, {mismatches} argmax mismatches, "
          f"max probability gap {max_gap:.4f}")


if __name__ == "__main__":
    export_float32()
    quantize_int8()
    verify()
