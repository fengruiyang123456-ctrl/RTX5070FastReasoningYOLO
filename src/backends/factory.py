from __future__ import annotations

from argparse import Namespace
from typing import Callable, Dict, Iterable

from src.backends.ort_fp16 import OrtFP16Backend
from src.backends.torch_fp16 import TorchFP16Backend
from src.backends.torch_fp32 import TorchFP32Backend
from src.backends.torch_fp32_modify import TorchFP32ModifyBackend
from src.backends.trt_fp16 import TensorRTBackend
from src.backends.trt_fp32 import TensorRTFP32Backend


BackendFactory = Callable[[Namespace], object]


def _build_torch_fp32(args: Namespace) -> TorchFP32Backend:
    return TorchFP32Backend(args.weights, args.device, args.conf, args.iou, args.imgsz)


def _build_torch_fp32_custom_preproc(args: Namespace) -> TorchFP32ModifyBackend:
    return TorchFP32ModifyBackend(args.weights, args.device, args.conf, args.iou, args.imgsz)


def _build_torch_fp16(args: Namespace) -> TorchFP16Backend:
    return TorchFP16Backend(args.weights, args.device, args.conf, args.iou, args.imgsz)


def _build_ort_fp16(args: Namespace) -> OrtFP16Backend:
    return OrtFP16Backend(args.onnx, args.conf, args.iou, args.imgsz)


def _build_trt_fp16(args: Namespace) -> TensorRTBackend:
    return TensorRTBackend(args.trt_engine, args.conf, args.iou, args.imgsz)


def _build_trt_fp32(args: Namespace) -> TensorRTFP32Backend:
    return TensorRTFP32Backend(args.trt_engine, args.conf, args.iou, args.imgsz)


BACKEND_FACTORIES: Dict[str, BackendFactory] = {
    "torch_fp32": _build_torch_fp32,
    "torch_fp32_custom_preproc": _build_torch_fp32_custom_preproc,
    "torch_fp32_modify": _build_torch_fp32_custom_preproc,
    "torch_fp16": _build_torch_fp16,
    "ort_fp16": _build_ort_fp16,
    "trt_fp16": _build_trt_fp16,
    "trt_fp32": _build_trt_fp32,
}


def available_backends() -> Iterable[str]:
    return BACKEND_FACTORIES.keys()


def build_backend(name: str, args: Namespace) -> object:
    try:
        factory = BACKEND_FACTORIES[name]
    except KeyError as exc:
        supported = ", ".join(BACKEND_FACTORIES.keys())
        raise ValueError(f"Unknown backend '{name}'. Supported backends: {supported}") from exc
    return factory(args)
