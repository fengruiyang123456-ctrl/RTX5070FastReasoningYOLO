import os
import time
from typing import List, Tuple

import cv2
import numpy as np

from common.postprocess import postprocess_yolov8, postprocess_yolov8_resize
from common.preprocess import letterbox, to_tensor

try:
    import torch
    import torch.nn.functional as torch_f
except Exception:  # pragma: no cover
    torch = None
    torch_f = None

class TensorRTBackend:
    def __init__(
        self,
        engine_path: str,
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
    ) -> None:
        try:
            import tensorrt as trt  # noqa: F401
            import pycuda.driver as cuda  # noqa: F401
            import pycuda.autoinit  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "TensorRT backend requires tensorrt and pycuda. "
                "Generate an engine and install dependencies first."
            ) from exc

        self.engine_path = engine_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self._trt = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.tensor_names: List[str] = []
        self.input_name = None
        self.output_names: List[str] = []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            self.tensor_names.append(name)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if self.input_name is None or not self.output_names:
            raise RuntimeError("TensorRT engine bindings not found.")

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtypes = {
            name: trt.nptype(self.engine.get_tensor_dtype(name)) for name in self.output_names
        }

        self.device_buffers: dict[str, object] = {}
        self.bindings: List[int] = []
        self.host_outputs: List[np.ndarray] = []
        self._allocate_buffers()
        self._profile = os.getenv("TRT_PROFILE", "0") == "1"
        self._profile_every = int(os.getenv("TRT_PROFILE_EVERY", "30"))
        self._profile_step = 0
        self._fast_resize = os.getenv("TRT_FAST_RESIZE", "0") == "1"
        self._gpu_preproc = os.getenv("TRT_GPU_PREPROC", "0") == "1"
        if self._gpu_preproc:
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("TRT_GPU_PREPROC=1 requires torch with CUDA.")

    def warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        _ = self.infer(dummy)

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        if self._gpu_preproc:
            frame_t = torch.from_numpy(frame).to(device="cuda", non_blocking=True)
            frame_t = frame_t[..., [2, 1, 0]]
            frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)
            frame_t = frame_t.float().div_(255.0)
            frame_t = torch_f.interpolate(
                frame_t, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False
            )
            if self.input_dtype == np.float16:
                frame_t = frame_t.half()
            else:
                frame_t = frame_t.float()
            tensor = frame_t.contiguous()
            ratio = None
            pad = None
        else:
            if self._fast_resize:
                resized = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
                ratio = None
                pad = None
            else:
                resized, ratio, pad = letterbox(frame, (self.imgsz, self.imgsz))
            tensor = to_tensor(resized, half=True)
            if tensor.dtype != self.input_dtype:
                tensor = tensor.astype(self.input_dtype, copy=False)
            tensor = np.ascontiguousarray(tensor)
        t1 = time.perf_counter()

        tensor_shape = tuple(tensor.shape)
        self._ensure_binding_shape(tensor_shape)
        if self._gpu_preproc:
            src_ptr = int(tensor.data_ptr())
            nbytes = int(tensor.numel() * tensor.element_size())
            self._cuda.memcpy_dtod(self.device_buffers[self.input_name], src_ptr, nbytes)
        else:
            self._cuda.memcpy_htod(self.device_buffers[self.input_name], tensor)
        if hasattr(self.context, "execute_v3"):
            self.context.execute_v3()
        else:
            self.context.execute_v2(self.bindings)
        t2 = time.perf_counter()

        outputs = []
        for out_name in self.output_names:
            host_out = self._copy_output(out_name)
            outputs.append(host_out)

        pred = outputs[0]
        if self._gpu_preproc or self._fast_resize:
            boxes, scores, class_ids = postprocess_yolov8_resize(
                pred,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                orig_shape=frame.shape[:2],
                new_shape=(self.imgsz, self.imgsz),
            )
        else:
            boxes, scores, class_ids = postprocess_yolov8(
                pred,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                ratio=ratio,
                pad=pad,
                orig_shape=frame.shape[:2],
            )
        t3 = time.perf_counter()
        if self._profile:
            self._profile_step += 1
            if self._profile_step % self._profile_every == 0:
                pre_ms = (t1 - t0) * 1000.0
                trt_ms = (t2 - t1) * 1000.0
                post_ms = (t3 - t2) * 1000.0
                total_ms = (t3 - t0) * 1000.0
                print(
                    f"[TRT] pre={pre_ms:.3f}ms trt={trt_ms:.3f}ms "
                    f"post={post_ms:.3f}ms total={total_ms:.3f}ms"
                )
        return boxes, scores, class_ids

    def _ensure_binding_shape(self, input_shape: Tuple[int, ...]) -> None:
        current_shape = tuple(self.context.get_tensor_shape(self.input_name))
        desired_shape = tuple(input_shape)
        if any(dim < 0 for dim in current_shape):
            self.context.set_input_shape(self.input_name, desired_shape)
            self._allocate_buffers()
            return
        if current_shape != desired_shape:
            self.context.set_input_shape(self.input_name, desired_shape)
            self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        if any(dim < 0 for dim in input_shape):
            input_shape = (1, 3, self.imgsz, self.imgsz)
            self.context.set_input_shape(self.input_name, input_shape)
        input_size = int(self._trt.volume(input_shape))
        input_bytes = input_size * np.dtype(self.input_dtype).itemsize
        self.device_buffers[self.input_name] = self._cuda.mem_alloc(input_bytes)
        self.context.set_tensor_address(
            self.input_name, int(self.device_buffers[self.input_name])
        )

        self.host_outputs = []
        for out_name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(out_name))
            if any(dim < 0 for dim in out_shape):
                raise RuntimeError("Dynamic output shapes are not supported without shape bindings.")
            out_dtype = self.output_dtypes[out_name]
            out_size = int(self._trt.volume(out_shape))
            out_bytes = out_size * np.dtype(out_dtype).itemsize
            self.device_buffers[out_name] = self._cuda.mem_alloc(out_bytes)
            self.context.set_tensor_address(
                out_name, int(self.device_buffers[out_name])
            )
            self.host_outputs.append(np.empty(out_size, dtype=out_dtype))

        self.bindings = [int(self.device_buffers[name]) for name in self.tensor_names]

    def _copy_output(self, out_name: str) -> np.ndarray:
        out_pos = self.output_names.index(out_name)
        host_out = self.host_outputs[out_pos]
        self._cuda.memcpy_dtoh(host_out, self.device_buffers[out_name])
        out_shape = tuple(self.context.get_tensor_shape(out_name))
        return host_out.reshape(out_shape)
