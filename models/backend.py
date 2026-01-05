"""
Model Backend - Unified interface for PyTorch, ONNX, and TensorRT models.
No dependencies on ultralytics internals.
"""

import ast
import json
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class ModelBackend(nn.Module):
    """
    Unified backend for loading and running inference on PyTorch, ONNX, and TensorRT models.

    Args:
        weights: Path to model weights (.pt, .onnx, or .engine)
        device: Device to run inference on ('cuda', 'cuda:0', 'cpu')
        fp16: Use half precision (FP16) inference
        fuse: Fuse Conv2d+BatchNorm layers for PyTorch models
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        fp16: bool = False,
        fuse: bool = True,
    ):
        super().__init__()

        # Parse device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Detect model type
        w = Path(weights)
        suffix = w.suffix.lower()
        self.pt = suffix == ".pt"
        self.onnx = suffix == ".onnx"
        self.engine = suffix == ".engine"

        # Check CUDA availability
        cuda = torch.cuda.is_available() and device.type != "cpu"
        if not cuda:
            device = torch.device("cpu")
            self.device = device

        # TensorRT requires CUDA
        if self.engine and not cuda:
            raise RuntimeError("TensorRT requires CUDA but no GPU available")

        # Default metadata
        self._names = {}
        self._stride = 32
        self._task = "detect"
        self._imgsz = (640, 640)
        self._nc = 80
        self._kpt_shape = None
        self._fp16 = fp16
        self._dynamic = False

        # Model placeholders
        self.model = None
        self.session = None
        self.context = None
        self.bindings = None
        self.binding_addrs = None
        self.output_names = []
        self.io = None

        # Load model based on type
        if self.pt:
            self._load_pytorch(weights, fuse)
        elif self.onnx:
            self._load_onnx(weights, cuda)
        elif self.engine:
            self._load_tensorrt(weights)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        # Warmup
        self.warmup()

    def _load_pytorch(self, weights: str, fuse: bool = True):
        """Load PyTorch model from checkpoint."""
        # Load checkpoint
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)

        # Extract model (prefer EMA weights)
        model = ckpt.get("ema") or ckpt.get("model")
        if model is None:
            raise ValueError(f"No model found in checkpoint: {weights}")

        # Convert to float and eval mode
        model = model.float().eval()

        # Fuse layers for inference optimization
        if fuse and hasattr(model, "fuse"):
            model = model.fuse()

        # Move to device
        model = model.to(self.device)

        # Set inplace operations
        for m in model.modules():
            if hasattr(m, "inplace"):
                m.inplace = True

        self.model = model

        # Extract metadata
        self._names = getattr(model, "names", {i: f"class{i}" for i in range(80)})
        self._stride = int(max(getattr(model, "stride", torch.tensor([32])).max(), 32))
        self._task = getattr(model, "task", "detect")
        self._nc = len(self._names)

        # Get kpt_shape for pose models
        if hasattr(model, "kpt_shape"):
            self._kpt_shape = model.kpt_shape

        # Get image size from model args
        args = getattr(model, "args", {})
        if isinstance(args, dict):
            imgsz = args.get("imgsz", 640)
            if isinstance(imgsz, int):
                self._imgsz = (imgsz, imgsz)
            else:
                self._imgsz = tuple(imgsz)

        # FP16 for PyTorch
        if self._fp16 and self.device.type != "cpu":
            model.half()

    def _load_onnx(self, weights: str, cuda: bool):
        """Load ONNX model using ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference. Install with: pip install onnxruntime-gpu")

        # Select providers
        providers = ["CPUExecutionProvider"]
        if cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": self.device.index or 0}))
        else:
            self.device = torch.device("cpu")

        # Create session
        self.session = ort.InferenceSession(weights, providers=providers)

        # Get output names
        self.output_names = [x.name for x in self.session.get_outputs()]

        # Extract metadata
        metadata = self.session.get_modelmeta().custom_metadata_map
        self._parse_metadata(metadata)

        # Check for dynamic shapes
        input_shape = self.session.get_inputs()[0].shape
        self._dynamic = isinstance(input_shape[0], str) or any(isinstance(d, str) for d in input_shape)

        # Check FP16
        input_type = self.session.get_inputs()[0].type
        if "float16" in input_type:
            self._fp16 = True

        # Setup IO binding for static shapes on GPU
        if not self._dynamic and self.device.type == "cuda":
            self.io = self.session.io_binding()
            self._bindings = []
            for output in self.session.get_outputs():
                out_fp16 = "float16" in output.type
                shape = output.shape
                # Replace dynamic dims with actual values
                shape = [d if isinstance(d, int) else 1 for d in shape]
                y_tensor = torch.empty(
                    shape,
                    dtype=torch.float16 if out_fp16 else torch.float32,
                    device=self.device
                )
                self.io.bind_output(
                    name=output.name,
                    device_type="cuda",
                    device_id=self.device.index or 0,
                    element_type=np.float16 if out_fp16 else np.float32,
                    shape=tuple(y_tensor.shape),
                    buffer_ptr=y_tensor.data_ptr(),
                )
                self._bindings.append(y_tensor)

    def _load_tensorrt(self, weights: str):
        """Load TensorRT engine."""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt is required for TensorRT inference. Install with: pip install tensorrt")

        # Binding info container
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        # Create logger and runtime
        logger = trt.Logger(trt.Logger.WARNING)

        with open(weights, "rb") as f:
            # Try to read embedded metadata
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                self._parse_metadata(metadata)
            except (UnicodeDecodeError, json.JSONDecodeError):
                f.seek(0)  # No metadata, reset to beginning

            # Deserialize engine
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = engine.create_execution_context()
        self._engine = engine

        # Check TensorRT version (10.0+ has different API)
        is_trt10 = not hasattr(engine, "num_bindings")
        self._is_trt10 = is_trt10

        # Get number of bindings/tensors
        num = range(engine.num_io_tensors) if is_trt10 else range(engine.num_bindings)

        # Setup bindings
        bindings = OrderedDict()
        self.output_names = []

        for i in num:
            if is_trt10:
                name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input:
                    shape = engine.get_tensor_shape(name)
                    if -1 in shape:
                        self._dynamic = True
                        shape = engine.get_tensor_profile_shape(name, 0)[2]
                        self.context.set_input_shape(name, shape)
                else:
                    self.output_names.append(name)
                shape = tuple(self.context.get_tensor_shape(name))
            else:
                name = engine.get_binding_name(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                if is_input:
                    shape = engine.get_binding_shape(i)
                    if -1 in shape:
                        self._dynamic = True
                else:
                    self.output_names.append(name)
                shape = tuple(self.context.get_binding_shape(i))

            if dtype == np.float16:
                self._fp16 = True

            # Allocate tensor
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.bindings = bindings
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

    def _parse_metadata(self, metadata: dict):
        """Parse metadata from model file."""
        if not metadata or not isinstance(metadata, dict):
            return

        # Convert string values
        for k, v in metadata.items():
            if k in {"stride", "batch"} and isinstance(v, str):
                metadata[k] = int(v)
            elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                try:
                    metadata[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass

        # Extract values
        if "stride" in metadata:
            self._stride = int(metadata["stride"])
        if "task" in metadata:
            self._task = metadata["task"]
        if "names" in metadata:
            names = metadata["names"]
            if isinstance(names, list):
                self._names = {i: n for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self._names = {int(k): v for k, v in names.items()}
        if "imgsz" in metadata:
            imgsz = metadata["imgsz"]
            if isinstance(imgsz, int):
                self._imgsz = (imgsz, imgsz)
            else:
                self._imgsz = tuple(imgsz)
        if "kpt_shape" in metadata:
            self._kpt_shape = tuple(metadata["kpt_shape"])

        self._nc = len(self._names) if self._names else 80

    def warmup(self, imgsz: Tuple[int, int] = None):
        """Warmup model with a dummy input."""
        if imgsz is None:
            imgsz = self._imgsz

        # Create dummy input
        im = torch.empty(1, 3, *imgsz, dtype=torch.float16 if self._fp16 else torch.float32, device=self.device)

        # Run inference
        for _ in range(2 if self.device.type == "cuda" else 1):
            self.forward(im)

    def forward(self, im: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run inference on input tensor.

        Args:
            im: Input tensor of shape (B, C, H, W)

        Returns:
            Model predictions
        """
        # Handle FP16
        if self._fp16 and im.dtype != torch.float16:
            im = im.half()

        # Move to device
        if im.device != self.device:
            im = im.to(self.device)

        # Backend-specific inference
        if self.pt:
            return self._forward_pytorch(im)
        elif self.onnx:
            return self._forward_onnx(im)
        elif self.engine:
            return self._forward_tensorrt(im)

    def _forward_pytorch(self, im: torch.Tensor) -> torch.Tensor:
        """PyTorch inference."""
        with torch.no_grad():
            return self.model(im)

    def _forward_onnx(self, im: torch.Tensor) -> torch.Tensor:
        """ONNX Runtime inference."""
        # Get expected batch size from model input shape
        input_shape = self.session.get_inputs()[0].shape
        expected_batch = input_shape[0]
        actual_batch = im.shape[0]

        # Check if model has dynamic batch (string dimension like 'batch')
        is_dynamic_batch = isinstance(expected_batch, str)

        # If fixed batch size and input batch doesn't match, process one at a time
        if not is_dynamic_batch and not self._dynamic and actual_batch != expected_batch:
            # Process images one at a time for fixed batch size models
            results = []
            for i in range(actual_batch):
                single_im = im[i:i+1]  # Keep batch dimension
                im_np = single_im.cpu().numpy()
                y_single = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im_np})
                results.append(y_single[0] if len(y_single) == 1 else y_single)

            # Concatenate results along batch dimension
            if isinstance(results[0], np.ndarray):
                y = [np.concatenate(results, axis=0)]
            else:
                # Multiple outputs
                num_outputs = len(results[0])
                y = [np.concatenate([r[j] for r in results], axis=0) for j in range(num_outputs)]
            return self._from_numpy(y)

        # Dynamic shapes or CPU: use numpy path
        if self._dynamic or is_dynamic_batch or self.device.type == "cpu":
            im_np = im.cpu().numpy()
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im_np})
        else:
            # Static shapes on GPU with matching batch: use IO binding
            self.io.bind_input(
                name="images",
                device_type="cuda",
                device_id=self.device.index or 0,
                element_type=np.float16 if self._fp16 else np.float32,
                shape=tuple(im.shape),
                buffer_ptr=im.data_ptr(),
            )
            self.session.run_with_iobinding(self.io)
            y = [b.clone() for b in self._bindings]

        return self._from_numpy(y)

    def _forward_tensorrt(self, im: torch.Tensor) -> torch.Tensor:
        """TensorRT inference."""
        # Handle dynamic shapes
        if self._dynamic and im.shape != self.bindings["images"].shape:
            if self._is_trt10:
                self.context.set_input_shape("images", im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
            else:
                i = self._engine.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self._engine.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

        # Update input binding address
        self.binding_addrs["images"] = int(im.data_ptr())

        # Execute
        self.context.execute_v2(list(self.binding_addrs.values()))

        # Collect outputs
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        return y[0] if len(y) == 1 else y

    def _from_numpy(self, y: Union[np.ndarray, List[np.ndarray]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Convert numpy arrays to tensors on the correct device."""
        if isinstance(y, (list, tuple)):
            return [self._from_numpy(x) for x in y] if len(y) > 1 else self._from_numpy(y[0])
        return torch.from_numpy(y).to(self.device) if isinstance(y, np.ndarray) else y

    # Properties
    @property
    def names(self) -> Dict[int, str]:
        """Class names dictionary."""
        return self._names

    @property
    def stride(self) -> int:
        """Model stride."""
        return self._stride

    @property
    def task(self) -> str:
        """Model task (detect, classify, pose)."""
        return self._task

    @property
    def imgsz(self) -> Tuple[int, int]:
        """Input image size."""
        return self._imgsz

    @property
    def nc(self) -> int:
        """Number of classes."""
        return self._nc

    @property
    def kpt_shape(self) -> Optional[Tuple[int, int]]:
        """Keypoint shape for pose models (num_keypoints, dims)."""
        return self._kpt_shape

    @property
    def fp16(self) -> bool:
        """Whether model uses FP16."""
        return self._fp16

    @property
    def dynamic(self) -> bool:
        """Whether model supports dynamic shapes."""
        return self._dynamic
