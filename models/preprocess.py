"""
Image Preprocessing - LetterBox and normalization utilities.
No dependencies on ultralytics internals.
"""

from typing import List, Tuple, Union

import cv2
import numpy as np
import torch


class LetterBox:
    """
    Resize and pad image while preserving aspect ratio.

    Args:
        new_shape: Target image size (height, width)
        auto: Minimum rectangle padding
        scale_fill: Stretch to fill (no padding)
        scaleup: Allow scaling up (False = only scale down)
        center: Center the image in the padded area
        stride: Stride for auto padding alignment
        pad_value: Padding pixel value (default: 114 gray)
    """

    def __init__(
        self,
        new_shape: Union[int, Tuple[int, int]] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        pad_value: int = 114,
    ):
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.center = center
        self.stride = stride
        self.pad_value = pad_value

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, Tuple[float, float]]]:
        """
        Apply letterbox transformation to image.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            Tuple of (transformed_image, (ratio, (pad_w, pad_h)))
        """
        shape = image.shape[:2]  # Current shape [height, width]
        new_shape = self.new_shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # Only scale down, don't scale up
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w, h
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if self.auto:  # Minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # Stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            r = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # Divide padding into 2 sides
            dh /= 2

        # Resize
        if shape[::-1] != new_unpad:  # Resize if different
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add border
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

        if not self.center:
            top, left = 0, 0
            bottom, right = int(round(dh + 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value)
        )

        return image, (r, (dw, dh))


class CenterCrop:
    """
    Center crop and resize for classification models.

    Args:
        size: Target size (height, width)
    """

    def __init__(self, size: Union[int, Tuple[int, int]] = 224):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply center crop transformation.

        Args:
            image: Input image (H, W, C)

        Returns:
            Cropped and resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = self.size

        # Calculate center crop dimensions
        crop_size = min(h, w)
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2

        # Crop center square
        image = image[top:top + crop_size, left:left + crop_size]

        # Resize to target size
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return image


def preprocess_images(
    images: List[np.ndarray],
    imgsz: Union[int, Tuple[int, int]],
    device: Union[str, torch.device],
    fp16: bool = False,
    stride: int = 32,
    auto: bool = False,
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[float, Tuple[float, float]]]]:
    """
    Preprocess a batch of images for inference.

    Args:
        images: List of images in BGR format (H, W, C)
        imgsz: Target image size
        device: Target device
        fp16: Use FP16 precision
        stride: Model stride for auto padding
        auto: Use minimum rectangle padding

    Returns:
        Tuple of (batch_tensor, original_images, ratio_pads)
    """
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    letterbox = LetterBox(new_shape=imgsz, auto=auto, stride=stride)

    processed = []
    ratio_pads = []

    for img in images:
        # Apply letterbox
        img_lb, ratio_pad = letterbox(img)
        processed.append(img_lb)
        ratio_pads.append(ratio_pad)

    # Stack images
    im = np.stack(processed, axis=0)

    # BGR to RGB
    im = im[..., ::-1]

    # HWC to CHW
    im = im.transpose((0, 3, 1, 2))

    # Ensure contiguous array
    im = np.ascontiguousarray(im)

    # Convert to tensor
    im = torch.from_numpy(im)

    # Move to device
    if isinstance(device, str):
        device = torch.device(device)
    im = im.to(device)

    # Convert dtype
    im = im.half() if fp16 else im.float()

    # Normalize: 0-255 to 0.0-1.0
    im /= 255.0

    return im, images, ratio_pads


def preprocess_classify(
    images: List[np.ndarray],
    imgsz: Union[int, Tuple[int, int]],
    device: Union[str, torch.device],
    fp16: bool = False,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Preprocess images for classification.

    Matches Ultralytics classify_transforms pipeline:
    1. Resize so shortest edge = target size (preserve aspect ratio)
    2. Center crop to exact target size
    3. Convert BGR to RGB
    4. Normalize to [0, 1]
    5. Apply mean/std normalization (default: no normalization for YOLO models)

    Args:
        images: List of images in BGR format
        imgsz: Target image size
        device: Target device
        fp16: Use FP16 precision

    Returns:
        Tuple of (batch_tensor, original_images)
    """
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    target_h, target_w = imgsz

    # Use NO normalization to match Ultralytics defaults for YOLO classification
    # Ultralytics DEFAULT_MEAN = (0.0, 0.0, 0.0), DEFAULT_STD = (1.0, 1.0, 1.0)
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    processed = []

    for img in images:
        h, w = img.shape[:2]

        # Step 1: Resize so shortest edge = target size (preserve aspect ratio)
        # This matches torchvision.transforms.Resize(size) with scalar size
        scale = target_h / min(h, w)  # Assuming square target
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Step 2: Center crop to exact target size
        start_h = (new_h - target_h) // 2
        start_w = (new_w - target_w) // 2
        img_crop = img_resized[start_h : start_h + target_h, start_w : start_w + target_w]

        # Step 3: BGR to RGB
        img_rgb = img_crop[..., ::-1].copy()

        # Step 4: Normalize to [0, 1]
        img_norm = img_rgb.astype(np.float32) / 255.0

        # Step 5: Apply mean/std normalization
        img_norm = (img_norm - mean) / std

        # Step 6: HWC to CHW
        img_chw = img_norm.transpose((2, 0, 1))

        processed.append(img_chw)

    # Stack images
    im = np.stack(processed, axis=0)

    # Ensure contiguous
    im = np.ascontiguousarray(im)

    # Convert to tensor
    im = torch.from_numpy(im)

    # Move to device
    if isinstance(device, str):
        device = torch.device(device)
    im = im.to(device)

    # Convert dtype
    im = im.half() if fp16 else im.float()

    return im, images


def load_image(path: str) -> np.ndarray:
    """
    Load image from file path.

    Args:
        path: Path to image file

    Returns:
        Image as numpy array in BGR format
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def load_images(paths: List[str]) -> List[np.ndarray]:
    """
    Load multiple images from file paths.

    Args:
        paths: List of image file paths

    Returns:
        List of images as numpy arrays
    """
    return [load_image(p) for p in paths]
