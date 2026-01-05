"""
Postprocessing - NMS and coordinate transformations.
No dependencies on ultralytics internals.
"""

from typing import List, Optional, Tuple, Union

import torch
import torchvision


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2) format.

    Args:
        x: Boxes in xywh format (..., 4)

    Returns:
        Boxes in xyxy format
    """
    assert x.shape[-1] >= 4, f"Expected at least 4 values, got {x.shape[-1]}"
    y = torch.empty_like(x)
    xy = x[..., :2]  # center xy
    wh = x[..., 2:4] / 2  # half wh
    y[..., :2] = xy - wh  # top left
    y[..., 2:4] = xy + wh  # bottom right
    if x.shape[-1] > 4:
        y[..., 4:] = x[..., 4:]
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (x_center, y_center, w, h) format.

    Args:
        x: Boxes in xyxy format (..., 4)

    Returns:
        Boxes in xywh format
    """
    y = torch.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    if x.shape[-1] > 4:
        y[..., 4:] = x[..., 4:]
    return y


def clip_boxes(boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: Boxes in xyxy format (..., 4)
        shape: Image shape (height, width)

    Returns:
        Clipped boxes
    """
    boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
    boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
    boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
    boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    return boxes


def clip_coords(coords: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Clip coordinates to image boundaries.

    Args:
        coords: Coordinates (..., 2) or (..., K, 2/3)
        shape: Image shape (height, width)

    Returns:
        Clipped coordinates
    """
    if coords.ndim == 3:
        # (N, K, 2/3) format
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:
        # (N, 2) format
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])
    return coords


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: torch.Tensor,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[float, Tuple[float, float]]] = None,
) -> torch.Tensor:
    """
    Rescale boxes from img1_shape (model input) to img0_shape (original image).

    Args:
        img1_shape: Model input shape (H, W)
        boxes: Boxes in xyxy format (N, 4+)
        img0_shape: Original image shape (H, W)
        ratio_pad: Tuple of (ratio, (pad_w, pad_h)) from letterbox

    Returns:
        Scaled boxes
    """
    if ratio_pad is None:
        # Calculate from shapes
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_w = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_h = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0]
        pad_w, pad_h = ratio_pad[1]

    # Remove padding
    boxes[..., 0] -= pad_w  # x1
    boxes[..., 1] -= pad_h  # y1
    boxes[..., 2] -= pad_w  # x2
    boxes[..., 3] -= pad_h  # y2

    # Undo scaling
    boxes[..., :4] /= gain

    # Clip to image boundaries
    return clip_boxes(boxes, img0_shape)


def scale_coords(
    img1_shape: Tuple[int, int],
    coords: torch.Tensor,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[float, Tuple[float, float]]] = None,
) -> torch.Tensor:
    """
    Rescale coordinates (e.g., keypoints) from img1_shape to img0_shape.

    Args:
        img1_shape: Model input shape (H, W)
        coords: Coordinates (N, K, 2/3) or (N, 2)
        img0_shape: Original image shape (H, W)
        ratio_pad: Tuple of (ratio, (pad_w, pad_h)) from letterbox

    Returns:
        Scaled coordinates
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_w = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_h = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0]
        pad_w, pad_h = ratio_pad[1]

    # Remove padding and scale
    coords[..., 0] = (coords[..., 0] - pad_w) / gain  # x
    coords[..., 1] = (coords[..., 1] - pad_h) / gain  # y

    # Clip to image boundaries
    return clip_coords(coords, img0_shape)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    max_det: int = 300,
    nc: int = 0,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> List[torch.Tensor]:
    """
    Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction: Model output tensor (B, N, 4+nc+...) in xywh format
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        classes: Filter by class (list of class indices)
        agnostic: Class-agnostic NMS
        multi_label: Allow multiple labels per box
        max_det: Maximum detections per image
        nc: Number of classes (auto-detect if 0)
        max_nms: Maximum boxes for NMS
        max_wh: Maximum box width/height for offset

    Returns:
        List of detections per image, each (N, 6+extra) with (x1, y1, x2, y2, conf, cls, ...extra)
        Extra columns (e.g., keypoints) are preserved if present in input.
    """
    # Check dimensions
    assert 0 <= conf_thres <= 1, f"Invalid conf_thres {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid iou_thres {iou_thres}"

    # Determine prediction format
    bs = prediction.shape[0]  # batch size
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)

    # Handle transposed format: (batch, features, predictions) -> (batch, predictions, features)
    # YOLO models output (batch, 4+nc, num_preds) which needs to be transposed to (batch, num_preds, 4+nc)
    if prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)

    # Determine number of classes
    if nc == 0:
        nc = prediction.shape[2] - 4  # classes = total - box coords

    # Indices for slicing
    mi = 4 + nc  # mask start index (if any extra data like keypoints)

    # Check if there's extra data (e.g., keypoints) after class predictions
    has_extra = prediction.shape[2] > mi

    # Convert xywh to xyxy
    xc = prediction[..., :4].clone()
    prediction = prediction.clone()
    prediction[..., :4] = xywh2xyxy(xc)

    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):  # Per image
        # Save extra data (keypoints, etc.) before filtering
        extra = x[:, mi:] if has_extra else None

        # Filter by confidence
        if nc == 1:
            # Single class: conf is in position 4
            conf = x[:, 4:5]
        else:
            # Multi-class: max of class confidences
            conf = x[:, 4:mi].max(1, keepdim=True)[0]

        # Filter candidates
        mask = conf.squeeze(-1) > conf_thres
        x = x[mask]
        conf = conf[mask]
        if extra is not None:
            extra = extra[mask]

        if x.shape[0] == 0:
            continue

        # Get class predictions
        if nc == 1:
            cls = torch.zeros((x.shape[0], 1), device=x.device)
            if extra is not None:
                x = torch.cat((x[:, :4], conf, cls, extra), 1)
            else:
                x = torch.cat((x[:, :4], conf, cls), 1)
        else:
            if multi_label:
                # Multiple labels per box
                i, j = (x[:, 4:mi] > conf_thres).nonzero(as_tuple=False).T
                if extra is not None:
                    x = torch.cat((x[i, :4], x[i, 4 + j, None], j[:, None].float(), extra[i]), 1)
                else:
                    x = torch.cat((x[i, :4], x[i, 4 + j, None], j[:, None].float()), 1)
            else:
                # Best class only
                cls_conf, cls_idx = x[:, 4:mi].max(1, keepdim=True)
                # Filter by class confidence first
                conf_mask = cls_conf.squeeze(-1) > conf_thres
                if extra is not None:
                    x = torch.cat((x[:, :4], cls_conf, cls_idx.float(), extra), 1)
                    x = x[conf_mask]
                else:
                    x = torch.cat((x[:, :4], cls_conf, cls_idx.float()), 1)
                    x = x[conf_mask]

        if x.shape[0] == 0:
            continue

        # Filter by class
        if classes is not None:
            mask = (x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)
            x = x[mask]

        if x.shape[0] == 0:
            continue

        # Sort by confidence (descending)
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Apply NMS
        if agnostic:
            # Class-agnostic NMS
            boxes = x[:, :4]
        else:
            # Per-class NMS (offset boxes by class)
            c = x[:, 5:6] * max_wh  # class offset
            boxes = x[:, :4] + c

        scores = x[:, 4]

        # NMS using torchvision
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        keep = keep[:max_det]

        output[xi] = x[keep]

    return output


def process_detections(
    preds: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    max_det: int = 300,
    nc: int = 0,
) -> List[torch.Tensor]:
    """
    Process raw model predictions into final detections.

    This is a convenience wrapper around non_max_suppression.

    Args:
        preds: Raw model predictions
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        classes: Filter classes
        agnostic: Class-agnostic NMS
        max_det: Max detections
        nc: Number of classes

    Returns:
        List of detection tensors (N, 6) per image
    """
    return non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic,
        max_det=max_det,
        nc=nc,
    )
