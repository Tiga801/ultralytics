"""
Comparison test cases between custom models/ module and Ultralytics built-in inference.

This test module generates visual and numerical comparisons for manual review.
Tests pass regardless of differences - outputs are for analysis purposes.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Custom models module
from models import Predictor

# Ultralytics built-in
from ultralytics import YOLO

# =============================================================================
# Configuration
# =============================================================================

WEIGHTS_DIR = PROJECT_ROOT / "weights"
IMAGES_DIR = PROJECT_ROOT / "datasets" / "images"
OUTPUT_DIR = PROJECT_ROOT / "results"

TEST_CONFIGS = {
    "detection": {
        "model": WEIGHTS_DIR / "yolo11m.pt",
        "image": IMAGES_DIR / "coco09.jpg",
        "task": "detect",
    },
    "pose": {
        "model": WEIGHTS_DIR / "yolo11m-pose.pt",
        "image": IMAGES_DIR / "coco36.jpg",
        "task": "pose",
    },
    "classification": {
        "model": WEIGHTS_DIR / "yolo11m-cls.pt",
        "image": IMAGES_DIR / "coco34.jpg",
        "task": "classify",
    },
}

INFERENCE_PARAMS = {
    "conf": 0.25,
    "iou": 0.45,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BoxComparison:
    """Comparison result for a single detection box."""

    custom_box: Optional[np.ndarray]
    ultra_box: Optional[np.ndarray]
    iou: float
    coord_diff: float
    conf_diff: float
    cls_match: bool


@dataclass
class DetectionComparisonResult:
    """Full comparison result for detection task."""

    custom_count: int
    ultra_count: int
    matched_pairs: List[BoxComparison]
    unmatched_custom: List[np.ndarray]
    unmatched_ultra: List[np.ndarray]
    mean_iou: float
    mean_coord_diff: float
    mean_conf_diff: float


@dataclass
class PoseComparisonResult:
    """Full comparison result for pose task."""

    custom_count: int
    ultra_count: int
    box_comparison: DetectionComparisonResult
    keypoint_diffs: List[Dict]
    mean_keypoint_distance: float
    mean_keypoint_conf_diff: float


@dataclass
class ClassificationComparisonResult:
    """Full comparison result for classification task."""

    custom_top1: int
    ultra_top1: int
    custom_top5: List[int]
    ultra_top5: List[int]
    custom_probs: np.ndarray
    ultra_probs: np.ndarray
    top1_match: bool
    top5_overlap: int
    prob_diff_top5: List[float]


# =============================================================================
# Utility Functions
# =============================================================================


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_boxes_hungarian(
    custom_boxes: np.ndarray,
    ultra_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match boxes using Hungarian algorithm based on IoU.

    Returns:
        matched_pairs: List of (custom_idx, ultra_idx) tuples
        unmatched_custom: Indices of unmatched custom boxes
        unmatched_ultra: Indices of unmatched ultralytics boxes
    """
    from scipy.optimize import linear_sum_assignment

    if len(custom_boxes) == 0 or len(ultra_boxes) == 0:
        return [], list(range(len(custom_boxes))), list(range(len(ultra_boxes)))

    # Build IoU cost matrix (negative for minimization)
    cost_matrix = np.zeros((len(custom_boxes), len(ultra_boxes)))
    for i, cb in enumerate(custom_boxes):
        for j, ub in enumerate(ultra_boxes):
            cost_matrix[i, j] = -compute_iou(cb[:4], ub[:4])

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        if -cost_matrix[i, j] >= iou_thresh:
            matched_pairs.append((i, j))

    matched_custom = {p[0] for p in matched_pairs}
    matched_ultra = {p[1] for p in matched_pairs}

    unmatched_custom = [i for i in range(len(custom_boxes)) if i not in matched_custom]
    unmatched_ultra = [j for j in range(len(ultra_boxes)) if j not in matched_ultra]

    return matched_pairs, unmatched_custom, unmatched_ultra


def compute_keypoint_distance(
    kpts1: np.ndarray, kpts2: np.ndarray, conf_thresh: float = 0.5
) -> Tuple[float, float]:
    """
    Compute mean Euclidean distance and confidence diff between keypoint sets.

    Args:
        kpts1, kpts2: Arrays of shape (17, 3) with [x, y, conf]

    Returns:
        (mean_distance, mean_conf_diff)
    """
    valid_mask = (kpts1[:, 2] > conf_thresh) & (kpts2[:, 2] > conf_thresh)

    if valid_mask.sum() == 0:
        return float("nan"), float("nan")

    dists = np.linalg.norm(kpts1[valid_mask, :2] - kpts2[valid_mask, :2], axis=1)
    conf_diffs = np.abs(kpts1[valid_mask, 2] - kpts2[valid_mask, 2])

    return float(np.mean(dists)), float(np.mean(conf_diffs))


def create_side_by_side_image(
    custom_result,
    ultra_result,
    title: str = "",
) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Returns:
        Combined image with custom result on left, ultralytics on right
    """
    # Plot both results
    custom_annotated = custom_result.plot()
    ultra_annotated = ultra_result.plot()

    # Ensure same height
    h1, w1 = custom_annotated.shape[:2]
    h2, w2 = ultra_annotated.shape[:2]

    max_h = max(h1, h2)
    if h1 < max_h:
        pad = np.zeros((max_h - h1, w1, 3), dtype=np.uint8)
        custom_annotated = np.vstack([custom_annotated, pad])
    if h2 < max_h:
        pad = np.zeros((max_h - h2, w2, 3), dtype=np.uint8)
        ultra_annotated = np.vstack([ultra_annotated, pad])

    # Add labels
    label_height = 40
    custom_label = np.zeros((label_height, w1, 3), dtype=np.uint8)
    ultra_label = np.zeros((label_height, w2, 3), dtype=np.uint8)

    cv2.putText(
        custom_label,
        "Custom models/",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        ultra_label,
        "Ultralytics YOLO",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    custom_with_label = np.vstack([custom_label, custom_annotated])
    ultra_with_label = np.vstack([ultra_label, ultra_annotated])

    # Combine horizontally with separator
    separator = np.ones((custom_with_label.shape[0], 5, 3), dtype=np.uint8) * 128
    combined = np.hstack([custom_with_label, separator, ultra_with_label])

    # Add title if provided
    if title:
        title_bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            title_bar,
            title,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        combined = np.vstack([title_bar, combined])

    return combined


# =============================================================================
# Comparison Functions
# =============================================================================


def compare_detection_results(
    custom_result,
    ultra_result,
    class_names: Dict[int, str],
) -> DetectionComparisonResult:
    """Compare detection results from both implementations."""
    # Extract boxes as numpy arrays
    if custom_result.boxes is not None and len(custom_result.boxes) > 0:
        custom_boxes = custom_result.boxes.data.cpu().numpy()
    else:
        custom_boxes = np.empty((0, 6))

    if ultra_result.boxes is not None and len(ultra_result.boxes) > 0:
        ultra_boxes = ultra_result.boxes.data.cpu().numpy()
    else:
        ultra_boxes = np.empty((0, 6))

    # Match boxes
    matched_pairs, unmatched_custom_idx, unmatched_ultra_idx = match_boxes_hungarian(
        custom_boxes, ultra_boxes, iou_thresh=0.5
    )

    # Compute comparison metrics for matched pairs
    box_comparisons = []
    ious, coord_diffs, conf_diffs = [], [], []

    for ci, ui in matched_pairs:
        cb, ub = custom_boxes[ci], ultra_boxes[ui]
        iou = compute_iou(cb[:4], ub[:4])

        # Center distance
        c_center = np.array([(cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2])
        u_center = np.array([(ub[0] + ub[2]) / 2, (ub[1] + ub[3]) / 2])
        coord_diff = np.linalg.norm(c_center - u_center)

        conf_diff = abs(cb[4] - ub[4])
        cls_match = int(cb[5]) == int(ub[5])

        box_comparisons.append(
            BoxComparison(
                custom_box=cb,
                ultra_box=ub,
                iou=iou,
                coord_diff=coord_diff,
                conf_diff=conf_diff,
                cls_match=cls_match,
            )
        )

        ious.append(iou)
        coord_diffs.append(coord_diff)
        conf_diffs.append(conf_diff)

    return DetectionComparisonResult(
        custom_count=len(custom_boxes),
        ultra_count=len(ultra_boxes),
        matched_pairs=box_comparisons,
        unmatched_custom=[custom_boxes[i] for i in unmatched_custom_idx],
        unmatched_ultra=[ultra_boxes[i] for i in unmatched_ultra_idx],
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        mean_coord_diff=float(np.mean(coord_diffs)) if coord_diffs else 0.0,
        mean_conf_diff=float(np.mean(conf_diffs)) if conf_diffs else 0.0,
    )


def compare_pose_results(
    custom_result,
    ultra_result,
    class_names: Dict[int, str],
) -> PoseComparisonResult:
    """Compare pose estimation results from both implementations."""
    # First compare boxes
    box_comparison = compare_detection_results(custom_result, ultra_result, class_names)

    # Extract keypoints
    if custom_result.keypoints is not None and len(custom_result.keypoints) > 0:
        custom_kpts = custom_result.keypoints.data.cpu().numpy()
    else:
        custom_kpts = np.empty((0, 17, 3))

    if ultra_result.keypoints is not None and len(ultra_result.keypoints) > 0:
        ultra_kpts = ultra_result.keypoints.data.cpu().numpy()
    else:
        ultra_kpts = np.empty((0, 17, 3))

    # Compare keypoints for matched boxes
    keypoint_diffs = []
    kpt_distances, kpt_conf_diffs = [], []

    # Match keypoints by box order (same as box matching)
    min_count = min(len(custom_kpts), len(ultra_kpts))
    for i in range(min_count):
        dist, conf_diff = compute_keypoint_distance(custom_kpts[i], ultra_kpts[i])
        keypoint_diffs.append(
            {
                "index": i,
                "distance": dist,
                "conf_diff": conf_diff,
            }
        )
        if not np.isnan(dist):
            kpt_distances.append(dist)
        if not np.isnan(conf_diff):
            kpt_conf_diffs.append(conf_diff)

    return PoseComparisonResult(
        custom_count=len(custom_kpts),
        ultra_count=len(ultra_kpts),
        box_comparison=box_comparison,
        keypoint_diffs=keypoint_diffs,
        mean_keypoint_distance=float(np.mean(kpt_distances)) if kpt_distances else 0.0,
        mean_keypoint_conf_diff=float(np.mean(kpt_conf_diffs))
        if kpt_conf_diffs
        else 0.0,
    )


def compare_classification_results(
    custom_result,
    ultra_result,
    class_names: Dict[int, str],
) -> ClassificationComparisonResult:
    """Compare classification results from both implementations."""
    custom_probs = custom_result.probs.data.cpu().numpy()
    ultra_probs = ultra_result.probs.data.cpu().numpy()

    custom_top1 = custom_result.probs.top1
    ultra_top1 = ultra_result.probs.top1
    custom_top5 = list(custom_result.probs.top5)
    ultra_top5 = list(ultra_result.probs.top5)

    # Calculate overlap in top5
    top5_overlap = len(set(custom_top5) & set(ultra_top5))

    # Probability differences for ultra top5
    prob_diff_top5 = [abs(custom_probs[i] - ultra_probs[i]) for i in ultra_top5]

    return ClassificationComparisonResult(
        custom_top1=custom_top1,
        ultra_top1=ultra_top1,
        custom_top5=custom_top5,
        ultra_top5=ultra_top5,
        custom_probs=custom_probs,
        ultra_probs=ultra_probs,
        top1_match=custom_top1 == ultra_top1,
        top5_overlap=top5_overlap,
        prob_diff_top5=prob_diff_top5,
    )


# =============================================================================
# Report Generators
# =============================================================================


def generate_detection_report(
    result: DetectionComparisonResult, class_names: Dict[int, str]
) -> str:
    """Generate text report for detection comparison."""
    lines = [
        "=" * 60,
        "DETECTION COMPARISON REPORT",
        "=" * 60,
        "",
        "SUMMARY:",
        f"  Custom detections: {result.custom_count}",
        f"  Ultralytics detections: {result.ultra_count}",
        f"  Matched pairs: {len(result.matched_pairs)}",
        f"  Unmatched (custom only): {len(result.unmatched_custom)}",
        f"  Unmatched (ultra only): {len(result.unmatched_ultra)}",
        "",
        "METRICS (for matched pairs):",
        f"  Mean IoU: {result.mean_iou:.4f}",
        f"  Mean center distance: {result.mean_coord_diff:.2f} px",
        f"  Mean confidence diff: {result.mean_conf_diff:.4f}",
        "",
        "MATCHED PAIRS DETAIL:",
    ]

    for i, bc in enumerate(result.matched_pairs):
        cls_name = class_names.get(int(bc.custom_box[5]), str(int(bc.custom_box[5])))
        lines.append(
            f"  [{i + 1}] {cls_name}: IoU={bc.iou:.3f}, "
            f"conf_diff={bc.conf_diff:.4f}, cls_match={bc.cls_match}"
        )

    if result.unmatched_custom:
        lines.append("")
        lines.append("UNMATCHED CUSTOM DETECTIONS:")
        for i, box in enumerate(result.unmatched_custom):
            cls_name = class_names.get(int(box[5]), str(int(box[5])))
            lines.append(f"  [{i + 1}] {cls_name} (conf={box[4]:.3f})")

    if result.unmatched_ultra:
        lines.append("")
        lines.append("UNMATCHED ULTRALYTICS DETECTIONS:")
        for i, box in enumerate(result.unmatched_ultra):
            cls_name = class_names.get(int(box[5]), str(int(box[5])))
            lines.append(f"  [{i + 1}] {cls_name} (conf={box[4]:.3f})")

    return "\n".join(lines)


def generate_pose_report(
    result: PoseComparisonResult, class_names: Dict[int, str]
) -> str:
    """Generate text report for pose comparison."""
    lines = [
        "=" * 60,
        "POSE ESTIMATION COMPARISON REPORT",
        "=" * 60,
        "",
        "SUMMARY:",
        f"  Custom poses: {result.custom_count}",
        f"  Ultralytics poses: {result.ultra_count}",
        "",
        "BOUNDING BOX COMPARISON:",
        f"  Matched pairs: {len(result.box_comparison.matched_pairs)}",
        f"  Mean IoU: {result.box_comparison.mean_iou:.4f}",
        f"  Mean center distance: {result.box_comparison.mean_coord_diff:.2f} px",
        "",
        "KEYPOINT COMPARISON:",
        f"  Mean keypoint distance: {result.mean_keypoint_distance:.2f} px",
        f"  Mean keypoint conf diff: {result.mean_keypoint_conf_diff:.4f}",
        "",
        "KEYPOINT DETAIL PER DETECTION:",
    ]

    for kd in result.keypoint_diffs:
        dist_str = f"{kd['distance']:.2f}" if not np.isnan(kd["distance"]) else "N/A"
        conf_str = f"{kd['conf_diff']:.4f}" if not np.isnan(kd["conf_diff"]) else "N/A"
        lines.append(f"  [{kd['index'] + 1}] distance={dist_str} px, conf_diff={conf_str}")

    return "\n".join(lines)


def generate_classification_report(
    result: ClassificationComparisonResult,
    class_names: Dict[int, str],
) -> str:
    """Generate text report for classification comparison."""
    lines = [
        "=" * 60,
        "CLASSIFICATION COMPARISON REPORT",
        "=" * 60,
        "",
        "TOP-1 PREDICTION:",
        f"  Custom: {class_names.get(result.custom_top1, result.custom_top1)} "
        f"(prob={result.custom_probs[result.custom_top1]:.4f})",
        f"  Ultralytics: {class_names.get(result.ultra_top1, result.ultra_top1)} "
        f"(prob={result.ultra_probs[result.ultra_top1]:.4f})",
        f"  Match: {result.top1_match}",
        "",
        "TOP-5 PREDICTIONS:",
        "  Custom:",
    ]

    for idx in result.custom_top5:
        lines.append(
            f"    {class_names.get(idx, idx)}: {result.custom_probs[idx]:.4f}"
        )

    lines.append("  Ultralytics:")
    for idx in result.ultra_top5:
        lines.append(
            f"    {class_names.get(idx, idx)}: {result.ultra_probs[idx]:.4f}"
        )

    lines.extend(
        [
            "",
            f"  Top-5 overlap: {result.top5_overlap}/5 classes in common",
            f"  Mean prob diff (ultra top5): {np.mean(result.prob_diff_top5):.4f}",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def output_dirs():
    """Create output directories for comparison results."""
    base_dir = OUTPUT_DIR
    dirs = {
        "detection": base_dir / "detection",
        "pose": base_dir / "pose",
        "classification": base_dir / "classification",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =============================================================================
# Test Classes
# =============================================================================


class TestDetectionComparison:
    """Test detection inference comparison."""

    def test_detection_comparison(self, output_dirs):
        """Compare detection results and generate outputs."""
        config = TEST_CONFIGS["detection"]
        image_path = str(config["image"])
        output_dir = output_dirs["detection"]

        # Check prerequisites
        if not config["model"].exists():
            pytest.skip(f"Model not found: {config['model']}")
        if not config["image"].exists():
            pytest.skip(f"Image not found: {config['image']}")

        print(f"\n{'=' * 60}")
        print("DETECTION COMPARISON TEST")
        print(f"{'=' * 60}")
        print(f"Model: {config['model']}")
        print(f"Image: {config['image']}")
        print(f"Device: {INFERENCE_PARAMS['device']}")

        # Load models
        print("\nLoading models...")
        custom_predictor = Predictor(
            model_path=str(config["model"]),
            task=config["task"],
            device=INFERENCE_PARAMS["device"],
            conf=INFERENCE_PARAMS["conf"],
            iou=INFERENCE_PARAMS["iou"],
        )
        ultra_model = YOLO(str(config["model"]))

        # Run inference
        print("Running inference...")
        custom_results = custom_predictor(
            image_path,
            conf=INFERENCE_PARAMS["conf"],
            iou=INFERENCE_PARAMS["iou"],
        )
        ultra_results = ultra_model(
            image_path,
            conf=INFERENCE_PARAMS["conf"],
            iou=INFERENCE_PARAMS["iou"],
            verbose=False,
        )

        custom_result = custom_results[0]
        ultra_result = ultra_results[0]

        # Compare
        print("Comparing results...")
        class_names = custom_result.names
        comparison = compare_detection_results(custom_result, ultra_result, class_names)

        # Generate side-by-side image
        side_by_side = create_side_by_side_image(
            custom_result,
            ultra_result,
            title="Detection Comparison: Custom vs Ultralytics",
        )
        output_image = output_dir / "side_by_side_coco09.jpg"
        cv2.imwrite(str(output_image), side_by_side)
        print(f"Saved: {output_image}")

        # Generate report
        report = generate_detection_report(comparison, class_names)
        output_report = output_dir / "diff_report_detection.txt"
        output_report.write_text(report)
        print(f"Saved: {output_report}")

        # Print summary to console
        print("\n" + report)

        # Always pass - this is for manual review
        assert True, "Detection comparison completed - check results/comparison/"


class TestPoseComparison:
    """Test pose estimation inference comparison."""

    def test_pose_comparison(self, output_dirs):
        """Compare pose results and generate outputs."""
        config = TEST_CONFIGS["pose"]
        image_path = str(config["image"])
        output_dir = output_dirs["pose"]

        # Check prerequisites
        if not config["model"].exists():
            pytest.skip(f"Model not found: {config['model']}")
        if not config["image"].exists():
            pytest.skip(f"Image not found: {config['image']}")

        print(f"\n{'=' * 60}")
        print("POSE ESTIMATION COMPARISON TEST")
        print(f"{'=' * 60}")
        print(f"Model: {config['model']}")
        print(f"Image: {config['image']}")
        print(f"Device: {INFERENCE_PARAMS['device']}")

        # Load models
        print("\nLoading models...")
        custom_predictor = Predictor(
            model_path=str(config["model"]),
            task=config["task"],
            device=INFERENCE_PARAMS["device"],
            conf=INFERENCE_PARAMS["conf"],
            iou=INFERENCE_PARAMS["iou"],
        )
        ultra_model = YOLO(str(config["model"]))

        # Run inference
        print("Running inference...")
        custom_results = custom_predictor(image_path)
        ultra_results = ultra_model(image_path, verbose=False)

        custom_result = custom_results[0]
        ultra_result = ultra_results[0]

        # Compare
        print("Comparing results...")
        class_names = custom_result.names
        comparison = compare_pose_results(custom_result, ultra_result, class_names)

        # Generate side-by-side image
        side_by_side = create_side_by_side_image(
            custom_result,
            ultra_result,
            title="Pose Estimation Comparison",
        )
        output_image = output_dir / "side_by_side_coco36.jpg"
        cv2.imwrite(str(output_image), side_by_side)
        print(f"Saved: {output_image}")

        # Generate report
        report = generate_pose_report(comparison, class_names)
        output_report = output_dir / "diff_report_pose.txt"
        output_report.write_text(report)
        print(f"Saved: {output_report}")

        # Print summary to console
        print("\n" + report)

        assert True, "Pose comparison completed - check results/comparison/"


class TestClassificationComparison:
    """Test classification inference comparison."""

    def test_classification_comparison(self, output_dirs):
        """Compare classification results and generate outputs."""
        config = TEST_CONFIGS["classification"]
        image_path = str(config["image"])
        output_dir = output_dirs["classification"]

        # Check prerequisites
        if not config["model"].exists():
            pytest.skip(f"Model not found: {config['model']}")
        if not config["image"].exists():
            pytest.skip(f"Image not found: {config['image']}")

        print(f"\n{'=' * 60}")
        print("CLASSIFICATION COMPARISON TEST")
        print(f"{'=' * 60}")
        print(f"Model: {config['model']}")
        print(f"Image: {config['image']}")
        print(f"Device: {INFERENCE_PARAMS['device']}")

        # Load models
        print("\nLoading models...")
        custom_predictor = Predictor(
            model_path=str(config["model"]),
            task=config["task"],
            device=INFERENCE_PARAMS["device"],
        )
        ultra_model = YOLO(str(config["model"]))

        # Run inference
        print("Running inference...")
        custom_results = custom_predictor(image_path)
        ultra_results = ultra_model(image_path, verbose=False)

        custom_result = custom_results[0]
        ultra_result = ultra_results[0]

        # Compare
        print("Comparing results...")
        class_names = custom_result.names
        comparison = compare_classification_results(
            custom_result, ultra_result, class_names
        )

        # Generate side-by-side image
        side_by_side = create_side_by_side_image(
            custom_result,
            ultra_result,
            title="Classification Comparison",
        )
        output_image = output_dir / "side_by_side_coco34.jpg"
        cv2.imwrite(str(output_image), side_by_side)
        print(f"Saved: {output_image}")

        # Generate report
        report = generate_classification_report(comparison, class_names)
        output_report = output_dir / "diff_report_classification.txt"
        output_report.write_text(report)
        print(f"Saved: {output_report}")

        # Print summary to console
        print("\n" + report)

        assert True, "Classification comparison completed - check results/comparison/"


class TestSummary:
    """Generate overall comparison summary."""

    def test_generate_summary(self, output_dirs):
        """Generate summary report after all comparisons."""
        summary_lines = [
            "=" * 60,
            "INFERENCE COMPARISON SUMMARY",
            "=" * 60,
            "",
            "Test Configuration:",
            f"  Confidence threshold: {INFERENCE_PARAMS['conf']}",
            f"  IoU threshold: {INFERENCE_PARAMS['iou']}",
            f"  Device: {INFERENCE_PARAMS['device']}",
            "",
            "Models tested:",
            f"  Detection: {TEST_CONFIGS['detection']['model'].name}",
            f"  Pose: {TEST_CONFIGS['pose']['model'].name}",
            f"  Classification: {TEST_CONFIGS['classification']['model'].name}",
            "",
            "Test images:",
            f"  Detection: {TEST_CONFIGS['detection']['image'].name}",
            f"  Pose: {TEST_CONFIGS['pose']['image'].name}",
            f"  Classification: {TEST_CONFIGS['classification']['image'].name}",
            "",
            "Output locations:",
            f"  Detection: {output_dirs['detection']}",
            f"  Pose: {output_dirs['pose']}",
            f"  Classification: {output_dirs['classification']}",
            "",
            "Note: All tests pass regardless of differences.",
            "Review the generated reports and images for manual analysis.",
        ]

        summary = "\n".join(summary_lines)
        summary_path = OUTPUT_DIR / "summary_report.txt"
        summary_path.write_text(summary)
        print(f"\nSaved: {summary_path}")
        print("\n" + summary)

        assert True


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run comparison tests standalone."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
