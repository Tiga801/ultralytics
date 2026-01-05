"""
Test script for standalone YOLO inference module.

Tests detection inference with the standalone models module.
"""

import cv2
import sys

from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import Predictor, create_predictor


def test_detection():
    """Test object detection inference."""
    print("=" * 60)
    print("Testing Object Detection")
    print("=" * 60)

    # Paths
    model_path = PROJECT_ROOT / "weights" / "yolo11m.pt"
    image_dir = PROJECT_ROOT / "datasets" / "images"
    output_dir = PROJECT_ROOT / "results"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check model exists
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please download the model first.")
        return False

    # Test images
    test_images = [
        "coco36.jpg",
        "coco42.jpg",
        "coco49.jpg",
        "coco61.jpg",
    ]

    # Check images exist
    image_paths = []
    for img_name in test_images:
        img_path = image_dir / img_name
        if img_path.exists():
            image_paths.append(str(img_path))
        else:
            print(f"Image not found: {img_path}")

    if not image_paths:
        print("No test images found!")
        return False

    print(f"Model: {model_path}")
    print(f"Images: {len(image_paths)}")
    print()

    # Create predictor
    print("Loading model...")
    predictor = Predictor(
        model_path=str(model_path),
        task="detect",
        device="cuda",  # Will fallback to CPU if CUDA not available
        conf=0.25,
        iou=0.45,
    )
    print(f"Model loaded: {predictor}")
    print(f"Task: {predictor.task}")
    print(f"Classes: {predictor.nc}")
    print(f"Image size: {predictor.imgsz}")
    print()

    # Run inference
    print("Running inference...")
    results = predictor(image_paths)

    # Process results
    for i, result in enumerate(results):
        img_name = Path(image_paths[i]).name
        print(f"\n[{img_name}]")
        print(f"  Shape: {result.orig_shape}")
        print(f"  Detections: {len(result)}")
        print(f"  Summary: {result.verbose()}")
        print(f"  Speed: {result.speed}")

        # Print box details
        if result.boxes is not None and len(result.boxes) > 0:
            for j, box in enumerate(result.boxes.data[:5]):  # First 5 boxes
                x1, y1, x2, y2, conf, cls = box.tolist()
                cls_name = result.names.get(int(cls), str(int(cls)))
                print(f"    Box {j+1}: {cls_name} ({conf:.2f}) [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        # Save annotated image
        output_path = output_dir / f"detect_{img_name}"
        result.save(str(output_path))
        print(f"  Saved: {output_path}")

    print("\n" + "=" * 60)
    print("Detection test completed!")
    print("=" * 60)
    return True


def test_batch_inference():
    """Test batch inference with multiple images."""
    print("\n" + "=" * 60)
    print("Testing Batch Inference")
    print("=" * 60)

    model_path = PROJECT_ROOT / "weights" / "yolo11m.pt"
    image_dir = PROJECT_ROOT / "datasets" / "images"
    output_dir = PROJECT_ROOT / "results"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return False

    # Get all images in directory
    image_paths = list(image_dir.glob("*.jpg"))[:4]

    if not image_paths:
        print("No images found!")
        return False

    # Create predictor using factory function
    predictor = create_predictor(
        str(model_path),
        task="detect",
        device="cuda",
        conf=0.3,
    )

    # Run batch inference
    print(f"Processing {len(image_paths)} images...")
    results = predictor([str(p) for p in image_paths])

    total_detections = sum(len(r) for r in results)
    print(f"Total detections: {total_detections}")

    # Save results
    for result, path in zip(results, image_paths):
        output_path = output_dir / f"batch_{path.name}"
        result.save(str(output_path))

    print("Batch inference completed!")
    return True


def test_numpy_input():
    """Test inference with numpy array input."""
    print("\n" + "=" * 60)
    print("Testing NumPy Array Input")
    print("=" * 60)

    model_path = PROJECT_ROOT / "weights" / "yolo11m.pt"
    image_path = PROJECT_ROOT / "datasets" / "images" / "coco36.jpg"
    output_dir = PROJECT_ROOT / "results"

    if not model_path.exists() or not image_path.exists():
        print("Model or image not found!")
        return False

    # Load image as numpy array
    img = cv2.imread(str(image_path))
    print(f"Image shape: {img.shape}")

    # Create predictor
    predictor = Predictor(str(model_path), task="detect", device="cuda")

    # Run inference with numpy array
    predictions = predictor(img)

    boxes = predictions[0].boxes
    for i, box in enumerate(boxes.data):
        pass

    for j, bbox in enumerate(boxes):
        pass

    print(f"Detections: {len(predictions[0])}")
    print(f"Summary: {predictions[0].verbose()}")

    # Save result
    output_path = output_dir / "numpy_input_result.jpg"
    predictions[0].save(str(output_path))
    print(f"Saved: {output_path}")

    return True


def test_different_confidence():
    """Test inference with different confidence thresholds."""
    print("\n" + "=" * 60)
    print("Testing Different Confidence Thresholds")
    print("=" * 60)

    model_path = PROJECT_ROOT / "weights" / "yolo11m.pt"
    image_path = PROJECT_ROOT / "datasets" / "images" / "coco42.jpg"
    output_dir = PROJECT_ROOT / "results"

    if not model_path.exists() or not image_path.exists():
        print("Model or image not found!")
        return False

    predictor = Predictor(str(model_path), task="detect", device="cuda")

    # Test different confidence thresholds
    conf_thresholds = [0.1, 0.25, 0.5, 0.75]

    for conf in conf_thresholds:
        results = predictor(str(image_path), conf=conf)
        n_det = len(results[0])
        print(f"  conf={conf}: {n_det} detections")

        # Save with threshold in filename
        output_path = output_dir / f"conf_{conf:.2f}_result.jpg"
        results[0].save(str(output_path))

    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Standalone YOLO Inference Module Test Suite")
    print("#" * 60 + "\n")

    tests = [
        ("Detection", test_detection),
        ("Batch Inference", test_batch_inference),
        ("NumPy Input", test_numpy_input),
        ("Confidence Thresholds", test_different_confidence),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
