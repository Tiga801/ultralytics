# -*- coding: utf-8 -*-
"""MinIO Client Tests.

Tests for the MinIO client module including configuration, upload operations,
and connectivity verification.

Usage:
    # Run all tests
    pytest tests/test_minio.py -v

    # Run only unit tests (no network required)
    pytest tests/test_minio.py -v -m "not integration"

    # Run integration tests (requires MinIO server)
    pytest tests/test_minio.py -v -m integration
"""

import sys
import time
import pytest
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Import the MinIO module
from utils.minio import (
    MinIOClient,
    MinIOConfig,
    MinIOStatistics,
    UploadResult,
    UploadTask,
    UploadType,
    build_object_url,
    generate_object_name,
    generate_timestamped_name,
    get_content_type,
)


class TestMinIOConfig:
    """Tests for MinIOConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values match requirements."""
        config = MinIOConfig()

        assert config.host == "easyair-minio"
        assert config.port == 9000
        assert config.access_key == "ZXYrzg2D6madjXxX8u8T"
        assert config.secret_key == "AxbvSyYHDIarCTCYMueGVp68rCDSgs1w7JrsGgyk"
        assert config.bucket_name == "algorithm"
        assert config.secure is False
        assert config.queue_max_size == 1000
        assert config.jpeg_quality == 95

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = MinIOConfig(
            host="192.168.2.234",
            port=9001,
            bucket_name="test-bucket",
            jpeg_quality=80,
        )

        assert config.host == "192.168.2.234"
        assert config.port == 9001
        assert config.bucket_name == "test-bucket"
        assert config.jpeg_quality == 80

    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        data = {
            "host": "test-host",
            "port": 9002,
            "access_key": "test-key",
            "secret_key": "test-secret",
            "bucket_name": "test-bucket",
        }
        config = MinIOConfig.from_dict(data)

        assert config.host == "test-host"
        assert config.port == 9002
        assert config.access_key == "test-key"
        assert config.secret_key == "test-secret"
        assert config.bucket_name == "test-bucket"

    def test_from_dict_legacy_endpoint(self):
        """Test from_dict handles legacy endpoint format."""
        data = {
            "endpoint": "legacy-host:9003",
            "access_key": "key",
            "secret_key": "secret",
        }
        config = MinIOConfig.from_dict(data)

        assert config.host == "legacy-host"
        assert config.port == 9003

    def test_to_dict(self):
        """Test configuration serialization."""
        config = MinIOConfig(host="test", port=9000)
        data = config.to_dict()

        assert data["host"] == "test"
        assert data["port"] == 9000
        assert "access_key" in data
        assert "secret_key" in data

    def test_validate_success(self):
        """Test validation passes for valid config."""
        config = MinIOConfig()
        config.validate()  # Should not raise

    def test_validate_empty_host(self):
        """Test validation fails for empty host."""
        config = MinIOConfig(host="")
        with pytest.raises(ValueError, match="host cannot be empty"):
            config.validate()

    def test_validate_invalid_port(self):
        """Test validation fails for invalid port."""
        config = MinIOConfig(port=0)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

        config = MinIOConfig(port=70000)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_validate_invalid_quality(self):
        """Test validation fails for invalid JPEG quality."""
        config = MinIOConfig(jpeg_quality=0)
        with pytest.raises(ValueError, match="jpeg_quality must be between"):
            config.validate()

        config = MinIOConfig(jpeg_quality=101)
        with pytest.raises(ValueError, match="jpeg_quality must be between"):
            config.validate()

    def test_validate_empty_credentials(self):
        """Test validation fails for empty credentials."""
        config = MinIOConfig(access_key="")
        with pytest.raises(ValueError, match="access_key cannot be empty"):
            config.validate()

        config = MinIOConfig(secret_key="")
        with pytest.raises(ValueError, match="secret_key cannot be empty"):
            config.validate()

    def test_endpoint_property(self):
        """Test endpoint property returns host:port."""
        config = MinIOConfig(host="myhost", port=9000)
        assert config.endpoint == "myhost:9000"

    def test_public_endpoint_fallback(self):
        """Test public_endpoint falls back to endpoint when not set."""
        config = MinIOConfig(host="myhost", port=9000)
        assert config.public_endpoint == "myhost:9000"

        config = MinIOConfig(
            host="myhost", port=9000, public_host="public", public_port=80
        )
        assert config.public_endpoint == "public:80"


class TestMinIOStatistics:
    """Tests for MinIOStatistics dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = MinIOStatistics()

        assert stats.uploads_total == 0
        assert stats.uploads_success == 0
        assert stats.uploads_failed == 0
        assert stats.bytes_uploaded == 0
        assert stats.images_uploaded == 0
        assert stats.videos_uploaded == 0

    def test_success_rate_zero_uploads(self):
        """Test success rate with no uploads."""
        stats = MinIOStatistics()
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Test success rate with all successful uploads."""
        stats = MinIOStatistics(uploads_total=10, uploads_success=10)
        assert stats.success_rate == 1.0

    def test_success_rate_partial(self):
        """Test success rate with partial success."""
        stats = MinIOStatistics(
            uploads_total=10, uploads_success=7, uploads_failed=3
        )
        assert stats.success_rate == 0.7

    def test_reset(self):
        """Test statistics reset."""
        stats = MinIOStatistics(
            uploads_total=100,
            uploads_success=80,
            uploads_failed=20,
            bytes_uploaded=1000000,
        )
        stats.reset()

        assert stats.uploads_total == 0
        assert stats.uploads_success == 0
        assert stats.uploads_failed == 0
        assert stats.bytes_uploaded == 0

    def test_copy(self):
        """Test statistics snapshot copy."""
        stats = MinIOStatistics(
            uploads_total=50, uploads_success=45, bytes_uploaded=500000
        )
        copy = stats.copy()

        # Values should be equal
        assert copy.uploads_total == 50
        assert copy.uploads_success == 45
        assert copy.bytes_uploaded == 500000

        # Modifying copy should not affect original
        copy.uploads_total = 100
        assert stats.uploads_total == 50


class TestUploadTask:
    """Tests for UploadTask dataclass."""

    def test_create_image_task(self):
        """Test creating an image upload task."""
        task = UploadTask(
            object_name="test/image.jpg",
            upload_type=UploadType.IMAGE,
            data=b"fake image data",
            content_type="image/jpeg",
        )

        assert task.object_name == "test/image.jpg"
        assert task.upload_type == UploadType.IMAGE
        assert task.content_type == "image/jpeg"
        assert len(task.task_id) == 16  # UUID hex[:16]
        assert task.created_at > 0

    def test_create_video_task(self):
        """Test creating a video upload task."""
        task = UploadTask(
            object_name="test/video.mp4",
            upload_type=UploadType.VIDEO,
            data=b"fake video data",
            content_type="video/mp4",
        )

        assert task.upload_type == UploadType.VIDEO
        assert task.content_type == "video/mp4"


class TestUploadResult:
    """Tests for UploadResult dataclass."""

    def test_success_result(self):
        """Test successful upload result."""
        result = UploadResult(
            success=True,
            object_name="test/image.jpg",
            url="http://host:9000/bucket/test/image.jpg",
            upload_time=0.5,
            file_size=1024,
        )

        assert result.success is True
        assert result.url is not None
        assert result.error is None

    def test_failure_result(self):
        """Test failed upload result."""
        result = UploadResult(
            success=False,
            object_name="test/image.jpg",
            error="Connection refused",
        )

        assert result.success is False
        assert result.url is None
        assert result.error == "Connection refused"

    def test_to_dict(self):
        """Test result serialization."""
        result = UploadResult(
            success=True,
            object_name="test.jpg",
            url="http://host/test.jpg",
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["object_name"] == "test.jpg"
        assert data["url"] == "http://host/test.jpg"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_build_object_url_http(self):
        """Test HTTP URL building."""
        url = build_object_url(
            "192.168.2.234", 9000, "algorithm", "test/image.jpg"
        )
        assert url == "http://192.168.2.234:9000/algorithm/test/image.jpg"

    def test_build_object_url_https(self):
        """Test HTTPS URL building."""
        url = build_object_url(
            "192.168.2.234", 9000, "algorithm", "test/image.jpg", secure=True
        )
        assert url == "https://192.168.2.234:9000/algorithm/test/image.jpg"

    def test_get_content_type_images(self):
        """Test content type detection for images."""
        assert get_content_type("photo.jpg") == "image/jpeg"
        assert get_content_type("photo.jpeg") == "image/jpeg"
        assert get_content_type("image.png") == "image/png"
        assert get_content_type("image.gif") == "image/gif"
        assert get_content_type("image.webp") == "image/webp"

    def test_get_content_type_videos(self):
        """Test content type detection for videos."""
        assert get_content_type("video.mp4") == "video/mp4"
        assert get_content_type("video.avi") == "video/x-msvideo"
        assert get_content_type("video.mov") == "video/quicktime"
        assert get_content_type("video.webm") == "video/webm"

    def test_get_content_type_unknown(self):
        """Test content type for unknown extension."""
        assert get_content_type("file.xyz") == "application/octet-stream"
        assert get_content_type("noextension") == "application/octet-stream"

    def test_generate_object_name(self):
        """Test object name generation."""
        name = generate_object_name(prefix="events", extension="jpg")

        # Should include prefix and date
        assert name.startswith("events/")
        assert name.endswith(".jpg")
        # Date format: YYYY/MM/DD
        parts = name.split("/")
        assert len(parts) == 5  # events/YYYY/MM/DD/uuid.jpg

    def test_generate_object_name_no_date(self):
        """Test object name generation without date."""
        name = generate_object_name(
            prefix="static", extension="png", include_date=False
        )

        assert name.startswith("static/")
        assert name.endswith(".png")
        parts = name.split("/")
        assert len(parts) == 2  # static/uuid.png

    def test_generate_timestamped_name(self):
        """Test timestamped name generation."""
        name = generate_timestamped_name(
            prefix="detections", extension="jpg", task_id="001", track_id=5
        )

        assert name.startswith("detections/")
        assert "task001" in name
        assert "track0005" in name
        assert name.endswith(".jpg")


class TestMinIOClientUnit:
    """Unit tests for MinIOClient (no network required)."""

    def test_client_requires_minio_library(self):
        """Test that client checks for minio library."""
        # This would require reimporting the module
        pass  # Skip as it's complex to test import errors

    def test_client_default_config(self):
        """Test client uses default config when None provided."""
        client = MinIOClient()
        assert client._config.host == "easyair-minio"
        assert client._config.port == 9000
            

    def test_upload_image_when_not_running(self):
        """Test upload_image returns None when not running."""
        client = MinIOClient()
        # Not started, so not running
        assert client.is_running() is False

        # Create a fake image
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = client.upload_image(fake_image, "test.jpg")
        assert result is None

    def test_get_statistics(self):
        """Test statistics retrieval."""
        client = MinIOClient()
        stats = client.get_statistics()
 
        assert isinstance(stats, MinIOStatistics)
        assert stats.uploads_total == 0

    def test_get_queue_size(self):
        """Test queue size retrieval."""
        client = MinIOClient()
        assert client.get_queue_size() == 0

    def test_get_object_url(self):
        """Test URL generation."""
        config = MinIOConfig(
            host="192.168.2.234",
            port=9000,
            bucket_name="algorithm",
        )
        client = MinIOClient(config)
        url = client.get_object_url("test/image.jpg")

        assert url == "http://192.168.2.234:9000/algorithm/test/image.jpg"


# ============================================================================
# Integration Tests - Require actual MinIO server
# ============================================================================


@pytest.mark.integration
class TestMinIOClientIntegration:
    """Integration tests (requires MinIO server at 192.168.2.234:9000)."""

    @pytest.fixture
    def config(self):
        """Create test configuration for MinIO server."""
        return MinIOConfig(
            host="192.168.2.234",
            port=9000,
            public_host="192.168.2.234",
            public_port=9000,
        )

    def test_upload_image(self, config):
        """Test actual image upload.

        Test file: datasets/images/000000000009.jpg
        Target: http://192.168.2.234:9000/algorithm/test/2025/12/24/
                tiga-ultralytics-system-minio-upload-image.jpg
        """
        # Load test image
        image_path = Path("datasets/images/000000000009.jpg")
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            pytest.skip("Failed to load test image")

        object_name = "test/2025/12/24/tiga-ultralytics-system-minio-upload-image.jpg"
        expected_url = f"http://192.168.2.234:9000/algorithm/{object_name}"

        # Track callback invocation
        callback_results = []

        def on_success(result):
            callback_results.append(result)

        with MinIOClient(config, on_success=on_success) as client:
            url = client.upload_image(image, object_name)

            # Verify URL returned immediately
            assert url == expected_url

            # Wait for async upload to complete
            time.sleep(3)

            # Check statistics
            stats = client.get_statistics()
            assert stats.uploads_success >= 1
            assert stats.images_uploaded >= 1

        # Verify callback was invoked
        assert len(callback_results) >= 1
        assert callback_results[0].success is True
        assert callback_results[0].url == expected_url

        print(f"Image uploaded successfully: {url}")

    def test_upload_video(self, config):
        """Test actual video upload.

        Test file: datasets/videos/cross_line.mp4
        Target: http://192.168.2.234:9000/algorithm/test/2025/12/24/
                tiga-ultralytics-system-minio-upload-video.mp4
        """
        video_path = Path("datasets/videos/cross_line.mp4")
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        object_name = "test/2025/12/24/tiga-ultralytics-system-minio-upload-video.mp4"
        expected_url = f"http://192.168.2.234:9000/algorithm/{object_name}"

        # Track callback invocation
        callback_results = []

        def on_success(result):
            callback_results.append(result)

        with MinIOClient(config, on_success=on_success) as client:
            url = client.upload_video(str(video_path), object_name)

            # Verify URL returned immediately
            assert url == expected_url

            # Wait for async upload (video is larger)
            time.sleep(10)

            # Check statistics
            stats = client.get_statistics()
            assert stats.uploads_success >= 1
            assert stats.videos_uploaded >= 1

        # Verify callback was invoked
        assert len(callback_results) >= 1
        assert callback_results[0].success is True
        assert callback_results[0].url == expected_url

        print(f"Video uploaded successfully: {url}")

    def test_callback_invocation(self, config):
        """Test success/failure callbacks are invoked."""
        success_calls = []
        failure_calls = []

        def on_success(result):
            success_calls.append(result)

        def on_failure(result):
            failure_calls.append(result)

        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [0, 0, 255]  # Red image

        with MinIOClient(
            config, on_success=on_success, on_failure=on_failure
        ) as client:
            object_name = f"test/callback_test_{time.time()}.jpg"
            url = client.upload_image(test_image, object_name)

            assert url is not None

            # Wait for upload
            time.sleep(2)

        # Should have success callback
        assert len(success_calls) >= 1
        assert success_calls[0].success is True

    def test_statistics_tracking(self, config):
        """Test statistics are properly tracked."""
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)

        with MinIOClient(config) as client:
            # Initial stats
            stats = client.get_statistics()
            initial_total = stats.uploads_total

            # Upload multiple images
            for i in range(3):
                client.upload_image(test_image, f"test/stats_test_{i}.jpg")

            # Wait for uploads
            time.sleep(5)

            # Check stats increased
            stats = client.get_statistics()
            assert stats.uploads_total >= initial_total + 3
            assert stats.success_rate > 0

    def test_context_manager_lifecycle(self, config):
        """Test client lifecycle with context manager."""
        with MinIOClient(config) as client:
            assert client.is_running() is True
            assert client.is_connected() is True

        # After context, should be stopped
        assert client.is_running() is False

    def test_queue_size_tracking(self, config):
        """Test queue size is tracked correctly."""
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        with MinIOClient(config) as client:
            # Queue should start empty
            assert client.get_queue_size() == 0

            # Add items quickly
            for i in range(5):
                client.upload_image(test_image, f"test/queue_{i}.jpg")

            # Queue might have items (depending on timing)
            # This is a basic sanity check
            initial_size = client.get_queue_size()

            # Wait for processing
            time.sleep(5)

            # Queue should drain
            final_size = client.get_queue_size()
            assert final_size <= initial_size


if __name__ == "__main__":
    # Run unit tests by default
    pytest.main([__file__, "-v", "-m", "not integration"])
