"""Inference manager for centralized model management.

This module provides the InferenceManager class that handles model loading,
caching, and inference queue management.
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from utils import SingletonMeta, Logger
from .config import get_engine_config
from .exceptions import ModelNotFoundException, ModelLoadException

if TYPE_CHECKING:
    from models import Predictor


class InferenceManager(metaclass=SingletonMeta):
    """Centralized inference resource manager (Singleton).

    This class manages model loading and caching to optimize resource usage.
    Each task process loads its own model (for GPU isolation), but this
    manager tracks loaded models and provides configuration.

    Note: In the process-based architecture, each task process loads its
    own model instance for GPU resource isolation. This manager primarily
    serves for tracking and configuration purposes.
    """

    def __init__(self):
        """Initialize inference manager."""
        self._predictors: Dict[str, "Predictor"] = {}
        self._model_configs: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self.log = Logger.get_logging_method("INFERENCE")
        self._config = get_engine_config()

    def get_predictor(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ) -> "Predictor":
        """Get or create a predictor instance.

        This method is primarily used for shared inference scenarios.
        In the process-based architecture, each task creates its own
        predictor in its subprocess.

        Args:
            model_name: Model identifier.
            model_path: Path to model weights.
            device: Inference device.
            **kwargs: Additional predictor arguments.

        Returns:
            Predictor instance.

        Raises:
            ModelNotFoundException: If model file not found.
            ModelLoadException: If model fails to load.
        """
        with self._lock:
            # Check cache
            cache_key = f"{model_name}_{device}"
            if cache_key in self._predictors:
                return self._predictors[cache_key]

            # Resolve model path
            if model_path is None:
                model_path = self._config.get_model_path(model_name)

            # Verify model exists
            if not Path(model_path).exists():
                raise ModelNotFoundException(model_name, model_path)

            # Load model
            try:
                from models import Predictor
                predictor = Predictor(
                    model_path=model_path,
                    device=device,
                    conf=kwargs.get("conf", self._config.default_confidence),
                    iou=kwargs.get("iou", self._config.default_iou),
                )
                self._predictors[cache_key] = predictor
                self.log(f"Loaded model: {model_name} on {device}")
                return predictor

            except Exception as e:
                raise ModelLoadException(model_name, str(e))

    def register_model(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda",
        **kwargs
    ) -> None:
        """Register a model configuration.

        This stores the configuration for a model without loading it.
        Useful for task processes to get model configuration.

        Args:
            model_name: Model identifier.
            model_path: Path to model weights.
            device: Inference device.
            **kwargs: Additional configuration.
        """
        with self._lock:
            self._model_configs[model_name] = {
                "model_path": model_path,
                "device": device,
                **kwargs
            }
            self.log(f"Registered model config: {model_name}")

    def get_model_config(self, model_name: str) -> Optional[Dict]:
        """Get model configuration.

        Args:
            model_name: Model identifier.

        Returns:
            Model configuration dictionary or None.
        """
        with self._lock:
            return self._model_configs.get(model_name)

    def unload_predictor(self, model_name: str, device: str = "cuda") -> bool:
        """Unload a predictor from cache.

        Args:
            model_name: Model identifier.
            device: Device the model was loaded on.

        Returns:
            True if unloaded, False if not found.
        """
        with self._lock:
            cache_key = f"{model_name}_{device}"
            if cache_key in self._predictors:
                del self._predictors[cache_key]
                self.log(f"Unloaded model: {model_name}")
                return True
            return False

    def unload_all(self) -> None:
        """Unload all cached predictors."""
        with self._lock:
            self._predictors.clear()
            self.log("Unloaded all models")

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names.

        Returns:
            List of cache keys for loaded models.
        """
        with self._lock:
            return list(self._predictors.keys())

    def is_model_loaded(self, model_name: str, device: str = "cuda") -> bool:
        """Check if a model is loaded.

        Args:
            model_name: Model identifier.
            device: Device to check.

        Returns:
            True if model is loaded.
        """
        with self._lock:
            cache_key = f"{model_name}_{device}"
            return cache_key in self._predictors

    def get_model_path(self, model_name: str) -> str:
        """Get full path to a model file.

        Args:
            model_name: Model filename.

        Returns:
            Full path to model.
        """
        # Check registered configs first
        config = self.get_model_config(model_name)
        if config and "model_path" in config:
            return config["model_path"]

        # Fall back to default path
        return self._config.get_model_path(model_name)

    def get_status(self) -> Dict:
        """Get manager status.

        Returns:
            Status dictionary.
        """
        with self._lock:
            return {
                "loaded_models": list(self._predictors.keys()),
                "registered_configs": list(self._model_configs.keys()),
                "model_count": len(self._predictors),
            }

    def __repr__(self) -> str:
        """String representation."""
        return f"InferenceManager(models={len(self._predictors)})"
