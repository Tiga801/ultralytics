"""Kalman filter implementations for object tracking.

This module provides Kalman filter classes for tracking bounding boxes in image space.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """A Kalman filter for tracking bounding boxes in image space using XYAH format.

    Implements a simple Kalman filter for tracking bounding boxes in image space. The 8-dimensional
    state space (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y),
    aspect ratio a, height h, and their respective velocities. Object motion follows a constant
    velocity model, and bounding box location (x, y, a, h) is taken as a direct observation of
    the state space (linear observation model).

    Attributes:
        _motion_mat: The motion matrix for the Kalman filter.
        _update_mat: The update matrix for the Kalman filter.
        _std_weight_position: Standard deviation weight for position.
        _std_weight_velocity: Standard deviation weight for velocity.
    """

    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a track from an unassociated measurement.

        Args:
            measurement: Bounding box coordinates (x, y, a, h) with center position (x, y),
                aspect ratio a, and height h.

        Returns:
            Tuple of (mean, covariance) for the new track.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Args:
            mean: The 8-dimensional mean vector of the object state at the previous time step.
            covariance: The 8x8-dimensional covariance matrix of the object state.

        Returns:
            Tuple of (mean, covariance) for the predicted state.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space.

        Args:
            mean: The state's mean vector (8 dimensional array).
            covariance: The state's covariance matrix (8x8 dimensional).

        Returns:
            Tuple of projected (mean, covariance).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step for multiple object states (vectorized).

        Args:
            mean: The Nx8 dimensional mean matrix of the object states.
            covariance: The Nx8x8 covariance matrix of the object states.

        Returns:
            Tuple of (mean, covariance) matrices for the predicted states.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step.

        Args:
            mean: The predicted state's mean vector (8 dimensional).
            covariance: The state's covariance matrix (8x8 dimensional).
            measurement: The 4 dimensional measurement vector (x, y, a, h).

        Returns:
            Tuple of measurement-corrected (mean, covariance).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Compute gating distance between state distribution and measurements.

        Args:
            mean: Mean vector over the state distribution (8 dimensional).
            covariance: Covariance of the state distribution (8x8 dimensional).
            measurements: An (N, 4) matrix of N measurements.
            only_position: If True, distance is computed with respect to box center only.
            metric: The metric to use ('gaussian' or 'maha').

        Returns:
            Array of length N with squared distances.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """A Kalman filter for tracking bounding boxes in image space using XYWH format.

    Implements a Kalman filter for tracking bounding boxes with state space (x, y, w, h, vx, vy, vw, vh),
    where (x, y) is the center position, w is the width, h is the height, and vx, vy, vw, vh are their
    respective velocities.
    """

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create track from unassociated measurement.

        Args:
            measurement: Bounding box coordinates (x, y, w, h) with center position (x, y),
                width, and height.

        Returns:
            Tuple of (mean, covariance) for the new track.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Args:
            mean: The 8-dimensional mean vector of the object state.
            covariance: The 8x8-dimensional covariance matrix.

        Returns:
            Tuple of (mean, covariance) for the predicted state.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space.

        Args:
            mean: The state's mean vector (8 dimensional array).
            covariance: The state's covariance matrix (8x8 dimensional).

        Returns:
            Tuple of projected (mean, covariance).
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step (vectorized).

        Args:
            mean: The Nx8 dimensional mean matrix of the object states.
            covariance: The Nx8x8 covariance matrix of the object states.

        Returns:
            Tuple of (mean, covariance) matrices for the predicted states.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step.

        Args:
            mean: The predicted state's mean vector (8 dimensional).
            covariance: The state's covariance matrix (8x8 dimensional).
            measurement: The 4 dimensional measurement vector (x, y, w, h).

        Returns:
            Tuple of measurement-corrected (mean, covariance).
        """
        return super().update(mean, covariance, measurement)
