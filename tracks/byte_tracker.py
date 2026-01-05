"""BYTETracker implementation for multi-object tracking.

This module provides the BYTETracker class for tracking multiple objects in video frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import TrackState
from .config import TrackerConfig, config_to_namespace
from .kalman_filter import KalmanFilterXYAH
from .strack import STrack
from . import matching


class BYTETracker:
    """BYTETracker: A tracking algorithm for object detection and tracking.

    This class encapsulates the functionality for initializing, updating, and managing
    the tracks for detected objects in a video sequence. It maintains the state of
    tracked, lost, and removed tracks over frames, utilizes Kalman filtering for
    predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks: List of successfully activated tracks.
        lost_stracks: List of lost tracks.
        removed_stracks: List of removed tracks.
        frame_id: The current frame ID.
        args: Configuration parameters.
        max_time_lost: The maximum frames for a track to be considered as 'lost'.
        kalman_filter: Kalman Filter object.
    """

    def __init__(self, args: TrackerConfig | Any, frame_rate: int = 30):
        """Initialize a BYTETracker instance for object tracking.

        Args:
            args: TrackerConfig or namespace containing tracking parameters.
            frame_rate: Frame rate of the video sequence.
        """
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0

        # Handle both TrackerConfig and namespace-like objects
        if isinstance(args, TrackerConfig):
            self.args = config_to_namespace(args)
        else:
            self.args = args

        self.max_time_lost = int(frame_rate / 30.0 * self.args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update the tracker with new detections and return the current list of tracked objects.

        Args:
            results: Detection results object with `conf`, `cls`, and `xywh` attributes,
                or numpy array of shape (N, 6) with columns [x, y, w, h, conf, cls].
            img: Optional image array for GMC (used in BoTSORT).
            feats: Optional feature vectors for ReID (used in BoTSORT).

        Returns:
            Array of tracked objects with shape (M, 7+) containing
            [x1, y1, x2, y2, track_id, score, class, idx].
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Handle different input formats
        if hasattr(results, 'conf'):
            scores = np.asarray(results.conf)
        else:
            scores = results[:, 4] if results.shape[1] > 4 else np.ones(len(results))

        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = self._filter_results(results, inds_second)
        results_first = self._filter_results(results, remain_inds)

        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results_first, feats_keep)

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        self.multi_predict(strack_pool)

        # Apply GMC if available
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, self._get_xyxy(results_first))
            except Exception:
                warp = np.eye(2, 3)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association, with low score detection boxes
        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def _filter_results(self, results, mask: np.ndarray):
        """Filter results by boolean mask.

        Args:
            results: Detection results (object or array).
            mask: Boolean mask for filtering.

        Returns:
            Filtered results.
        """
        if hasattr(results, '__getitem__') and hasattr(results, 'conf'):
            # Object with indexing support
            return results[mask]
        elif isinstance(results, np.ndarray):
            return results[mask]
        else:
            return results

    def _get_xyxy(self, results) -> np.ndarray | None:
        """Get xyxy bounding boxes from results.

        Args:
            results: Detection results.

        Returns:
            xyxy bounding boxes or None.
        """
        if hasattr(results, 'xyxy'):
            return np.asarray(results.xyxy)
        elif isinstance(results, np.ndarray) and len(results) > 0:
            # Convert xywh to xyxy
            xywh = results[:, :4]
            xyxy = np.empty_like(xywh)
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
            return xyxy
        return None

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """Initialize object tracking with given detections, scores, and class labels.

        Args:
            results: Detection results object or numpy array.
            img: Optional image/features (for subclass override).

        Returns:
            List of STrack objects.
        """
        if hasattr(results, 'conf'):
            # Object with attributes
            if len(results) == 0:
                return []
            bboxes = np.asarray(results.xywh if hasattr(results, 'xywh') else results[:, :4])
            bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
            conf = np.asarray(results.conf)
            cls = np.asarray(results.cls)
            return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, conf, cls)]
        elif isinstance(results, np.ndarray):
            # Numpy array format [x, y, w, h, conf, cls]
            if len(results) == 0:
                return []
            bboxes = np.concatenate([results[:, :4], np.arange(len(results)).reshape(-1, 1)], axis=-1)
            conf = results[:, 4]
            cls = results[:, 5] if results.shape[1] > 5 else np.zeros(len(results))
            return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, conf, cls)]
        return []

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[STrack]):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Filter out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
