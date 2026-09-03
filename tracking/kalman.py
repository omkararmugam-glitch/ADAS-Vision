"""Constant-velocity Kalman filter over a bounding box.

State is ``(cx, cy, a, h, vcx, vcy, va, vh)``: centre, aspect ratio ``w/h``,
height, and their velocities. Tracking height rather than width is deliberate --
for a vehicle ahead, height is the dimension that grows cleanly as it approaches,
while width changes with yaw as the vehicle turns. The time-to-collision
calculation in :mod:`tracking.ttc` reads the height rate directly, so a noisy
height estimate becomes a noisy TTC.

Process and measurement noise scale with the box height, which is the standard
SORT/ByteTrack formulation: a box 20 px tall is located far less precisely, in
absolute pixels, than one 300 px tall, and fixed noise would either over-trust
the distant box or under-trust the near one.
"""

from __future__ import annotations

import numpy as np

#: Chi-square 0.95 quantile for 4 degrees of freedom, for gating by Mahalanobis
#: distance. Measurements beyond this are not plausibly the tracked object.
CHI2_INV_95_4DOF = 9.4877

#: Standard deviation of the velocity prior on a brand-new track, as a fraction
#: of the box height. At birth the velocity is genuinely unknown, and a tight
#: prior here silently drops fast-moving objects on their second frame.
INITIAL_VELOCITY_FRACTION = 0.25


class KalmanBoxFilter:
    """Linear Kalman filter for one box track.

    The model is deliberately simple -- constant velocity, no acceleration term.
    A dashcam runs at 25-30 fps, so between frames a vehicle's apparent motion is
    near-linear, and an acceleration state would mostly fit noise.
    """

    def __init__(self, std_position: float = 1.0 / 20, std_velocity: float = 1.0 / 160) -> None:
        ndim, dt = 4, 1.0
        # x' = F x : position gains its velocity each step, velocity persists.
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        # z = H x : only the box itself is measured, never its velocity.
        self._H = np.eye(ndim, 2 * ndim)
        self._std_pos = std_position
        self._std_vel = std_velocity

    # ------------------------------------------------------------------ #
    # noise
    # ------------------------------------------------------------------ #

    def _process_noise(self, mean: np.ndarray) -> np.ndarray:
        h = mean[3]
        std = [
            self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
            self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h,
        ]
        return np.diag(np.square(std))

    def _measurement_noise(self, mean: np.ndarray) -> np.ndarray:
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]
        return np.diag(np.square(std))

    # ------------------------------------------------------------------ #
    # filter
    # ------------------------------------------------------------------ #

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a track from one box, with velocity unknown and so uncertain.

        Args:
            measurement: ``(cx, cy, a, h)``.

        Returns:
            ``(mean, covariance)`` of the new state.
        """
        mean = np.concatenate([np.asarray(measurement, dtype=np.float64), np.zeros(4)])
        h = mean[3]
        # Velocity is not measured, so its initial variance must be genuinely
        # large. The usual SORT/DeepSORT prior (a fraction of h/160) encodes an
        # assumption of slow motion, and a track born on a fast-crossing vehicle
        # is then so over-confident that its *second* frame falls outside the
        # gate and the object is dropped.
        #
        # INITIAL_VELOCITY_FRACTION is the honest prior instead: an object can
        # move up to a fair fraction of its own box size between frames. Beyond
        # about one box height per frame there is no overlap left for IoU to
        # associate on, so this spans everything the tracker could match anyway.
        vel_std = INITIAL_VELOCITY_FRACTION * h
        std = [
            2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
            vel_std, vel_std, 1e-5, vel_std,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance the state one frame."""
        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + self._process_noise(mean)
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map the state into measurement space, adding measurement noise."""
        return self._H @ mean, self._H @ covariance @ self._H.T + self._measurement_noise(mean)

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with an observed box."""
        proj_mean, proj_cov = self.project(mean, covariance)
        # Solve rather than invert: proj_cov is small but can be ill-conditioned
        # for a nearly-static box, and an explicit inverse loses precision there.
        kalman_gain = np.linalg.solve(proj_cov.T, (covariance @ self._H.T).T).T
        innovation = np.asarray(measurement, dtype=np.float64) - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Squared Mahalanobis distance from the state to each measurement."""
        proj_mean, proj_cov = self.project(mean, covariance)
        d = np.atleast_2d(measurements) - proj_mean
        chol = np.linalg.cholesky(proj_cov)
        z = np.linalg.solve(chol, d.T)
        return np.sum(z * z, axis=0)


# --------------------------------------------------------------------------- #
# box conversions
# --------------------------------------------------------------------------- #


def xyxy_to_cxah(box: np.ndarray) -> np.ndarray:
    """``(x0, y0, x1, y1)`` -> ``(cx, cy, aspect, height)``."""
    b = np.asarray(box, dtype=np.float64)
    w = b[..., 2] - b[..., 0]
    h = b[..., 3] - b[..., 1]
    cx = b[..., 0] + w / 2.0
    cy = b[..., 1] + h / 2.0
    # A zero-height box would make aspect undefined and poison the filter.
    h_safe = np.maximum(h, 1e-6)
    return np.stack([cx, cy, w / h_safe, h], axis=-1)


def cxah_to_xyxy(state: np.ndarray) -> np.ndarray:
    """``(cx, cy, aspect, height)`` -> ``(x0, y0, x1, y1)``."""
    s = np.asarray(state, dtype=np.float64)
    h = s[..., 3]
    w = s[..., 2] * h
    return np.stack([
        s[..., 0] - w / 2.0, s[..., 1] - h / 2.0,
        s[..., 0] + w / 2.0, s[..., 1] + h / 2.0,
    ], axis=-1)
