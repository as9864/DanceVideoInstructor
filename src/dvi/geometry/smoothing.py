"""Causal keypoint smoothing for the real-time path (ARCHITECTURE.md §5.4, §11).

Offline analysis can afford Savitzky-Golay (non-causal, needs future
frames); the live runner can't, so it uses the One-Euro filter
(Casiez et al., CHI 2012): a first-order low-pass whose cutoff rises with
speed. Slow drift (standing in an embrace) gets heavy smoothing — jitter
disappears — while fast motion (an ocho, a boleo) gets a high cutoff, so
the skeleton doesn't lag behind the body. `min_cutoff` trades jitter for
lag at rest, `beta` trades lag for jitter in motion.
"""

import numpy as np


def _alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorized One-Euro filter over an arbitrary-shaped array (e.g. (J, 2)).

    Timestamps come from the caller rather than an assumed fixed rate —
    webcam frame intervals jitter, and dropped frames (the live runner
    always takes the *latest* frame) make the gap uneven.

    `mask` lets invalid joints pass through without polluting the filter
    state: a joint that drops out keeps its last filtered value internally
    and restarts cleanly from the raw value when it reappears.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._seen: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._seen = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float, mask: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if mask is None:
            mask = np.ones(x.shape[: x.ndim - 1] if x.ndim > 1 else x.shape, dtype=bool)
        mask_b = mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))
        mask_b = np.broadcast_to(mask_b, x.shape)

        if self._x is None or self._x.shape != x.shape or self._t is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._seen = mask_b.copy()
            self._t = t
            return x.copy()

        dt = t - self._t
        if dt <= 0:
            # duplicate/out-of-order timestamp: don't advance the filter
            return np.where(mask_b, self._x, x)
        self._t = t

        fresh = mask_b & ~self._seen  # joint (re)appearing: restart from the raw value
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x

        x_hat = np.where(fresh, x, x_hat)
        dx_hat = np.where(fresh, 0.0, dx_hat)

        self._x = np.where(mask_b, x_hat, self._x)
        self._dx = np.where(mask_b, dx_hat, self._dx)
        self._seen = mask_b.copy()
        return np.where(mask_b, x_hat, x)
