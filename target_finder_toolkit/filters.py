import math
import time


FILTER_OPTIONS = {
    "none": "None",
    "one_euro": "One Euro",
}


class LowPassFilter:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._initialized = False
        self._value = 0.0

    def filter(self, value: float, alpha: float | None = None) -> float:
        if alpha is not None:
            self.alpha = float(alpha)
        if not self._initialized:
            self._initialized = True
            self._value = float(value)
            return self._value
        self._value = self.alpha * float(value) + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value


class OneEuroFilter1D:
    def __init__(
        self,
        *,
        freq: float = 120.0,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = LowPassFilter(1.0)
        self._dx = LowPassFilter(1.0)
        self._last_time = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / max(self.freq, 1e-6)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, value: float, timestamp: float | None = None) -> float:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self._last_time is not None:
            dt = max(now - self._last_time, 1e-6)
            self.freq = 1.0 / dt
        self._last_time = now

        prev_x = self._x.value if self._x._initialized else float(value)
        dx = (float(value) - prev_x) * self.freq
        edx = self._dx.filter(dx, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(float(value), self._alpha(cutoff))


class PointFilter2D:
    def __init__(
        self,
        filter_name: str = "none",
        *,
        freq: float = 120.0,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.filter_name = filter_name if filter_name in FILTER_OPTIONS else "none"
        if self.filter_name == "one_euro":
            kwargs = {
                "freq": freq,
                "min_cutoff": min_cutoff,
                "beta": beta,
                "d_cutoff": d_cutoff,
            }
            self._fx = OneEuroFilter1D(**kwargs)
            self._fy = OneEuroFilter1D(**kwargs)
        else:
            self._fx = None
            self._fy = None

    @property
    def enabled(self) -> bool:
        return self.filter_name != "none"

    def filter(self, x: float, y: float, timestamp: float | None = None) -> tuple[float, float]:
        if not self.enabled:
            return float(x), float(y)
        return self._fx.filter(x, timestamp), self._fy.filter(y, timestamp)
