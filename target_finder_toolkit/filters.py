import time

try:
    from OneEuroFilter import OneEuroFilter
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    OneEuroFilter = None
    _ONE_EURO_IMPORT_ERROR = exc
else:
    _ONE_EURO_IMPORT_ERROR = None


FILTER_OPTIONS = {
    "none": "None",
    "one_euro": "One Euro",
}


class OneEuroFilter1D:
    def __init__(
        self,
        *,
        freq: float = 120.0,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        if OneEuroFilter is None:
            raise RuntimeError(
                "OneEuroFilter is not available. Install it with `pip install OneEuroFilter --upgrade`."
            ) from _ONE_EURO_IMPORT_ERROR
        self._freq = float(freq)
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._filter = OneEuroFilter(
            freq=self._freq,
            mincutoff=self._min_cutoff,
            beta=self._beta,
            dcutoff=self._d_cutoff,
        )
        self._last_value = None

    def filter(self, value: float, timestamp: float | None = None) -> float:
        now = time.monotonic() if timestamp is None else float(timestamp)
        filtered = self._filter.filter(float(value), now)
        self._last_value = float(filtered)
        return self._last_value

    def reset(self, value: float, timestamp: float | None = None) -> float:
        now = time.monotonic() if timestamp is None else float(timestamp)
        self._filter = OneEuroFilter(
            freq=self._freq,
            mincutoff=self._min_cutoff,
            beta=self._beta,
            dcutoff=self._d_cutoff,
        )
        self._last_value = float(value)
        # Re-prime the official filter at the current timestamp so the next
        # sample does not reuse stale time or derivative state.
        self._filter.filter(float(value), now)
        return self._last_value


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

    def reset(self, x: float, y: float, timestamp: float | None = None) -> tuple[float, float]:
        if not self.enabled:
            return float(x), float(y)
        return self._fx.reset(x, timestamp), self._fy.reset(y, timestamp)
