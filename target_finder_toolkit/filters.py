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

DEFAULT_FILTER_FREQ = 120.0
DEFAULT_FILTER_MIN_CUTOFF = 1.0
DEFAULT_FILTER_BETA = 0.02
DEFAULT_FILTER_D_CUTOFF = 1.0


def add_filter_arguments(parser):
    parser.add_argument("--filter", choices=sorted(FILTER_OPTIONS.keys()), default="none", help="Optional pointer filter")
    parser.add_argument("--filter-freq", type=float, default=DEFAULT_FILTER_FREQ, help="One Euro filter sampling frequency")
    parser.add_argument("--filter-min-cutoff", type=float, default=DEFAULT_FILTER_MIN_CUTOFF, help="One Euro filter minimum cutoff")
    parser.add_argument("--filter-beta", type=float, default=DEFAULT_FILTER_BETA, help="One Euro filter beta")
    parser.add_argument("--filter-d-cutoff", type=float, default=DEFAULT_FILTER_D_CUTOFF, help="One Euro filter derivative cutoff")


def filter_kwargs_from_args(args) -> dict[str, float]:
    return {
        "freq": args.filter_freq,
        "min_cutoff": args.filter_min_cutoff,
        "beta": args.filter_beta,
        "d_cutoff": args.filter_d_cutoff,
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
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        if self.filter_name == "one_euro":
            kwargs = {
                "freq": self.freq,
                "min_cutoff": self.min_cutoff,
                "beta": self.beta,
                "d_cutoff": self.d_cutoff,
            }
            self._fx = OneEuroFilter1D(**kwargs)
            self._fy = OneEuroFilter1D(**kwargs)
        else:
            self._fx = None
            self._fy = None

    @property
    def enabled(self) -> bool:
        return self.filter_name != "none"

    @property
    def params(self) -> dict[str, float]:
        return {
            "filter_freq": self.freq,
            "filter_min_cutoff": self.min_cutoff,
            "filter_beta": self.beta,
            "filter_d_cutoff": self.d_cutoff,
        }

    def filter(self, x: float, y: float, timestamp: float | None = None) -> tuple[float, float]:
        if not self.enabled:
            return float(x), float(y)
        return self._fx.filter(x, timestamp), self._fy.filter(y, timestamp)

    def reset(self, x: float, y: float, timestamp: float | None = None) -> tuple[float, float]:
        if not self.enabled:
            return float(x), float(y)
        return self._fx.reset(x, timestamp), self._fy.reset(y, timestamp)
