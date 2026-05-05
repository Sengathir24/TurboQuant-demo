# metrics.py
from collections import deque
import time

class MetricsCollector:
    def __init__(self, window=100):
        self.tps_history = deque(maxlen=window)
        self.mse_history = deque(maxlen=window)
        self.lat_history = deque(maxlen=window)

    def record(self, tps, mse, lat_ms):
        self.tps_history.append((time.time(), tps))
        self.mse_history.append((time.time(), mse))
        self.lat_history.append((time.time(), lat_ms))

    def summary(self):
        return {
            "tps": list(self.tps_history),
            "mse": list(self.mse_history),
            "latency_ms": list(self.lat_history),
        }
