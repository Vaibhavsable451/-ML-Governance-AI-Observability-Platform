"""Metrics: in-memory rolling window (for the UI) + buffered CloudWatch publishing."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np

from core import config

log = logging.getLogger("metrics")


class MetricsRecorder:
    def __init__(self, maxlen: int = 5000, flush_size: int = 20):
        self._lock = threading.Lock()
        self._records = deque(maxlen=maxlen)
        self._buffer: list[dict] = []
        self._flush_size = flush_size
        self._cw = None

    def _client(self):
        if self._cw is None:
            import boto3
            self._cw = boto3.client("cloudwatch", region_name=config.AWS_REGION)
        return self._cw

    def record(self, name: str, value: float, unit: str = "None", dimensions: dict | None = None) -> None:
        item = {"ts": time.time(), "name": name, "value": float(value), "unit": unit, "dimensions": dimensions or {}}
        with self._lock:
            self._records.append(item)
            if config.USE_CLOUDWATCH:
                self._buffer.append(item)
                if len(self._buffer) >= self._flush_size:
                    self._flush_locked()

    def _flush_locked(self) -> None:
        batch, self._buffer = self._buffer, []
        if not batch:
            return
        try:
            data = [{"MetricName": m["name"], "Value": m["value"], "Unit": m["unit"],
                     "Dimensions": [{"Name": k, "Value": str(v)} for k, v in m["dimensions"].items()]}
                    for m in batch]
            self._client().put_metric_data(Namespace=config.CLOUDWATCH_NAMESPACE, MetricData=data)
        except Exception as exc:  # metrics must never break requests
            log.warning("cloudwatch flush failed: %s", exc)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    @contextmanager
    def timer(self, name: str, dimensions: dict | None = None):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - t0) * 1000, "Milliseconds", dimensions)

    def summary(self) -> dict:
        with self._lock:
            recs = list(self._records)
        groups = defaultdict(list)
        for r in recs:
            groups[r["name"]].append(r["value"])
        out = {}
        for name, vals in groups.items():
            arr = np.asarray(vals)
            out[name] = {"count": len(vals), "avg": round(float(arr.mean()), 2),
                         "p95": round(float(np.percentile(arr, 95)), 2), "max": round(float(arr.max()), 2),
                         "sum": round(float(arr.sum()), 4)}
        return out

    def series(self, name: str, limit: int = 200) -> list[dict]:
        with self._lock:
            return [{"ts": r["ts"], "value": r["value"]} for r in self._records if r["name"] == name][-limit:]


metrics = MetricsRecorder()
