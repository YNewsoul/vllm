from __future__ import annotations

import threading
from queue import Empty, Full, Queue
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

from vllm.logger import init_logger

try:
    from .multislo_predictor import REQUIRED_KEYS, FitMetrics, MultiSloPredictor
except ImportError:
    from multislo_predictor import REQUIRED_KEYS, FitMetrics, MultiSloPredictor

logger = init_logger(__name__)


@dataclass
class MultiSloOnlineTrainConfig:
    enabled: bool
    buffer_size: int
    warmup_samples: int
    retrain_interval: int
    l2: float
    use_scene_models: bool
    min_scene_samples: int
    min_ms: float
    save_path: str
    ingest_queue_size: int = 4096


class MultiSloOnlineTrainer:
    """在线样本缓冲 + 异步重训练 + 原子热切换。"""
    _SAVE_EVERY_SUCCESSFUL_TRAIN_ROUNDS = 100

    def __init__(
        self,
        initial_predictor: MultiSloPredictor,
        config: MultiSloOnlineTrainConfig,
    ) -> None:
        self._config = config
        self._predictor = initial_predictor
        self._records: deque[dict[str, Any]] = deque(
            maxlen=max(1, int(config.buffer_size))
        )
        self._state_lock = threading.Lock()
        self._ingest_queue: Queue[dict[str, Any]] = Queue(
            maxsize=max(1, int(config.ingest_queue_size))
        )
        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._internal_worker_loop,
            name="MultiSloOnlineWorker",
            daemon=True,
        )
        self._worker_thread.start()

        self._total_seen_samples = 0
        self._dropped_samples = 0
        self._last_train_sample_id = 0
        self._train_round = 0
        self._last_metrics: FitMetrics | None = None
        self._train_thread: threading.Thread | None = None

    def predict(
        self,
        chunk_sizes: Sequence[int],
        cached_tokens: Sequence[int],
        sched_tokens: int,
        computed_tokens: Sequence[int] | None = None,
    ) -> float:
        # 仅原子读取当前模型引用，避免在线路径锁竞争。
        predictor = self._predictor
        return predictor.predict(
            chunk_sizes=chunk_sizes,
            cached_tokens=cached_tokens,
            sched_tokens=sched_tokens,
            computed_tokens=computed_tokens,
        )

    def should_capture_runtime_record(self) -> bool:
        return bool(self._config.enabled)

    def observe_record(self, record: dict[str, Any]) -> None:
        if not self._config.enabled:
            return
        try:
            # 在线路径仅做非阻塞入队；满队列时直接丢样本，避免拖慢推理。
            self._ingest_queue.put_nowait(record)
        except Full:
            with self._state_lock:
                self._dropped_samples += 1

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "enabled": self._config.enabled,
                "buffer_size": len(self._records),
                "buffer_capacity": self._records.maxlen,
                "ingest_queue_size": self._ingest_queue.qsize(),
                "ingest_queue_capacity": self._ingest_queue.maxsize,
                "total_seen_samples": self._total_seen_samples,
                "dropped_samples": self._dropped_samples,
                "train_round": self._train_round,
                "last_metrics": (
                    {
                        "mae": self._last_metrics.mae,
                        "rmse": self._last_metrics.rmse,
                        "r2": self._last_metrics.r2,
                    }
                    if self._last_metrics is not None
                    else None
                ),
            }

    def _internal_worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                record = self._ingest_queue.get(timeout=0.1)
            except Empty:
                continue

            normalized = self._internal_normalize_record(record)
            if normalized is None:
                continue

            with self._state_lock:
                self._records.append(normalized)
                self._total_seen_samples += 1
                total_seen = self._total_seen_samples
                warmup_ok = len(self._records) >= int(self._config.warmup_samples)
                retrain_due = (
                    total_seen - self._last_train_sample_id
                    >= int(self._config.retrain_interval)
                )
                trainer_busy = (
                    self._train_thread is not None and self._train_thread.is_alive()
                )
                if not warmup_ok or not retrain_due or trainer_busy:
                    continue

                snapshot = list(self._records)
                train_round = self._train_round + 1
                self._last_train_sample_id = total_seen
                self._train_thread = threading.Thread(
                    target=self._internal_train_and_swap,
                    args=(snapshot, train_round),
                    name=f"MultiSloOnlineTrainer-{train_round}",
                    daemon=True,
                )
                self._train_thread.start()

    def _internal_train_and_swap(
        self,
        records_snapshot: list[dict[str, Any]],
        train_round: int,
    ) -> None:
        try:
            old_predictor = self._predictor
            predictor = MultiSloPredictor(
                min_ms=max(float(self._config.min_ms), float(old_predictor.min_ms))
            )
            metrics = predictor.fit(
                records_snapshot,
                l2=self._config.l2,
                use_scene_models=self._config.use_scene_models,
                min_scene_samples=self._config.min_scene_samples,
            )
            self._predictor = predictor
            if (self._config.save_path and
                    train_round % self._SAVE_EVERY_SUCCESSFUL_TRAIN_ROUNDS == 0):
                predictor.save(self._config.save_path)
                logger.info(
                    "MultiSlo online model checkpoint saved at round %d: %s",
                    train_round,
                    self._config.save_path,
                )
            with self._state_lock:
                self._train_round = train_round
                self._last_metrics = metrics
            logger.info(
                "MultiSlo online train round %d done: samples=%d mae=%.4f rmse=%.4f r2=%.4f",
                train_round,
                len(records_snapshot),
                metrics.mae,
                metrics.rmse,
                metrics.r2,
            )
        except Exception as e:
            logger.warning("MultiSlo online training failed: %s", e)

    def _internal_normalize_record(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not REQUIRED_KEYS.issubset(record.keys()):
            return None
        try:
            chunk_sizes = [int(x) for x in record["chunk_sizes"]]
            cached_tokens = [int(x) for x in record["cached_tokens"]]
            computed_tokens = [int(x) for x in record.get("computed_tokens", [])]
            sched_tokens = int(record["sched_tokens"])
            model_run_ms = float(record["model_run_ms"])
        except (TypeError, ValueError):
            return None

        if not chunk_sizes or sched_tokens <= 0:
            return None
        if len(chunk_sizes) != len(cached_tokens):
            return None
        if computed_tokens and len(computed_tokens) != len(chunk_sizes):
            return None

        normalized: dict[str, Any] = {
            "chunk_sizes": chunk_sizes,
            "cached_tokens": cached_tokens,
            "sched_tokens": sched_tokens,
            "model_run_ms": model_run_ms,
        }
        if computed_tokens:
            normalized["computed_tokens"] = computed_tokens
        return normalized
