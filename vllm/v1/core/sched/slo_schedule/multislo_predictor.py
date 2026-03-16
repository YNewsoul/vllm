"""Standalone theoretical iteration-time model for chunk-prefill inference.

This module is independent from the previous sklearn polynomial predictor.
Design goals:
1) Keep training/inference lightweight (closed-form Ridge).
2) Use stronger theory-driven features (including computed_tokens).
3) Improve robustness with a global model + optional scene submodels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import joblib
import numpy as np

# Keep backward-compatible minimum required keys for data loaders.
# computed_tokens is optional at runtime; when absent we fall back to cached_tokens.
REQUIRED_KEYS = {"chunk_sizes", "cached_tokens", "sched_tokens", "model_run_ms"}


@dataclass
class TheoryModelParams:
    b0: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class FitMetrics:
    mae: float
    rmse: float
    r2: float

class MultiSloPredictor:
    """Lightweight theoretical latency model with optional scene experts."""

    FEATURE_NAMES = (
        "decode_total_kv_cache",
        "decode_total_computed",
        "decode_count",
        "num_prefill_reqs",
        "num_reqs",
        "prefill_total_tokens",
        "prefill_max_tokens",
        "prefill_min_tokens",
        "prefill_total_kv_cache",
        "prefill_attention_cost",
        "prefill_kv_product",
        "prefill_self_attention",
        "sched_tokens",
        "total_kv_cache",
        "total_computed_tokens",
        "decode_prefill_interaction",
        # selected high-value nonlinearity without full polynomial expansion
        "prefill_attention_cost_sq",
        "prefill_attention_x_self",
        "prefill_attention_x_total_computed",
        "decode_kv_x_decode_count",
    )

    SCENES = ("decode", "mixed", "prefill")

    def __init__(
        self,
        params: TheoryModelParams | None = None,
        min_ms: float = 0.0,
        feature_mean: Sequence[float] | None = None,
        feature_scale: Sequence[float] | None = None,
        scene_models: Dict[str, Dict] | None = None,
    ):
        self.params = params or TheoryModelParams()
        self.min_ms = float(min_ms)

        n = len(self.FEATURE_NAMES)
        self.feature_mean = np.asarray(
            feature_mean if feature_mean is not None else np.zeros(n, dtype=np.float64),
            dtype=np.float64,
        )
        self.feature_scale = np.asarray(
            (
                feature_scale
                if feature_scale is not None
                else np.ones(n, dtype=np.float64)
            ),
            dtype=np.float64,
        )

        if self.feature_mean.shape[0] != n or self.feature_scale.shape[0] != n:
            raise ValueError("feature_mean/feature_scale length mismatch")

        self.feature_scale = np.where(
            np.abs(self.feature_scale) < 1e-12, 1.0, self.feature_scale
        )
        self.params.weights = self._align_weights(self.params.weights)
        self.scene_models = self._sanitize_scene_models(scene_models or {})
        self._fast_global = (0.0, tuple([0.0] * len(self.FEATURE_NAMES)))
        self._fast_scene_models: Dict[str, tuple] = {}
        self._rebuild_fast_models()

    def _align_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        aligned = {name: 0.0 for name in self.FEATURE_NAMES}
        for k, v in weights.items():
            if k in aligned:
                aligned[k] = float(v)
        return aligned

    def _sanitize_scene_models(self, scene_models: Dict[str, Dict]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        n = len(self.FEATURE_NAMES)
        for scene, blob in scene_models.items():
            if scene not in self.SCENES or not isinstance(blob, dict):
                continue
            mean = np.asarray(blob.get("feature_mean", np.zeros(n)), dtype=np.float64)
            scale = np.asarray(blob.get("feature_scale", np.ones(n)), dtype=np.float64)
            if mean.shape[0] != n or scale.shape[0] != n:
                continue
            scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
            weights = self._align_weights(blob.get("weights", {}))
            out[scene] = {
                "b0": float(blob.get("b0", 0.0)),
                "weights": weights,
                "feature_mean": mean,
                "feature_scale": scale,
                "num_samples": int(blob.get("num_samples", 0)),
            }
        return out

    @staticmethod
    def _safe_scale(v: float) -> float:
        return 1.0 if abs(float(v)) < 1e-12 else float(v)

    def _blob_to_fast_params(self, blob: Dict) -> tuple:
        """
        Convert normalized linear model:
          y = b0 + sum_i w_i * ((x_i - mean_i) / scale_i)
        into:
          y = b_fast + sum_i c_i * x_i
        """
        weights = self._align_weights(blob["weights"])
        means = np.asarray(blob["feature_mean"], dtype=np.float64)
        scales = np.asarray(blob["feature_scale"], dtype=np.float64)

        intercept = float(blob["b0"])
        coefs: List[float] = []
        for i, name in enumerate(self.FEATURE_NAMES):
            s = self._safe_scale(scales[i])
            c = float(weights[name]) / s
            intercept -= c * float(means[i])
            coefs.append(c)
        return intercept, tuple(coefs)

    def _rebuild_fast_models(self) -> None:
        self._fast_global = self._blob_to_fast_params(self._global_blob())
        fast_scene: Dict[str, tuple] = {}
        for scene, blob in self.scene_models.items():
            fast_scene[scene] = self._blob_to_fast_params(blob)
        self._fast_scene_models = fast_scene

    @staticmethod
    def _extract_terms_fast(
        chunk_sizes: Sequence[int],
        cached_tokens: Sequence[int],
        computed_tokens: Sequence[int] | None,
        sched_tokens: int,
    ) -> tuple[str, tuple]:
        if computed_tokens is None:
            computed_tokens = cached_tokens

        n = len(chunk_sizes)
        if n != len(cached_tokens) or n != len(computed_tokens):
            raise ValueError(
                "chunk_sizes/cached_tokens/computed_tokens length mismatch"
            )

        decode_total_kv_cache = 0
        decode_total_computed = 0
        decode_count = 0

        num_prefill_reqs = 0
        prefill_total_tokens = 0
        prefill_max_tokens = 0
        prefill_min_tokens = 10**18
        prefill_total_kv_cache = 0
        prefill_attention_cost = 0
        prefill_kv_product = 0
        prefill_self_attention = 0

        total_kv_cache = 0
        total_computed_tokens = 0
        num_reqs = n

        has_decode = False
        has_prefill = False

        for i in range(n):
            c = int(chunk_sizes[i])
            k = int(cached_tokens[i])
            u = int(computed_tokens[i])

            total_kv_cache += k
            total_computed_tokens += u

            if c <= 1:
                has_decode = True
                decode_count += 1
                decode_total_kv_cache += k
                decode_total_computed += u
            else:
                has_prefill = True
                num_prefill_reqs += 1
                prefill_total_tokens += c
                if c > prefill_max_tokens:
                    prefill_max_tokens = c
                if c < prefill_min_tokens:
                    prefill_min_tokens = c
                prefill_total_kv_cache += k
                prefill_attention_cost += c * (u + c)
                prefill_kv_product += c * k
                prefill_self_attention += c * c

        if num_prefill_reqs == 0:
            prefill_min_tokens = 0

        if has_decode and has_prefill:
            scene = "mixed"
        elif has_prefill:
            scene = "prefill"
        else:
            scene = "decode"

        decode_prefill_interaction = decode_count * prefill_total_tokens
        prefill_attention_cost_sq = float(prefill_attention_cost) * float(
            prefill_attention_cost
        )
        prefill_attention_x_self = float(prefill_attention_cost) * float(
            prefill_self_attention
        )
        prefill_attention_x_total_computed = float(prefill_attention_cost) * float(
            total_computed_tokens
        )
        decode_kv_x_decode_count = float(decode_total_kv_cache) * float(decode_count)

        feats = (
            float(decode_total_kv_cache),
            float(decode_total_computed),
            float(decode_count),
            float(num_prefill_reqs),
            float(num_reqs),
            float(prefill_total_tokens),
            float(prefill_max_tokens),
            float(prefill_min_tokens),
            float(prefill_total_kv_cache),
            float(prefill_attention_cost),
            float(prefill_kv_product),
            float(prefill_self_attention),
            float(sched_tokens),
            float(total_kv_cache),
            float(total_computed_tokens),
            float(decode_prefill_interaction),
            float(prefill_attention_cost_sq),
            float(prefill_attention_x_self),
            float(prefill_attention_x_total_computed),
            float(decode_kv_x_decode_count),
        )
        return scene, feats

    @staticmethod
    def _apply_fast_linear(intercept: float, coef: tuple, feats: tuple) -> float:
        (
            f0,
            f1,
            f2,
            f3,
            f4,
            f5,
            f6,
            f7,
            f8,
            f9,
            f10,
            f11,
            f12,
            f13,
            f14,
            f15,
            f16,
            f17,
            f18,
            f19,
        ) = feats
        (
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            c10,
            c11,
            c12,
            c13,
            c14,
            c15,
            c16,
            c17,
            c18,
            c19,
        ) = coef
        return (
            intercept
            + c0 * f0
            + c1 * f1
            + c2 * f2
            + c3 * f3
            + c4 * f4
            + c5 * f5
            + c6 * f6
            + c7 * f7
            + c8 * f8
            + c9 * f9
            + c10 * f10
            + c11 * f11
            + c12 * f12
            + c13 * f13
            + c14 * f14
            + c15 * f15
            + c16 * f16
            + c17 * f17
            + c18 * f18
            + c19 * f19
        )

    def _global_blob(self) -> Dict:
        return {
            "b0": float(self.params.b0),
            "weights": self._align_weights(self.params.weights),
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "num_samples": 0,
        }

    @classmethod
    def _fit_linear_model(cls, X_raw: np.ndarray, y: np.ndarray, l2: float) -> Dict:
        mean = np.mean(X_raw, axis=0)
        scale = np.std(X_raw, axis=0)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)

        Xn = (X_raw - mean) / scale
        X = np.column_stack([np.ones(Xn.shape[0], dtype=np.float64), Xn])

        xtx = X.T @ X
        reg = np.eye(xtx.shape[0], dtype=np.float64)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(xtx + float(l2) * reg, X.T @ y)

        return {
            "b0": float(beta[0]),
            "weights": {
                name: float(beta[i + 1]) for i, name in enumerate(cls.FEATURE_NAMES)
            },
            "feature_mean": mean,
            "feature_scale": scale,
        }


    def predict(
        self,
        chunk_sizes: Sequence[int],
        cached_tokens: Sequence[int],
        sched_tokens: int,
        computed_tokens: Sequence[int] | None = None,
    ) -> float:
        scene, feats = self._extract_terms_fast(
            chunk_sizes=chunk_sizes,
            cached_tokens=cached_tokens,
            computed_tokens=computed_tokens,
            sched_tokens=sched_tokens,
        )
        intercept, coef = self._fast_scene_models.get(scene, self._fast_global)
        y = self._apply_fast_linear(intercept, coef, feats)
        return max(self.min_ms, float(y))

    def predict_record(self, record: Dict) -> float:
        return self.predict(
            chunk_sizes=record["chunk_sizes"],
            cached_tokens=record["cached_tokens"],
            sched_tokens=int(record["sched_tokens"]),
            computed_tokens=record.get("computed_tokens"),
        )

    def fit(
        self,
        records: List[Dict],
        l2: float = 1e-4,
        use_scene_models: bool = True,
        min_scene_samples: int = 200,
    ) -> FitMetrics:
        if not records:
            raise ValueError("No records to train on")

        x_rows = []
        y_vals = []
        scenes: List[str] = []

        for rec in records:
            scene, feats = self._extract_terms_fast(
                chunk_sizes=rec["chunk_sizes"],
                cached_tokens=rec["cached_tokens"],
                computed_tokens=rec.get("computed_tokens"),
                sched_tokens=int(rec["sched_tokens"]),
            )
            x_rows.append(feats)
            y_vals.append(float(rec["model_run_ms"]))
            scenes.append(scene)

        X_raw = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_vals, dtype=np.float64)

        global_blob = self._fit_linear_model(X_raw, y, l2=l2)
        self.params = TheoryModelParams(
            b0=float(global_blob["b0"]),
            weights=self._align_weights(global_blob["weights"]),
        )
        self.feature_mean = np.asarray(global_blob["feature_mean"], dtype=np.float64)
        self.feature_scale = np.asarray(global_blob["feature_scale"], dtype=np.float64)
        self.feature_scale = np.where(
            np.abs(self.feature_scale) < 1e-12, 1.0, self.feature_scale
        )

        # Train scene experts for stable high-volume scenes.
        self.scene_models = {}
        if use_scene_models:
            scene_arr = np.asarray(scenes)
            for scene in self.SCENES:
                idx = np.where(scene_arr == scene)[0]
                if idx.shape[0] < int(min_scene_samples):
                    continue
                sub_blob = self._fit_linear_model(X_raw[idx], y[idx], l2=l2)
                sub_blob["num_samples"] = int(idx.shape[0])
                self.scene_models[scene] = self._sanitize_scene_models(
                    {scene: sub_blob}
                )[scene]

        self._rebuild_fast_models()

        pred = np.asarray([self.predict_record(r) for r in records], dtype=np.float64)
        return self._metrics(y, pred)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> FitMetrics:
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(math.sqrt(np.mean(err * err)))
        ss_res = float(np.sum(err * err))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return FitMetrics(mae=mae, rmse=rmse, r2=r2)

    def evaluate(self, records: List[Dict]) -> FitMetrics:
        if not records:
            raise ValueError("No records to evaluate")
        y = np.asarray([float(r["model_run_ms"]) for r in records], dtype=np.float64)
        yp = np.asarray([self.predict_record(r) for r in records], dtype=np.float64)
        return self._metrics(y, yp)

    def save(self, path: str) -> None:
        payload = {
            "model": "theoretical_iteration_linear_v3",
            "feature_names": list(self.FEATURE_NAMES),
            "min_ms": self.min_ms,
            "global": {
                "b0": float(self.params.b0),
                "weights": self._align_weights(self.params.weights),
                "feature_mean": self.feature_mean.tolist(),
                "feature_scale": self.feature_scale.tolist(),
            },
            "scene_models": {
                scene: {
                    "b0": float(blob["b0"]),
                    "weights": self._align_weights(blob["weights"]),
                    "feature_mean": np.asarray(
                        blob["feature_mean"], dtype=np.float64
                    ).tolist(),
                    "feature_scale": np.asarray(
                        blob["feature_scale"], dtype=np.float64
                    ).tolist(),
                    "num_samples": int(blob.get("num_samples", 0)),
                }
                for scene, blob in self.scene_models.items()
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "MulsloPredictor":
        data = joblib.load(path)

        if isinstance(data, dict) and "global" in data:
            g = data["global"]
            params = TheoryModelParams(
                b0=float(g.get("b0", 0.0)),
                weights={str(k): float(v) for k, v in g.get("weights", {}).items()},
            )
            return cls(
                params=params,
                min_ms=float(data.get("min_ms", 0.0)),
                feature_mean=g.get("feature_mean"),
                feature_scale=g.get("feature_scale"),
                scene_models=data.get("scene_models", {}),
            )

        raise ValueError(
            f"Unsupported model format in {path}. "
            "Please retrain with current theoretical_trainer.py to generate v3 model."
        )


__all__ = [
    "MulSloPredictor",
]
