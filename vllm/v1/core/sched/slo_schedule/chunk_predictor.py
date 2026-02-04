# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
用于分块耗时预测的训练与推理工具函数

背景知识：
- Chunk Prefill 是 LLM 推理中的一种调度策略，将大的 prefill 请求
  分成多个小块（chunk）与 decode 请求混合执行，以减少首 token 延迟（TTFT）
- 每个 chunk 包含多个请求，每个请求可能处于 prefill（预填充）或
  decode（解码）阶段
- Prefill 阶段：处理输入 prompt，一次计算多个 token，计算密集型
- Decode 阶段：自回归生成，每次只生成 1 个 token，内存带宽受限
"""

import warnings
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 静默 scikit-learn 关于"无有效特征名"的告警，配合 ndarray 快速路径提升性能
warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning,
)


def extract_features(record: dict) -> dict:
    """
    从单条 profiling 记录提取用于预测 model_run_duration_ms 的特征

    LLM 推理耗时主要由以下因素决定：
    1. Attention 计算复杂度: O(seq_len * context_len)
    2. KV Cache 读取量（影响内存带宽）
    3. 请求数量（影响 batch 并行度）

    Args:
        record: 包含以下字段的dict
            - chunk_sizes: 每个请求在本次 chunk 中处理的 token 数
                          =1 表示 decode 请求，>1 表示 prefill 请求
            - all_cached_tokens: 每个请求已缓存的 KV cache token 数
            - all_computed_tokens: 每个请求已计算的总 token 数
            - total_scheduled_tokens: 本次 chunk 的总 token budget
            - num_running_reqs: 正在运行的请求数

    Returns:
        features: 提取的特征字典
    """
    chunk_sizes = record["chunk_sizes"]
    all_cached_tokens = record["all_cached_tokens"]
    all_computed_tokens = record["all_computed_tokens"]

    features: dict[str, float] = {}

    # ==================== Decode 请求特征 ====================
    # Decode 请求特点：chunk_size == 1，每次只生成一个 token
    # 性能瓶颈：内存带宽（需要读取完整的 KV cache）
    decode_indices = [i for i, s in enumerate(chunk_sizes) if s == 1]

    # decode 请求数量：影响 batch size 和并行度
    features["num_decode_reqs"] = len(decode_indices)

    # decode 请求的总 KV cache 大小：反映内存读取压力
    # 每个 decode token 需要与所有历史 KV 做 attention
    features["decode_total_kv_cache"] = sum(all_cached_tokens[i]
                                            for i in decode_indices)

    # decode 请求已计算的 token 总数（通常 = cached_tokens，用于交叉验证）
    features["decode_total_computed"] = sum(all_computed_tokens[i]
                                            for i in decode_indices)

    # ==================== Prefill 请求特征 ====================
    # Prefill 请求特点：chunk_size > 1，一次处理多个输入 token
    # 性能瓶颈：计算密集（矩阵乘法、attention 计算）
    prefill_indices = [i for i, s in enumerate(chunk_sizes) if s > 1]

    # prefill 请求数量
    features["num_prefill_reqs"] = len(prefill_indices)

    if prefill_indices:
        prefill_tokens = [chunk_sizes[i] for i in prefill_indices]
        prefill_cached = [all_cached_tokens[i] for i in prefill_indices]

        # prefill 总 token 数：直接影响 FFN 和 QKV 投影的计算量
        features["prefill_total_tokens"] = sum(prefill_tokens)

        # prefill 最大/最小 token 数：反映请求的不均衡程度
        # 不均衡会导致 GPU 利用率下降（短请求需要 padding）
        features["prefill_max_tokens"] = max(prefill_tokens)
        features["prefill_min_tokens"] = min(prefill_tokens)

        # prefill 请求的总 KV cache：用于 cross-attention
        features["prefill_total_kv_cache"] = sum(prefill_cached)

        # ===== Attention 计算复杂度建模 =====
        # Attention 复杂度 = Q * (K + V)
        #   = chunk_size * (cached_tokens + chunk_size)
        # 其中：
        #   - chunk_size: 当前要计算的 query token 数
        #   - cached_tokens: 已有的 KV cache（cross-attention 部分）
        #   - chunk_size: 当前 chunk 内部的 self-attention 部分
        features["prefill_attention_cost"] = sum(
            chunk_sizes[i] * (all_cached_tokens[i] + chunk_sizes[i])
            for i in prefill_indices)

        # Cross-attention 开销：Q 与历史 KV 的 attention
        # 复杂度 ~ chunk_size * cached_tokens
        features["prefill_kv_product"] = sum(
            chunk_sizes[i] * all_cached_tokens[i] for i in prefill_indices)

        # Self-attention 开销：当前 chunk 内部 token 之间的 attention
        # 复杂度 ~ chunk_size^2（因为 mask 下实际是 chunk_size^2 / 2）
        features["prefill_self_attention"] = sum(chunk_sizes[i]**2
                                                 for i in prefill_indices)
    else:
        # 无 prefill 请求时，所有 prefill 相关特征置零
        features["prefill_total_tokens"] = 0
        features["prefill_max_tokens"] = 0
        features["prefill_min_tokens"] = 0
        features["prefill_total_kv_cache"] = 0
        features["prefill_attention_cost"] = 0
        features["prefill_kv_product"] = 0
        features["prefill_self_attention"] = 0

    # ==================== 全局特征 ====================
    # 总调度 token 数：本次 chunk 的 token budget 上限
    features["total_scheduled_tokens"] = record["total_scheduled_tokens"]

    # 总运行请求数：影响 batch 并行效率
    features["num_running_reqs"] = record["num_running_reqs"]

    # 所有请求的总 KV cache：反映总内存占用和读取量
    features["total_kv_cache"] = sum(all_cached_tokens)

    # 所有请求已计算的 token 总数
    features["total_computed_tokens"] = sum(all_computed_tokens)

    # ==================== 交互特征 ====================
    # Decode-Prefill 交互项：当 decode 和 prefill 混合时，
    # 会产生额外的调度开销和 GPU 资源竞争
    # 这个特征捕捉混合 batching 带来的性能干扰
    features["decode_prefill_interaction"] = (features["num_decode_reqs"] *
                                              features["prefill_total_tokens"])

    # 可选：目标字段（训练时使用）
    if "model_run_duration_ms" in record:
        features["model_run_duration_ms"] = record["model_run_duration_ms"]

    return features


def prepare_dataset(data: list[dict], ) -> tuple[pd.DataFrame, np.ndarray]:
    """
    将原始 profiling 列表转换为特征矩阵和目标值

    Args:
        data: profiling 列表，每条记录包含 chunk 的详细信息

    Returns:
        feature_df: 特征 DataFrame，每行一个样本，每列一个特征
        target: 目标值数组（model_run_duration_ms）
    """
    features_list = [extract_features(record) for record in data]
    df = pd.DataFrame(features_list)

    target = df["model_run_duration_ms"].values
    feature_df = df.drop(columns=["model_run_duration_ms"])

    return feature_df, target


class DurationPredictor:
    """
    用于预测 LLM Chunk Prefill 单次前向计算耗时的回归模型

    模型架构：多项式特征 + 标准化 + Ridge 回归
    - 多项式特征：捕捉特征之间的非线性交互
    - 标准化：消除不同特征量纲差异，加速收敛
    - Ridge 回归：L2 正则化防止过拟合，对多重共线性鲁棒

    使用场景：
    1. 离线训练：用历史 profiling 数据训练模型
    2. 在线推理：调度器实时预测不同调度方案的耗时，辅助决策
    """

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 1.0,
        model_path: Optional[str] = None,
    ):
        """
        初始化预测器

        Args:
            degree: 多项式特征的最高次数，默认 2（二次交互）
                   - degree=1: 线性模型
                   - degree=2: 包含 x1*x2, x1^2 等二次项
                   - degree>2: 更复杂的非线性，但容易过拟合
            alpha: Ridge 回归的正则化强度，默认 1.0
                   - alpha 越大，模型越简单，防止过拟合
                   - alpha 越小，模型越复杂，拟合能力更强
            model_path: 预训练模型路径，如果提供则直接加载
        """
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.feature_names = None  # 原始特征名，用于推理时特征对齐

        if model_path:
            self._load(model_path)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        训练模型

        流程：原始特征 -> 多项式展开 -> 标准化 -> Ridge 回归

        Args:
            X: 特征 DataFrame，由 prepare_dataset() 生成
            y: 目标值数组（model_run_duration_ms）

        Returns:
            self: 训练后的模型实例（支持链式调用）
        """
        self.feature_names = X.columns.tolist()
        X_poly = self.poly.fit_transform(X)  # 生成多项式特征
        X_scaled = self.scaler.fit_transform(X_poly)  # 标准化
        self.model.fit(X_scaled, y)  # 训练 Ridge 回归
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        批量预测
        """
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """
        评估模型性能

        Args:
            X: 测试集特征
            y: 测试集真实耗时

        Returns:
            评估指标字典：
            - MSE: 均方误差，对大误差敏感
            - RMSE: 均方根误差，与目标值同量纲（ms）
            - MAE: 平均绝对误差，更鲁棒
            - R2: 决定系数，1 表示完美预测，0 表示等于预测均值
            - mean_abs_pct_err: 平均绝对百分比误差，反映相对误差（%）
        """
        y_pred = self.predict(X)
        return {
            "MSE": mean_squared_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred),
            "mean_abs_pct_err": np.mean(np.abs((y - y_pred) / y)) * 100,
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性排名

        重要性计算：|系数| * 特征标准差
        这反映了特征对预测值的实际贡献（考虑了特征的变化范围）

        Args:
            top_n: 返回前 N 个最重要的特征

        Returns:
            DataFrame，包含 feature（特征名）和 importance（重要性）列
        """
        poly_feature_names = self.poly.get_feature_names_out(
            self.feature_names)
        importance = np.abs(self.model.coef_) * self.scaler.scale_
        df = pd.DataFrame({
            "feature": poly_feature_names,
            "importance": importance
        })
        return df.nlargest(top_n, "importance")

    def save(self, filepath: str):
        """
        保存模型到文件

        保存内容包括：多项式变换器、标准化器、Ridge 模型、特征名

        Args:
            filepath: 保存路径（.pkl 或 .joblib）
        """
        model_data = {
            "degree": self.degree,
            "alpha": self.alpha,
            "poly": self.poly,
            "scaler": self.scaler,
            "model": self.model,
            "feature_names": self.feature_names,
            # 新增元数据
            "created_at": datetime.now().isoformat(),
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DurationPredictor":
        """
        从文件加载预训练模型（类方法）
        """
        instance = cls()
        instance._load(filepath)
        return instance

    def _load(self, filepath: str):
        """
        模型加载方法
        """
        model_data = joblib.load(filepath)
        self.degree = model_data["degree"]
        self.alpha = model_data["alpha"]
        self.poly = model_data["poly"]
        self.scaler = model_data["scaler"]
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]

    def predict_single(self, record: dict) -> float:
        """
        单条记录预测（适用于在线推理场景）
        """
        return self.predict_record(record)

    def predict_record(self, record: dict) -> float:
        """
        单条记录快速预测，避免 DataFrame 开销
        """
        features = extract_features(record)
        row = [[features[name] for name in self.feature_names]]
        X_poly = self.poly.transform(row)
        X_scaled = self.scaler.transform(X_poly)
        return float(self.model.predict(X_scaled)[0])

    def predict_batch(self, records: list[dict]) -> np.ndarray:
        """
        批量数据预测
        """
        features_list = [extract_features(record) for record in records]
        X = pd.DataFrame(features_list)[self.feature_names]
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

    def predict_from_raw(
        self,
        chunk_sizes: list[int],
        all_cached_tokens: list[int],
        all_computed_tokens: list[int],
        total_scheduled_tokens: int,
        num_running_reqs: int,
    ) -> float:
        """
        从原始参数直接预测
        """
        record = {
            "chunk_sizes": chunk_sizes,
            "all_cached_tokens": all_cached_tokens,
            "all_computed_tokens": all_computed_tokens,
            "total_scheduled_tokens": total_scheduled_tokens,
            "num_running_reqs": num_running_reqs,
        }
        return self.predict_single(record)


__all__ = [
    "DurationPredictor",
    "extract_features",
    "prepare_dataset",
]
