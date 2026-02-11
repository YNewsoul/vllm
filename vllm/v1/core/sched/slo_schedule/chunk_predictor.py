"""
用于分块耗时预测的训练与推理工具函数

背景知识：
- Chunk Prefill 是 LLM 推理中的一种调度策略，将大的 prefill 请求分成多个小块（chunk）
  与 decode 请求混合执行，以减少首 token 延迟（TTFT）
- 每个 chunk 包含多个请求，每个请求可能处于 prefill（预填充）或 decode（解码）阶段
- Prefill 阶段：处理输入 prompt，一次计算多个 token，计算密集型（compute-bound）
- Decode 阶段：自回归生成，每次只生成 1 个 token，内存带宽受限（memory-bound）
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional

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


class DurationPredictor:
    """
    用于预测 LLM Chunk Prefill 单次前向计算耗时的回归模型

    模型架构：多项式特征 + 标准化 + Ridge 回归
    - 多项式特征：捕捉特征之间的非线性交互（如 prefill_tokens * kv_cache）
    - 标准化：消除不同特征量纲差异，加速收敛
    - Ridge 回归：L2 正则化防止过拟合，对多重共线性鲁棒
    """

    def __init__(
        self, degree: int = 2, alpha: float = 1.0, model_path: Optional[str] = None
    ):
        """
        初始化预测器

        Args:
            degree: 多项式特征的最高次数，默认 2（二次交互）
                   - degree=1: 线性模型
                   - degree=2: 包含 x1*x2, x1^2 等二次项
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

        # 合并系数，用于 ultrafast 预测（将 scaler 和 model 合并）
        # 公式: y = X_poly @ combined_coef + combined_intercept
        self._combined_coef: Optional[np.ndarray] = None
        self._combined_intercept: Optional[float] = None

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
        self._precompute_combined_coef()  # 预计算合并系数（用于加速预测）
        return self

    def _precompute_combined_coef(self):
        """
        预计算合并后的系数，将标准化器和模型合并为单次运算

        数学推导：
            原始公式: y = ((X_poly - mean) / scale) @ coef + intercept
                      = X_poly @ (coef / scale) - mean @ (coef / scale) + intercept
            合并公式: y = X_poly @ combined_coef + combined_intercept

        其中：
            combined_coef = coef / scale          （系数除以标准差）
            combined_intercept = intercept - mean @ combined_coef  （截距减去均值贡献）

        优化效果：将预测从 3 次运算减少到 1 次矩阵乘法
        """
        # combined_coef = coef / scale
        self._combined_coef = self.model.coef_ / self.scaler.scale_
        # combined_intercept = intercept - mean @ combined_coef
        self._combined_intercept = float(
            self.model.intercept_ - np.dot(self.scaler.mean_, self._combined_coef)
        )
        # Precompute polynomial expansion indices for ultrafast prediction
        self._precompute_poly_indices()

    def _precompute_poly_indices(self):
        """
        预计算多项式展开的索引，用于极致优化的手动展开

        对于 degree=2、n 个特征的多项式展开，会产生：
        - n 个线性项: x0, x1, ..., x(n-1)
        - n*(n+1)/2 个二次项: x0^2, x0*x1, ..., x(n-1)^2

        我们将索引存储为元组，以便直接数组访问
        这完全消除了 sklearn PolynomialFeatures.transform() 的开销
        """
        n_features = len(self.feature_names)

        # 对于 degree=2: 线性项 + 二次项（包括交叉项）
        # 线性项: (i,) 表示 x[i]
        # 二次项: (i, j) 表示 x[i] * x[j]，其中 i <= j
        self._poly_indices = []

        # 首先添加线性项
        for i in range(n_features):
            self._poly_indices.append((i,))

        # 然后添加二次项（上三角矩阵，包含对角线）
        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    self._poly_indices.append((i, j))

        self._n_poly_features = len(self._poly_indices)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        批量预测（用于训练评估）

        Args:
            X: 特征 DataFrame

        Returns:
            预测的耗时数组（毫秒）
        """
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
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
            - MAPE: 平均绝对百分比误差，反映相对误差（%）
        """
        y_pred = self.predict(X)
        return {
            "MSE": mean_squared_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred),
            "MAPE": np.mean(np.abs((y - y_pred) / y)) * 100,
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
        poly_feature_names = self.poly.get_feature_names_out(self.feature_names)
        importance = np.abs(self.model.coef_) * self.scaler.scale_
        df = pd.DataFrame({"feature": poly_feature_names, "importance": importance})
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
            # Combined coefficients for ultrafast prediction
            "_combined_coef": self._combined_coef,
            "_combined_intercept": self._combined_intercept,
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
        # 加载合并系数（如果存在），否则重新计算
        self._combined_coef = model_data.get("_combined_coef")
        self._combined_intercept = model_data.get("_combined_intercept")
        if self._combined_coef is None:
            # 向后兼容：为旧版模型重新计算合并系数
            self._precompute_combined_coef()
        else:
            # 仍需预计算多项式索引，用于 ultrafast 预测
            self._precompute_poly_indices()

    def predict_ultrafast(
        self,
        chunk_sizes: List[int],
        all_cached_tokens: List[int],
        all_computed_tokens: List[int],
        total_scheduled_tokens: int,
    ) -> float:
        """
        极致优化的预测方法，完全手写多项式展开

        Args:
            chunk_sizes: 本次 chunk 中每个请求处理的 token 数
            all_cached_tokens: 每个请求已缓存的 KV cache token 数
            all_computed_tokens: 每个请求已计算的总 token 数
            total_scheduled_tokens: 本次 chunk 的总 token 预算

        Returns:
            预测的耗时（毫秒）
        """
        # ===== 内联特征提取（单次遍历所有请求） =====
        decode_count = 0  # decode 请求数量
        decode_total_kv_cache = 0  # decode 请求的总 KV cache
        decode_total_computed = 0  # decode 请求的总已计算 token
        num_prefill_reqs = 0  # prefill 请求数量
        prefill_total_tokens = 0  # prefill 总 token 数
        prefill_max_tokens = 0  # prefill 最大 token 数
        prefill_min_tokens = float("inf")  # prefill 最小 token 数
        prefill_total_kv_cache = 0  # prefill 请求的总 KV cache
        prefill_attention_cost = 0  # prefill attention 计算量
        prefill_kv_product = 0  # prefill cross-attention 开销
        prefill_self_attention = 0  # prefill self-attention 开销
        total_kv_cache = 0  # 所有请求的总 KV cache
        total_computed_tokens_sum = 0  # 所有请求的总已计算 token

        for i, chunk_size in enumerate(chunk_sizes):
            cached = all_cached_tokens[i]
            computed = all_computed_tokens[i]
            total_kv_cache += cached
            total_computed_tokens_sum += computed

            if chunk_size == 1:
                decode_count += 1
                decode_total_kv_cache += cached
                decode_total_computed += computed
            else:
                num_prefill_reqs += 1
                prefill_total_tokens += chunk_size
                if chunk_size > prefill_max_tokens:
                    prefill_max_tokens = chunk_size
                if chunk_size < prefill_min_tokens:
                    prefill_min_tokens = chunk_size
                prefill_total_kv_cache += cached
                prefill_attention_cost += chunk_size * (cached + chunk_size)
                prefill_kv_product += chunk_size * cached
                prefill_self_attention += chunk_size * chunk_size

        # 无 prefill 请求时，min 设为 0
        if num_prefill_reqs == 0:
            prefill_min_tokens = 0

        # 特征元组
        # 顺序必须与 self.feature_names 完全一致
        x = (
            decode_total_kv_cache,
            decode_total_computed,
            num_prefill_reqs,
            prefill_total_tokens,
            prefill_max_tokens,
            prefill_min_tokens,
            prefill_total_kv_cache,
            prefill_attention_cost,
            prefill_kv_product,
            prefill_self_attention,
            total_scheduled_tokens,
            total_kv_cache,
            total_computed_tokens_sum,
            decode_count
            * prefill_total_tokens,  # decode_prefill_interaction（混合 batching 交互项）
        )

        # ===== 手写多项式展开 + 点积运算 =====
        # 将展开和点积融合为单次循环，避免中间数组分配
        result = self._combined_intercept  # 从截距开始累加
        coef = self._combined_coef  # 合并后的系数数组
        n_features = len(x)  # 特征数量（动态获取）
        idx = 0  # 系数索引

        # 线性项: x[0], x[1], ..., x[n_features-1]
        for i in range(n_features):
            result += x[i] * coef[idx]
            idx += 1

        # 二次项: x[i] * x[j]，其中 i <= j（仅当 degree >= 2 时）
        # 包括平方项（i==j）和交叉项（i<j）
        if self.degree >= 2:
            for i in range(n_features):
                xi = x[i]
                for j in range(i, n_features):
                    result += xi * x[j] * coef[idx]
                    idx += 1

        return result


__all__ = [
    "DurationPredictor",
]
