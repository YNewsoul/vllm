import os
from dataclasses import dataclass,asdict
import logging

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

@dataclass
class SloSchedulerConfig:
    """RL 调度器配置"""
    model: str = "profiling_result_a6000_model_predictor.joblib"
    @classmethod
    def from_env(cls) -> 'SloSchedulerConfig':
        config = cls(
            model = os.getenv("VLLM_SLO_MODEL", "profiling_result_a6000_model_predictor.joblib"),

        )
        return config

    def to_dict(self)  -> dict:
        """将配置转换为字典形式"""
        return asdict(self)