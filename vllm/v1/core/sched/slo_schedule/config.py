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
    min_chunk: int = 32
    max_chunk: int = 2048
    fixed_chunk_size: int = 0
    random_chunk_enabled: bool = False
        
    @classmethod
    def from_env(cls) -> 'SloSchedulerConfig':
        config = cls(
            model = os.getenv("VLLM_SLO_MODEL", "profiling_result_a6000_model_predictor.joblib"),
            min_chunk = int(os.getenv("VLLM_SLO_MIN_CHUNK", "1")),
            max_chunk = int(os.getenv("VLLM_SLO_MAX_CHUNK", "2048")),
            fixed_chunk_size = int(os.getenv("VLLM_FIXED_CHUNK_SIZE", "0")),
            random_chunk_enabled = os.getenv("VLLM_RANDOM_CHUNK_ENABLED", "false").lower() == "true",
        )
        return config

    def to_dict(self)  -> dict:
        """将配置转换为字典形式"""
        return asdict(self)