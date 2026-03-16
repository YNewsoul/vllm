import os
from dataclasses import dataclass,asdict

@dataclass
class SloSchedulerConfig:
    """RL 调度器配置"""
    multislo_model: str = "profiling_result_a6000_model_predictor.joblib"
    min_chunk: int = 32
    max_chunk: int = 2048
    fixed_chunk_size: int = 2048
    sched_mode: str = "multislo" # 调度模型，可选 "multislo","random_chunk","fixed_chunk","sarathi"
        
    @classmethod
    def from_env(cls) -> 'SloSchedulerConfig':
        config = cls(
            multislo_model = os.getenv("VLLM_SLO_MODEL", "profiling_result_a6000_model_predictor.joblib"),
            min_chunk = int(os.getenv("VLLM_SLO_MIN_CHUNK", "1")),
            max_chunk = int(os.getenv("VLLM_SLO_MAX_CHUNK", "2048")),
            fixed_chunk_size = int(os.getenv("VLLM_FIXED_CHUNK_SIZE", "0")),
            sched_mode = os.getenv("VLLM_SLO_SCHED_MODE", "multislo"),
        )
        return config

    def to_dict(self)  -> dict:
        """将配置转换为字典形式"""
        return asdict(self)