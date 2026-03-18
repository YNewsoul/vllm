import os
from dataclasses import dataclass,asdict

@dataclass
class SloSchedulerConfig:
    """RL 调度器配置"""
    predictor_model: str = "3090_qwen7b_tp2_multislo_model.joblib"
    min_chunk: int = 32
    max_chunk: int = 2048
    fixed_chunk_size: int = 2048
    sched_mode: str = "multislo" # 调度模型，可选 "multislo","random_chunk","fixed_chunk","sarathi","qoserve"
    sarathi_mode: str = "fcfs" # sarathi 调度模式，可选 "fcfs","edf","srpf"
    qoserve_alpha: float = 0.1
    multislo_urgency_threshold: float = 0.5
        
    @classmethod
    def from_env(cls) -> 'SloSchedulerConfig':
        config = cls(
            predictor_model = os.getenv("VLLM_SLO_PREDICTOR_MODEL", "3090_qwen7b_tp2_multislo_model.joblib"),
            min_chunk = int(os.getenv("VLLM_SLO_MIN_CHUNK", "1")),
            max_chunk = int(os.getenv("VLLM_SLO_MAX_CHUNK", "2048")),
            fixed_chunk_size = int(os.getenv("VLLM_FIXED_CHUNK_SIZE", "2048")),
            sched_mode = os.getenv("VLLM_SLO_SCHED_MODE", "multislo"),
            sarathi_mode = os.getenv("VLLM_SARATHI_MODE", "fcfs"),
            qoserve_alpha = float(os.getenv("VLLM_QOSERVE_ALPHA", "0.1")),
            multislo_urgency_threshold = float(os.getenv("VLLM_MULTISLO_URGENCY_THRESHOLD", "0.5")),
        )
        return config

    def to_dict(self)  -> dict:
        """将配置转换为字典形式"""
        return asdict(self)