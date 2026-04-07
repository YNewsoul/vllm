import os
from dataclasses import asdict, dataclass

@dataclass
class SloSchedulerConfig:
    """RL 调度器配置"""
    predictor_model: str = "3090_qwen7b_tp2_multislo_model.joblib"
    min_chunk: int = 32
    max_chunk: int = 2048
    fixed_chunk_size: int = 2048
    sched_mode: str = "multislo" # 调度模型，可选 "multislo","random-chunk","fixed-chunk","sarathi-fcfs","qoserve"
    qoserve_alpha: float = 0.1
    multislo_urgency_threshold: float = 0.5

    # 在线模型训练相关参数
    online_train_enabled: bool = False
    online_buffer_size: int = 4000
    online_warmup_samples: int = 400
    online_retrain_interval: int = 200
    online_l2: float = 1e-4
    online_use_scene_models: bool = True
    online_min_scene_samples: int = 200
    online_min_ms: float = 0.05
    online_save_path: str = "online_model.joblib"
    online_ingest_queue_size: int = 4096
        
    @classmethod
    def from_env(cls) -> 'SloSchedulerConfig':
        online_train_enabled = (
            os.getenv("VLLM_MULTISLO_ONLINE_TRAIN_ENABLED", "false").lower()
            == "true"
        )
        online_use_scene_models = (
            os.getenv("VLLM_MULTISLO_ONLINE_USE_SCENE_MODELS", "true").lower()
            == "true"
        )
        config = cls(
            predictor_model = os.getenv("VLLM_SLO_PREDICTOR_MODEL", "3090_qwen7b_tp2_multislo_model.joblib"),
            min_chunk = int(os.getenv("VLLM_SLO_MIN_CHUNK", "1")),
            max_chunk = int(os.getenv("VLLM_SLO_MAX_CHUNK", "2048")),
            fixed_chunk_size = int(os.getenv("VLLM_FIXED_CHUNK_SIZE", "2048")),
            sched_mode = os.getenv("VLLM_SLO_SCHED_MODE", "multislo"),
            qoserve_alpha = float(os.getenv("VLLM_QOSERVE_ALPHA", "0.1")),
            multislo_urgency_threshold = float(os.getenv("VLLM_MULTISLO_URGENCY_THRESHOLD", "0.5")),
            online_train_enabled = online_train_enabled,
            online_buffer_size = int(os.getenv("VLLM_MULTISLO_ONLINE_BUFFER_SIZE", "4000")),
            online_warmup_samples = int(os.getenv("VLLM_MULTISLO_ONLINE_WARMUP_SAMPLES", "800")),
            online_retrain_interval = int(os.getenv("VLLM_MULTISLO_ONLINE_RETRAIN_INTERVAL", "200")),
            online_l2 = float(os.getenv("VLLM_MULTISLO_ONLINE_L2", "1e-4")),
            online_use_scene_models = online_use_scene_models,
            online_min_scene_samples = int(os.getenv("VLLM_MULTISLO_ONLINE_MIN_SCENE_SAMPLES", "200")),
            online_min_ms = float(os.getenv("VLLM_MULTISLO_ONLINE_MIN_MS", "0.05")),
            online_save_path = os.getenv("VLLM_MULTISLO_ONLINE_SAVE_PATH", "online_model.joblib"),
            online_ingest_queue_size = int(os.getenv("VLLM_MULTISLO_ONLINE_INGEST_QUEUE_SIZE", "4096")),
        )
        return config

    def to_dict(self) -> dict:
        """将配置转换为字典形式"""
        return asdict(self)
