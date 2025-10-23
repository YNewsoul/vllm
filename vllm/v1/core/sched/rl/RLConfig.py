import os
from dataclasses import dataclass
import logging

from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass
class RLSchedulerConfig:
    """RL 调度器配置"""

    # === 通用 ===
    verbose_logging: bool = False       # 是否启用详细日志记录

    # === RLScheduler 参数 ===
    enabled: bool = False              # 是否启用RL SLA调度器
    
    # === agent 参数 ===
    device: str = "cpu"                # 训练所用设备（cuda/cpu）
    train_enabled: bool = True         # 是否进行训练
    state_dim: int = 4                 # 状态维度（根据环境定义）
    action_dim: int = 16               # 动作维度（根据环境定义）
    lr: float = 1e-4                   # 学习率
    gamma: float = 0.99                # 折扣因子（长期奖励权重）
    epsilon: float = 0.2               # 初始探索概率
    epsilon_decay: float = 0.995       # 探索概率衰减率
    epsilon_min: float = 0.01          # 最小探索概率
    target_net_update_freq: int = 100  # 目标网络更新频率（单位：step）
    rl_model: str = "MLPNetwork"        # 强化学习模型类型（MLPNetwork/TransformerNetwork）
    use_pretrained_model: bool = False  # 是否使用预训练模型
    pretrained_model_path: str = ""     # 预训练模型路径
    report_monitor_frequency: float = 30.0  # 报告监控频率（单位：秒）
    
    # === RLOptimizer 参数 ===
    optimization_timeout_ms: float = 20.0    # 优化器超时时间（ms）

    # === Trainer 参数 ===
    replay_buffer_size: int = 10000    # 经验回放缓冲区大小
    max_episodes: int = 100          # 最大训练轮数（单位：episode）
    train_batch_size: int = 32        # 训练批次大小

    @classmethod
    def from_env(cls) -> 'RLSchedulerConfig':
        config = cls(
            # 功能开关
            verbose_logging=os.getenv('VLLM_RL_VERBOSE_LOGGING', 'false').lower() == 'true',

            # RLScheduler 参数
            enabled=os.getenv('VLLM_RL_SCHEDULER_ENABLED', 'false').lower() == 'true',

            # agent 参数
            device=os.getenv('VLLM_RL_DEVICE', 'cpu').lower(),
            train_enabled=os.getenv('VLLM_RL_TRAIN_ENABLED', 'false').lower() == 'true',
            lr=float(os.getenv('VLLM_RL_LR', '1e-4')),
            gamma=float(os.getenv('VLLM_RL_GAMMA', '0.99')),
            epsilon=float(os.getenv('VLLM_RL_EPSILON', '0.2')),
            epsilon_decay=float(os.getenv('VLLM_RL_EPSILON_DECAY', '0.995')),
            epsilon_min=float(os.getenv('VLLM_RL_EPSILON_MIN', '0.01')),
            target_net_update_freq=int(os.getenv('VLLM_RL_TARGET_NET_UPDATE_FREQ', '100')),
            rl_model=os.getenv('VLLM_RL_MODEL', 'MLPNetwork'),
            use_pretrained_model=os.getenv('VLLM_RL_USE_PRETRAINED_MODEL', 'false').lower() == 'true',
            pretrained_model_path=os.getenv('VLLM_RL_PRETRAINED_MODEL_PATH', ''),
            report_monitor_frequency=float(os.getenv('VLLM_RL_REPORT_MONITOR_FREQUENCY', '30.0')),
            
            # RLOptimizer 参数
            optimization_timeout_ms=float(os.getenv('VLLM_RL_OPTIMIZATION_TIMEOUT_MS', '20.0')),

            # Trainer 参数
            replay_buffer_size=int(os.getenv('VLLM_RL_REPLAY_BUFFER_SIZE', '10000')),
            max_episodes=int(os.getenv('VLLM_RL_MAX_EPISODES', '1000')),
            train_batch_size=int(os.getenv('VLLM_RL_TRAIN_BATCH_SIZE', '32')),
        )
        return config
