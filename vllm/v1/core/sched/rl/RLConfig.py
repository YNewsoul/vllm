import os
from dataclasses import dataclass
import logging

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

@dataclass
class RLSchedulerConfig:
    """RL 调度器配置"""

    # === 通用 ===
    verbose_logging: bool = False       # 是否启用详细日志记录

    # === RLScheduler 参数 ===
    enabled: bool = False              # 是否启用RL SLA调度器
    
    # === RLUtils 参数 ===
    rl_finished_reqs_buffer_size: int = 10  # 已完成请求缓冲区大小
    throughput_buffer_size: int = 100  # 吞吐缓冲区大小
    latency_buffer_size: int = 100  # 延迟缓冲区大小

    # === agent 参数 ===
    device: str = "cpu"                # 训练所用设备（cuda/cpu）
    train_enabled: bool = False         # 是否进行训练
    rl_model: str = "DualAttentionNetwork"        # 强化学习模型类型（MLPNetwork/TransformerNetwork）
    use_pretrained_model: bool = False  # 是否使用预训练模型
    pretrained_model_path: str = ""     # 预训练模型路径
    save_model_frequency: float = 180.0  # 保存模型频率（单位：秒）
    log_frequency: int = 40           # 日志记录频率（单位：step）

    # === DQN 参数 ===
    lr: float = 1e-4                   # 学习率
    gamma: float = 0.9                 # 折扣因子（长期奖励权重）
    epsilon: float = 0.2               # 初始探索概率
    epsilon_max_step: int = 100000       # 最大探索步数
    epsilon_min: float = 0.1          # 最小探索概率
    target_net_update_freq: int = 200  # 目标网络更新频率（单位：step）

    # === MLPNetwork 参数
    state_dim: int = 4                 # 状态维度（根据环境定义）
    action_dim: int = 4               # 动作维度（根据环境定义）

    # === env 参数 ===
    llm_model_len: int = 10000              # LLM模型长度
    token_budget_norm: int = 2048       # 序列长度归一化因子
    prompt_norm: float = 30000.0      # 提示归一化因子
    time_norm: float = 300.0           # 时间归一化因子

    # === reward 函数参数 ===
    lambda_recent_comform_slo: float = 2.0         # 最近符合SLO请求奖励权重
    lambda_decode: float = 1.0         # decode 奖励权重
    lambda_prefill: float = 1.0        # prefill 奖励权重
    lambda_finish: float = 2.0        # finish 奖励权重

    # === DualAttentionNetwork 参数 ===
    Global_state_dim: int = 10          # G: 全局状态维度
    K_waiting: int = 5                # 取top-K个等待请求提取特征
    Feature_waiting: int = 2           # 等待队列特征维度
    K_running: int = 20                # 取top-K个运行请求提取特征
    Feature_running: int = 2           # 运行队列特征维度
        
    # === RLOptimizer 参数 ===
    optimization_timeout_ms: float = 10.0    # 优化器超时时间（ms）

    # === Trainer 参数 ===
    replay_buffer_size: int = 10000    # 经验回放缓冲区大小
    train_batch_size: int = 128        # 训练批次大小
    train_total_time: float = 3600.0    # 训练总时间（单位：s）

    @classmethod
    def from_env(cls) -> 'RLSchedulerConfig':
        config = cls(
            # 通用
            verbose_logging=os.getenv('VLLM_RL_VERBOSE_LOGGING', 'false').lower() == 'true',

            # RLScheduler 参数
            enabled=os.getenv('VLLM_RL_SCHEDULER_ENABLED', 'false').lower() == 'true',
            
            # RLUtils 参数
            rl_finished_reqs_buffer_size=int(os.getenv('VLLM_RL_FINISHED_REQS_BUFFER_SIZE', '10')),
            throughput_buffer_size=int(os.getenv('VLLM_RL_THROUGHPUT_BUFFER_SIZE', '100')),
            latency_buffer_size=int(os.getenv('VLLM_RL_LATENCY_BUFFER_SIZE', '100')),

            # agent 参数
            device=os.getenv('VLLM_RL_DEVICE', 'cpu').lower(),
            train_enabled=os.getenv('VLLM_RL_TRAIN_ENABLED', 'false').lower() == 'true',
            rl_model=os.getenv('VLLM_RL_MODEL', 'DualAttentionNetwork'),
            use_pretrained_model=os.getenv('VLLM_RL_USE_PRETRAINED_MODEL', 'false').lower() == 'true',
            pretrained_model_path=os.getenv('VLLM_RL_PRETRAINED_MODEL_PATH', ''),
            save_model_frequency=float(os.getenv('VLLM_RL_SAVE_MODEL_FREQUENCY', '180.0')),
            log_frequency=int(os.getenv('VLLM_RL_LOG_FREQUENCY', '40')),

            # DQN 参数
            lr=float(os.getenv('VLLM_RL_LR', '1e-4')),
            gamma=float(os.getenv('VLLM_RL_GAMMA', '0.9')),
            epsilon=float(os.getenv('VLLM_RL_EPSILON', '0.2')),
            epsilon_max_step=int(os.getenv('VLLM_RL_EPSILON_MAX_STEP', '100000')),
            epsilon_min=float(os.getenv('VLLM_RL_EPSILON_MIN', '0.1')),
            target_net_update_freq=int(os.getenv('VLLM_RL_TARGET_NET_UPDATE_FREQ', '200')),

            # MLPNetwork 参数
            state_dim=int(os.getenv('VLLM_RL_STATE_DIM', '4')),
            action_dim=int(os.getenv('VLLM_RL_ACTION_DIM', '4')),

            # env 参数
            llm_model_len=int(os.getenv('VLLM_RL_LLM_MODEL_LEN', '10000')),
            token_budget_norm=int(os.getenv('VLLM_RL_TOKEN_BUDGET_NORM', '2048')),
            prompt_norm=float(os.getenv('VLLM_RL_PROMPT_NORM', '30000.0')),
            time_norm=float(os.getenv('VLLM_RL_TIME_NORM', '300.0')),

            # reward 参数
            lambda_recent_comform_slo=float(os.getenv('VLLM_RL_LAMBDA_RECENT_CONFORM_SLO', '2.0')),
            lambda_decode=float(os.getenv('VLLM_RL_LAMBDA_DECODE', '1.0')),
            lambda_prefill=float(os.getenv('VLLM_RL_LAMBDA_PREFILL', '1.0')),
            lambda_finish=float(os.getenv('VLLM_RL_LAMBDA_FINISH', '2.0')),

            # DualAttentionNetwork 参数
            Global_state_dim=int(os.getenv('VLLM_RL_GLOBAL_STATE_DIM', '10')),
            K_waiting=int(os.getenv('VLLM_RL_K_WAITING', '5')),
            Feature_waiting=int(os.getenv('VLLM_RL_FEATURE_WAITING', '2')),
            K_running=int(os.getenv('VLLM_RL_K_RUNNING', '20')),
            Feature_running=int(os.getenv('VLLM_RL_FEATURE_RUNNING', '2')),

            # RLOptimizer 参数
            optimization_timeout_ms=float(os.getenv('VLLM_RL_OPTIMIZATION_TIMEOUT_MS', '10.0')),

            # Trainer 参数
            replay_buffer_size=int(os.getenv('VLLM_RL_REPLAY_BUFFER_SIZE', '10000')),
            train_batch_size=int(os.getenv('VLLM_RL_TRAIN_BATCH_SIZE', '128')),
            train_total_time=float(os.getenv('VLLM_RL_TRAIN_TOTAL_TIME', '3600.0')),
        )
        return config
