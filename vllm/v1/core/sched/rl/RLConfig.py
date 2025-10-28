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
    rl_finished_reqs_buffer_size: int = 100  # 已完成请求缓冲区大小
    throughput_buffer_size: int = 100  # 吞吐缓冲区大小
    latency_buffer_size: int = 100  # 延迟缓冲区大小

    # === agent 参数 ===
    device: str = "cpu"                # 训练所用设备（cuda/cpu）
    train_enabled: bool = True         # 是否进行训练
    state_dim: int = 4                 # 状态维度（根据环境定义）
    action_dim: int = 16               # 动作维度（根据环境定义）
    lr: float = 1e-4                   # 学习率
    gamma: float = 0.9                 # 折扣因子（长期奖励权重）
    epsilon: float = 0.2               # 初始探索概率
    epsilon_decay: float = 0.995       # 探索概率衰减率
    epsilon_min: float = 0.01          # 最小探索概率
    target_net_update_freq: int = 100  # 目标网络更新频率（单位：step）
    rl_model: str = "DualAttentionNetwork"        # 强化学习模型类型（MLPNetwork/TransformerNetwork）
    use_pretrained_model: bool = False  # 是否使用预训练模型
    pretrained_model_path: str = ""     # 预训练模型路径
    report_monitor_frequency: float = 30.0  # 报告监控频率（单位：秒）

    # === env 参数 ===
    llm_model_len: int = 10000              # LLM模型长度
    B_norm: int = 16                  # 批次归一化因子
    S_norm: int = 2048                  # 序列长度归一化因子
    throughput_norm: float = 20000.0    # 吞吐量归一化因子
    slo_norm: float = 30.0              # SLO 归一化因子
    prompt_norm: float = 10000.0      # 提示归一化因子

    # === reward 函数参数 ===
    lambda_slo: float = 1            # SLO 奖励权重
    lambda_tp: float = 0.3           # 吞吐量奖励权重
    lambda_latency: float = 0.2      # 延迟奖励权重
    lambda_match: float = 0.3        # 匹配奖励权重

    # === DualAttentionNetwork 参数 ===
    Global_state_dim: int = 18          # G: 全局状态维度
    K_waiting: int = 10                # 取top-K个等待请求提取特征
    Feature_waiting: int = 3           # 等待队列特征维度
    K_running: int = 10                # 取top-K个运行请求提取特征
    Feature_running: int = 7           # 运行队列特征维度
        
    # === RLOptimizer 参数 ===
    optimization_timeout_ms: float = 20.0    # 优化器超时时间（ms）

    # === Trainer 参数 ===
    replay_buffer_size: int = 10000    # 经验回放缓冲区大小
    max_episodes: int = 100          # 最大训练轮数（单位：episode）
    train_batch_size: int = 32        # 训练批次大小

    @classmethod
    def from_env(cls) -> 'RLSchedulerConfig':
        config = cls(
            # 通用
            verbose_logging=os.getenv('VLLM_RL_VERBOSE_LOGGING', 'false').lower() == 'true',

            # RLScheduler 参数
            enabled=os.getenv('VLLM_RL_SCHEDULER_ENABLED', 'false').lower() == 'true',
            
            # RLUtils 参数
            rl_finished_reqs_buffer_size=int(os.getenv('VLLM_RL_FINISHED_REQS_BUFFER_SIZE', '100')),
            throughput_buffer_size=int(os.getenv('VLLM_RL_THROUGHPUT_BUFFER_SIZE', '100')),
            latency_buffer_size=int(os.getenv('VLLM_RL_LATENCY_BUFFER_SIZE', '100')),

            # agent 参数
            device=os.getenv('VLLM_RL_DEVICE', 'cpu').lower(),
            train_enabled=os.getenv('VLLM_RL_TRAIN_ENABLED', 'false').lower() == 'true',
            state_dim=int(os.getenv('VLLM_RL_STATE_DIM', '4')),
            action_dim=int(os.getenv('VLLM_RL_ACTION_DIM', '16')),
            lr=float(os.getenv('VLLM_RL_LR', '1e-4')),
            gamma=float(os.getenv('VLLM_RL_GAMMA', '0.99')),
            epsilon=float(os.getenv('VLLM_RL_EPSILON', '0.2')),
            epsilon_decay=float(os.getenv('VLLM_RL_EPSILON_DECAY', '0.995')),
            epsilon_min=float(os.getenv('VLLM_RL_EPSILON_MIN', '0.01')),
            target_net_update_freq=int(os.getenv('VLLM_RL_TARGET_NET_UPDATE_FREQ', '100')),
            rl_model=os.getenv('VLLM_RL_MODEL', 'DualAttentionNetwork'),
            use_pretrained_model=os.getenv('VLLM_RL_USE_PRETRAINED_MODEL', 'false').lower() == 'true',
            pretrained_model_path=os.getenv('VLLM_RL_PRETRAINED_MODEL_PATH', ''),
            report_monitor_frequency=float(os.getenv('VLLM_RL_REPORT_MONITOR_FREQUENCY', '30.0')),
            
            # env 参数
            llm_model_len=int(os.getenv('VLLM_RL_LLM_MODEL_LEN', '10000')),
            B_norm=int(os.getenv('VLLM_RL_B_NORM', '16')),
            S_norm=int(os.getenv('VLLM_RL_S_NORM', '2048')),
            throughput_norm=float(os.getenv('VLLM_RL_THROUGHPUT_NORM', '20000.0')),
            slo_norm=float(os.getenv('VLLM_RL_SLO_NORM', '30.0')),
            prompt_norm=float(os.getenv('VLLM_RL_PROMPT_NORM', '10000.0')),

            # reward 函数参数
            lambda_slo=float(os.getenv('VLLM_RL_LAMBDA_SLO', '1')),
            lambda_tp=float(os.getenv('VLLM_RL_LAMBDA_TP', '0.3')),
            lambda_latency=float(os.getenv('VLLM_RL_LAMBDA_LATENCY', '0.2')),
            lambda_match=float(os.getenv('VLLM_RL_LAMBDA_MATCH', '0.3')),

            # DualAttentionNetwork 参数
            Global_state_dim=int(os.getenv('VLLM_RL_GLOBAL_STATE_DIM', '18')),
            K_waiting=int(os.getenv('VLLM_RL_K_WAITING', '10')),
            Feature_waiting=int(os.getenv('VLLM_RL_FEATURE_WAITING', '3')),
            K_running=int(os.getenv('VLLM_RL_K_RUNNING', '10')),
            Feature_running=int(os.getenv('VLLM_RL_FEATURE_RUNNING', '7')),

            # RLOptimizer 参数
            optimization_timeout_ms=float(os.getenv('VLLM_RL_OPTIMIZATION_TIMEOUT_MS', '20.0')),

            # Trainer 参数
            replay_buffer_size=int(os.getenv('VLLM_RL_REPLAY_BUFFER_SIZE', '10000')),
            max_episodes=int(os.getenv('VLLM_RL_MAX_EPISODES', '1000')),
            train_batch_size=int(os.getenv('VLLM_RL_TRAIN_BATCH_SIZE', '32')),
        )
        return config
