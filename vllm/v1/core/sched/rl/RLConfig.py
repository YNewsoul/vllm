import os
from dataclasses import dataclass,asdict
import logging

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

@dataclass
class RLSchedulerConfig:
    """RL 调度器配置"""

    # === 1 RLUtils 参数 ===========
    rl_finished_reqs_buffer_size: int = 10  # 已完成请求缓冲区大小
    max_num_scheduled_tokens: int = 2048  # 最大调度令牌数

    # === 2 Agent 参数 =============
    ## ==== 2.1 整体参数 ====
    heuristic_algorithm_enabled: bool = False    # 是否使用启发式算法
    device: str = "cpu"                  # 训练所用设备（cuda/cpu）
    train_enabled: bool = False          # 是否进行训练
    rl_model: str = "DualAttentionNetwork"        # 强化学习模型类型（MLPNetwork/TransformerNetwork）
    use_pretrained_model: bool = False   # 是否使用预训练模型
    pretrained_model_path: str = ""      # 预训练模型路径
    model_save_frequency: float = 600.0  # 保存模型频率（单位：秒）
    log_frequency: int = 40              # 日志记录频率（单位：step）
    action_dim: int = 6                # 动作维度（根据环境定义）

    # ==== 2.2 DQN 参数 ====
    lr: float = 1e-4                   # 学习率
    gamma: float = 0.9                 # 折扣因子（长期奖励权重）
    epsilon: float = 0.2               # 初始探索概率
    epsilon_max_step: int = 100000       # 最大探索步数
    epsilon_min: float = 0.05          # 最小探索概率
    target_net_update_freq: int = 400  # 目标网络更新频率（单位：step）

    # ==== 2.3 DualAttentionNetwork 参数 ====
    Global_state_dim: int = 3          # G: 全局状态维度
    K_waiting: int = 5                # 取top-K个等待请求提取特征
    Feature_waiting: int = 3           # 等待队列特征维度
    K_running: int = 15                # 取top-K个运行请求提取特征
    Feature_running: int = 3           # 运行队列特征维度

    # ==== 2.4 env_to_state 参数 ====
    prompt_norm: float = 30000.0      # 提示归一化因子
    model_time_norm: float = 500.0    # 模型运行时间归一化因子
    token_budget_norm: float = 2048.0 # 令牌预算归一化因子

    # === 3 Trainer 参数 ========================
    # ==== 3.1 常规参数 ====
    replay_buffer_size: int = 30000    # 经验回放缓冲区大小
    train_batch_size: int = 64        # 训练批次大小
    train_total_time: float = 28800.0 # 训练总时间（单位：s）
    tpot_slo: float = 50.0            # TPOT SLO
    tpot_start: float = 0.3           # TPOT 计算开始比例

    # ==== 3.2 Reward 函数参数 ====
    lambda_decode: float = 1.5         # decode 奖励权重
    lambda_prefill: float = 1.0        # prefill 奖励权重
    lambda_budget: float = 0.5        # budget 奖励权重
    progress_d1: float = 0.3          # progress 奖励权重 d1
    progress_d2: float = 0.5          # progress 奖励权重 d2
    progress_d3: float = 0.65         # progress 奖励权重 d3
    progress_d4: float = 0.7          # progress 奖励权重 d4
    lambda_progress_d1: float = 0.1    # progress 奖励权重 d1
    lambda_progress_d2: float = 0.3    # progress 奖励权重 d2
    lambda_progress_d3: float = 0.5    # progress 奖励权重 d3
    lambda_progress_d4: float = 1.4    # progress 奖励权重 d4
    lambda_progress_d5: float = 1.7    # progress 奖励权重 d5



    @classmethod
    def from_env(cls) -> 'RLSchedulerConfig':
        config = cls(

            # 1 RLUtils 参数
            rl_finished_reqs_buffer_size=int(os.getenv('VLLM_RL_FINISHED_REQS_BUFFER_SIZE', '10')),
            max_num_scheduled_tokens=int(os.getenv('VLLM_RL_MAX_NUM_SCHEDULED_TOKENS', '2048')),

            # 2 Agent 参数
            ## ==== 2.1 整体参数 ====
            heuristic_algorithm_enabled=os.getenv('VLLM_RL_HEURISTIC_ALGORITHM_ENABLED', 'false').lower() == 'true',
            device=os.getenv('VLLM_RL_DEVICE', 'cpu').lower(),
            train_enabled=os.getenv('VLLM_RL_TRAIN_ENABLED', 'false').lower() == 'true',
            rl_model=os.getenv('VLLM_RL_MODEL', 'DualAttentionNetwork'),
            use_pretrained_model=os.getenv('VLLM_RL_USE_PRETRAINED_MODEL', 'false').lower() == 'true',
            pretrained_model_path=os.getenv('VLLM_RL_PRETRAINED_MODEL_PATH', ''),
            model_save_frequency=float(os.getenv('VLLM_RL_MODEL_SAVE_FREQUENCY', '600.0')),
            log_frequency=int(os.getenv('VLLM_RL_LOG_FREQUENCY', '40')),
            action_dim=int(os.getenv('VLLM_RL_ACTION_DIM', '6')),

            ## ==== 2.2 DQN 参数 ====
            lr=float(os.getenv('VLLM_RL_LR', '1e-4')),
            gamma=float(os.getenv('VLLM_RL_GAMMA', '0.9')),
            epsilon=float(os.getenv('VLLM_RL_EPSILON', '0.2')),
            epsilon_max_step=int(os.getenv('VLLM_RL_EPSILON_MAX_STEP', '100000')),
            epsilon_min=float(os.getenv('VLLM_RL_EPSILON_MIN', '0.1')),
            target_net_update_freq=int(os.getenv('VLLM_RL_TARGET_NET_UPDATE_FREQ', '400')),

            ## ==== 2.3 DualAttentionNetwork 参数 ====
            Global_state_dim=int(os.getenv('VLLM_RL_GLOBAL_STATE_DIM', '3')),
            K_waiting=int(os.getenv('VLLM_RL_K_WAITING', '5')),
            Feature_waiting=int(os.getenv('VLLM_RL_FEATURE_WAITING', '3')),
            K_running=int(os.getenv('VLLM_RL_K_RUNNING', '15')),
            Feature_running=int(os.getenv('VLLM_RL_FEATURE_RUNNING', '3')),

            ## ==== 2.4 env_to_state 参数 ====
            prompt_norm=float(os.getenv('VLLM_RL_PROMPT_NORM', '30000.0')),
            model_time_norm=float(os.getenv('VLLM_RL_MODEL_TIME_NORM', '500.0')),
            token_budget_norm=float(os.getenv('VLLM_RL_TOKEN_BUDGET_NORM', '2048.0')),

            # 3 Trainer 参数
            ## ==== 3.1 常规参数 ====
            replay_buffer_size=int(os.getenv('VLLM_RL_REPLAY_BUFFER_SIZE', '30000')),
            train_batch_size=int(os.getenv('VLLM_RL_TRAIN_BATCH_SIZE', '64')),
            train_total_time=float(os.getenv('VLLM_RL_TRAIN_TOTAL_TIME', '28800.0')),
            tpot_slo=float(os.getenv('VLLM_RL_TPOT_SLO', '50.0')),
            tpot_start=float(os.getenv('VLLM_RL_TPOT_START', '0.3')),

            ## ==== 3.2 Reward 函数参数 ====
            lambda_decode=float(os.getenv('VLLM_RL_LAMBDA_DECODE', '1.5')),
            lambda_prefill=float(os.getenv('VLLM_RL_LAMBDA_PREFILL', '1.0')),
            lambda_budget=float(os.getenv('VLLM_RL_LAMBDA_BUDGET', '0.5')),

            progress_d1=float(os.getenv('VLLM_RL_PROGRESS_D1', '0.3')),
            progress_d2=float(os.getenv('VLLM_RL_PROGRESS_D2', '0.45')),
            progress_d3=float(os.getenv('VLLM_RL_PROGRESS_D3', '0.6')),
            progress_d4=float(os.getenv('VLLM_RL_PROGRESS_D4', '0.7')),
            lambda_progress_d1=float(os.getenv('VLLM_RL_LAMBDA_PROGRESS_D1', '0.1')),
            lambda_progress_d2=float(os.getenv('VLLM_RL_LAMBDA_PROGRESS_D2', '0.5')),
            lambda_progress_d3=float(os.getenv('VLLM_RL_LAMBDA_PROGRESS_D3', '1')),
            lambda_progress_d4=float(os.getenv('VLLM_RL_LAMBDA_PROGRESS_D4', '1.4')),
            lambda_progress_d5=float(os.getenv('VLLM_RL_LAMBDA_PROGRESS_D5', '1.7')),

        )
        return config

    def to_dict(self)  -> dict:
        """将配置转换为字典形式"""
        return asdict(self)