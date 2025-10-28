import threading
import logging
import random
from collections import deque
from copy import deepcopy

try:
    from .RLConfig import RLSchedulerConfig
    from .RLAgent import RLAgent
except ImportError:
    from RLConfig import RLSchedulerConfig
    from RLAgent import RLAgent

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,rl_agent:RLAgent):
        self.config = RLSchedulerConfig.from_env()
        self.rl_agent = rl_agent
        self.rl_replay_buffer = deque(maxlen=self.config.replay_buffer_size)  # 双端队列（自动淘汰旧数据）
        self.max_episodes = self.config.max_episodes

        # 线程安全相关参数
        self.is_training = False
        self.train_lock = threading.Lock()

    def _train_in_thread(self):
        """在单独线程中执行模型训练"""
        try:
            with self.train_lock:
                self.is_training = True
                try:
                    for episode in range(self.max_episodes):
                        batch = self.sample_exp(self.config.train_batch_size)
                        self.rl_agent.learn(batch)
                    self.rl_agent.save_model()
                    logger.info(f"Finished the {self.max_episodes} round of model training and save the model")
                except Exception as e:
                    logger.error(f"Failed to train episode {episode}: {str(e)}")
                finally:
                    self.is_training = False
        except Exception as e:
            logger.error(f"Failed to start training thread: {str(e)}")

    def add_exp(self, before_env_info,after_env_info,action) :
        """添加经验到回放池（每个经验对应一轮迭代的交互）"""
        reward = self.caculate_reward(before_env_info,after_env_info)
        self.rl_replay_buffer.append((deepcopy(before_env_info), deepcopy(after_env_info), action, reward))

        # 当经验池达到阈值且满足训练间隔时，启动异步训练
        if (len(self.rl_replay_buffer) >= self.config.train_batch_size and 
            not self.is_training):
            
            logger.info(f"start train ..........")
            # 创建并启动训练线程
            train_thread = threading.Thread(target=self._train_in_thread, daemon=True)
            train_thread.start()
            logger.info(f"Start the asynchronous training thread and determine the current size of the experience pool: {len(self.rl_replay_buffer)}")

    def sample_exp(self, batch_size: int):
        """从回放池采样经验（每个经验对应一轮迭代的交互）"""
        return random.sample(self.rl_replay_buffer, batch_size)
    
    def caculate_reward(self, before_env_info,after_env_info):
        """
        综合奖励函数：
        R = λ1*SLO成功率 + λ2*吞吐量 - λ3*延迟惩罚 - λ4*资源浪费
        """
        reward = 0.0

        # ---------- 1. SLO 满足情况 ----------
        before_recent_comform_slo_rate = before_env_info.get("recent_comform_slo_rate", 0.0)
        after_recent_comform_slo_rate = after_env_info.get("recent_comform_slo_rate", 0.0)
        if after_recent_comform_slo_rate > before_recent_comform_slo_rate:
            R_slo = 1
        else:
            R_slo = -1
        
        # ---------- 2. 吞吐量奖励 ----------

        R_tp = after_env_info.get("current_throughput", 0.0) / self.config.throughput_norm

        # ---------- 3. 延迟惩罚 ----------
        # 如果当前iteration选择的S很大、且运行队列中decode请求比例高，则惩罚
        select_B = after_env_info.get("select_B", 0)
        select_S = after_env_info.get("select_S", 0)
        actual_B = after_env_info.get("actual_B", 0)
        actual_S = after_env_info.get("actual_S", 0)
        decode_count = after_env_info.get("decode_count", 0)
        prefill_count = after_env_info.get("prefill_count", 0)

        # 匹配B、S惩罚
        R_match_B_penalty = 0
        R_match_S_penalty = 0
        if abs(select_B - actual_B) > 1:
            R_match_B_penalty = -1
        if abs(select_S - actual_S) > 255:
            R_match_S_penalty = -1
        R_match_penalty = 0.5*R_match_B_penalty + 0.5*R_match_S_penalty



        # running = after_env_info.get("running_requests", [])
        # if len(running) > 0:
        #     num_decode = sum(1 for r in running if self._is_decode_phase(r))
        #     decode_ratio = num_decode / len(running)
        # else:
        #     decode_ratio = 0.0

        # S_ratio = min(1.0, S / self.config.S_norm)
        # R_latency_penalty = decode_ratio * S_ratio  # decode越多，S越大惩罚越强


        # ---------- 综合 ----------
        reward = (self.config.lambda_slo * R_slo) + \
                (self.config.lambda_tp * R_tp) + \
                (self.config.lambda_match * R_match_penalty)

        return reward

    def _is_decode_phase(self, request) -> bool:
        """判断请求是否处于decode阶段"""
        try:
            return request.num_computed_tokens >= request.num_prompt_tokens
        except AttributeError:
            # 如果字段不存在，假设是prefill阶段
            return False