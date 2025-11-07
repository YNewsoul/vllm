import threading
import logging
import random
import time
from collections import deque

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

        # 线程安全相关参数
        self.is_training = False
        self.train_lock = threading.Lock()

    def _train_in_thread(self):
        """在单独线程中执行模型训练"""
        try:
            with self.train_lock:
                self.is_training = True
                try:
                    self.rl_agent.reset()
                    start_time = time.monotonic()
                    while True:
                        batch = self._sample_exp(self.config.train_batch_size)
                        self.rl_agent.learn(batch)
                        now = time.monotonic()
                        if now - start_time >= self.config.train_total_time:
                            break
                    logger.info(f"Finished the training process in {self.config.train_total_time} seconds")
                except Exception as e:
                    logger.error(f"Failed to train episode {episode}: {str(e)}")
                finally:
                    self.is_training = False
        except Exception as e:
            logger.error(f"Failed to start training thread: {str(e)}")

    def add_exp(self, before_env_info,after_env_info,action) :
        """添加经验到回放池（每个经验对应一轮迭代的交互）"""
        reward = self._caculate_reward(before_env_info,after_env_info)
        self.rl_replay_buffer.append((before_env_info, after_env_info, action, reward))

        # 当经验池达到阈值且满足训练间隔时，启动异步训练
        if len(self.rl_replay_buffer) >= self.config.train_batch_size and not self.is_training:
            logger.info(f"start train ..........")
            # 创建并启动训练线程
            train_thread = threading.Thread(target=self._train_in_thread, daemon=True)
            train_thread.start()
            logger.info(f"Start the asynchronous training thread and determine the current size of the experience pool: {len(self.rl_replay_buffer)}")

    def _sample_exp(self, batch_size: int):
        """从回放池采样经验（每个经验对应一轮迭代的交互）"""
        return random.sample(self.rl_replay_buffer, batch_size)
    
    def _caculate_reward(self, before_env_info,after_env_info):
        """
        综合奖励函数：
        R = λ1*SLO成功率 + λ2*吞吐量 - λ3*延迟惩罚 - λ4*资源浪费
        """
        reward = 0.0

        # rl_env_info = {'running_requests': None,
        # 'waiting_requests': None,
        # 'now_time': None,
        # 'recent_throughput': 0.0,
        # 'recent_avg_latency': 0.0,
        # 'recent_comform_slo_rate': 0.0,
        # 'current_throughput': 0.0,
        # 'last_S':0.0,
        # 'select_S':0.0,
        # 'actual_S':0.0}

        # ========== 1 长期奖励 ==========
        # ---------- 1.1 最近一段时间/n个请求的SLO 满足情况 ----------
        recent_comform_slo_rate = after_env_info.get("recent_comform_slo_rate", 0.0)

        # ---------- 1.2 最近一段时间/n个iteration 吞吐量奖励 ----------
        recent_throughput = after_env_info.get("recent_throughput", 0.0)/ self.config.throughput_norm

        # ========== 2 短期奖励 ==========
        # ---------- 2.1 匹配B、S惩罚 ----------
        select_S = after_env_info.get("select_S", 0)
        actual_S = after_env_info.get("actual_S", 0)
        R_match_S_penalty = 0
        if abs(select_S - actual_S) > 255:
            R_match_S_penalty = -1
        R_match_penalty =  R_match_S_penalty

        # ----------- 2.2 请求在slo内完成奖励,请求违反slo惩罚 ----------
        before_running_req = before_env_info.get("running_requests",[])
        after_running_req = after_env_info.get("running_requests",[])
        after_running_req_ids = [req.request_id for req in after_running_req]
        now_time = after_env_info.get("now_time", 0.0)
        comform_req_count = 0
        violate_req_count = 0
        R_comform_violate = 0
        for req in before_running_req:
            if req.request_id not in after_running_req_ids:
                # 请求已完成
                if now_time - req.arrival_time < req.slo:
                    comform_req_count += 1
                else:
                    violate_req_count += 1
        R_comform_violate = comform_req_count*0.5 - violate_req_count*0.5

        # ---------- 综合 ----------
        reward = (self.config.lambda_recent_comform_slo * recent_comform_slo_rate) + \
                (self.config.lambda_recent_throughput * recent_throughput) + \
                (self.config.lambda_R_match_penalty * R_match_penalty) + \
                (self.config.lambda_R_comform_violate * R_comform_violate)

        return reward