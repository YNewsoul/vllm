import threading
import logging
import random
import time
import math
import numpy as np
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
        reward = 0.0

        # 获取信息
        before_running_req = before_env_info.get("running_requests",[])
        before_waiting_req = before_env_info.get("waiting_requests",[])
        before_time = before_env_info.get("now_time")

        after_time = after_env_info.get("now_time")
        model_run_time = after_env_info.get("model_run_time",0.0)
        select_token_budget = after_env_info.get("select_token_budget", 0)

        # ========== 1 短期奖励 ==========
        rew_decode = 0
        rew_prefill = 0
        total_prompt_count = 0

        for req in before_running_req:
            # 运行的请求
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens>= 0:
                # decode 阶段请求
                # 当超过20%的时候再考虑tpot，因为有可能一开始tpot就超了
                if output_tokens > req.max_tokens*0.2:
                    tpot = (after_time -req.arrival_time)/(output_tokens+1)*1000
                    if tpot <= 50:
                        rew_decode += 1
                    else:
                        rew_decode += np.tanh((50-tpot)/50)
            else:
                # prefill 请求
                total_prompt_count += req.num_prompt_tokens - req.num_computed_tokens
                ttft_remaining_time = (math.ceil(total_prompt_count/select_token_budget))*model_run_time/1000
                if ttft_remaining_time <= (req.ttft_slo-(before_time-req.arrival_time)):
                    rew_prefill += 1
                else:
                    rew_prefill -=  np.tanh((ttft_remaining_time + before_time-req.arrival_time -req.ttft_slo)/req.ttft_slo)
                
        for req in before_waiting_req:
            # 等待的请求
            total_prompt_count += req.num_prompt_tokens
            ttft_remaining_time = (math.ceil(total_prompt_count/select_token_budget))*model_run_time/1000
            if ttft_remaining_time <= (req.ttft_slo-(before_time-req.arrival_time)):
                rew_prefill += 1
            else:
                rew_prefill -=  np.tanh((ttft_remaining_time + before_time-req.arrival_time -req.ttft_slo)/req.ttft_slo)

        rew_decode /= (len(before_running_req)-1)
        rew_prefill = rew_prefill/(1 + len(before_waiting_req))

        # 正向奖励
        rew_token_budget = select_token_budget/2048

        # ---------- 综合 ----------
        reward = (self.config.lambda_decode * float(f"{rew_decode:.3f}")) + \
                (self.config.lambda_prefill * float(f"{rew_prefill:.3f}")) + \
                rew_token_budget

        return reward