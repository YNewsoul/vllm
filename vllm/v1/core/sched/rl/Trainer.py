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
        reward = 0.0

        # 获取信息
        before_running_req = before_env_info.get("running_requests",[])
        before_waiting_req = before_env_info.get("waiting_requests",[])
        before_time = before_env_info.get("now_time")

        after_running_req = after_env_info.get("running_requests",[])
        after_req_ids = [req.request_id for req in after_running_req]
        after_time = after_env_info.get("now_time")
        model_run_time = after_env_info.get("model_run_time",0.0)
        select_token_budget = after_env_info.get("select_token_budget", 0)

        # ========== 1 短期奖励 ==========
        rew_decode = 0
        rew_prefill = 0
        rew_finish = 0
        finish_count = 0
        total_prompt_count = 0
        for req in before_running_req:
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens>= 0:
                # decode 阶段请求
                if req.request_id not in after_req_ids:
                    # 该请求在after_running_req中不存在，说明该请求在此轮完成
                    if after_time - req.arrival_time <= req.slo:
                        rew_finish += 1
                    else:
                        rew_finish -= 1
                    finish_count += 1
                elif ((req.max_tokens/2 - output_tokens)*model_run_time/1000) <= (req.slo-(before_time-req.arrival_time)):
                    rew_decode += 1
                else:
                    rew_decode -= 1
            else:
                # prefill 请求
                total_prompt_count += req.num_prompt_tokens - req.num_computed_tokens
        
        for req in before_waiting_req:
            total_prompt_count += req.num_prompt_tokens

        rew_decode /= (len(before_running_req)-1)
        rew_finish = rew_finish/finish_count if finish_count > 0 else 0
        rew_prefill = float(f"{select_token_budget/2048:.3f}")*(total_prompt_count/20480)

        # ========== 2 长期奖励 ==========
        # ---------- 2.1 最近一段时间/n个请求的SLO 满足情况 ----------
        recent_comform_slo_rate = after_env_info.get("recent_comform_slo_rate", 0.0)

        # ---------- 综合 ----------
        reward = (self.config.lambda_recent_comform_slo * recent_comform_slo_rate) + \
                (self.config.lambda_decode * float(f"{rew_decode:.3f}")) + \
                (self.config.lambda_prefill * rew_prefill) + \
                (self.config.lambda_finish * rew_finish)

        return reward