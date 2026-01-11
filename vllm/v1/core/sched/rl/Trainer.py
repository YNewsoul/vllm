import threading
import logging
import random
import time
import math
import numpy as np
from collections import deque
from prometheus_client import start_http_server, Gauge

REW_DECODE = Gauge('rew_decode', 'reward decode')
REW_PREFILL = Gauge('rew_prefill', 'reward prefill')
REWARD = Gauge('reward', 'reward')
REW_BUDGET = Gauge('rew_budget', 'reward budget')



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
        self.rl_replay_buffer = deque(maxlen=self.config.replay_buffer_size)

        # 线程安全相关参数
        self.is_training = False
        self.train_lock = threading.Lock()

        # 训练相关
        self.tpot_slo = self.config.tpot_slo
        self.tpot_start = self.config.tpot_start
        self.train_batch_size = self.config.train_batch_size
        self.train_total_time = self.config.train_total_time
        self.lambda_decode = self.config.lambda_decode
        self.lambda_prefill = self.config.lambda_prefill
        self.lambda_budget = self.config.lambda_budget
        self.progress_d1 = self.config.progress_d1
        self.progress_d2 = self.config.progress_d2
        self.progress_d3 = self.config.progress_d3
        self.progress_d4 = self.config.progress_d4
        self.lambda_progress_d1 = self.config.lambda_progress_d1
        self.lambda_progress_d2 = self.config.lambda_progress_d2
        self.lambda_progress_d3 = self.config.lambda_progress_d3
        self.lambda_progress_d4 = self.config.lambda_progress_d4
        self.lambda_progress_d5 = self.config.lambda_progress_d5

        self.model_time_norm = self.config.model_time_norm
        self.token_budget_norm = self.config.token_budget_norm

    def _train_in_thread(self):
        """在单独线程中执行模型训练"""
        try:
            with self.train_lock:
                self.rl_agent.reset()
                start_time = time.time()
                while True:
                    batch = self._sample_exp(self.train_batch_size)
                    self.rl_agent.learn(batch)
                    if time.time() - start_time > self.train_total_time + 10:
                        break
                logger.info(f"Finished the training process in {self.train_total_time} seconds")
        except Exception as e:
            logger.error(f"Failed to train: {str(e)}")
        finally:
            self.is_training = False

    def add_exp(self, before_env_info,after_env_info,action) :
        """添加经验到回放池（每个经验对应一轮迭代的交互）"""
        # reward = self._caculate_reward(before_env_info,after_env_info)
        reward = self._caculate_reward_v3(before_env_info,after_env_info)
        self.rl_replay_buffer.append((before_env_info, after_env_info, action, reward))

        # 当经验池达到阈值且满足训练间隔时，启动异步训练
        if not self.is_training and len(self.rl_replay_buffer) >= 4*self.train_batch_size :
            self.is_training = True
            logger.info(f"start train ..........")
            # 创建并启动训练线程
            train_thread = threading.Thread(target=self._train_in_thread, daemon=True)
            train_thread.start()
            logger.info(f"Start the asynchronous training thread and the current size of the experience pool: {len(self.rl_replay_buffer)}")

    def _sample_exp(self, batch_size: int):
        """从回放池采样经验（每个经验对应一轮迭代的交互）"""
        return random.sample(self.rl_replay_buffer, batch_size)
    
    def _caculate_reward(self, before_env_info,after_env_info):
        reward = 0.0

        # env 信息
        before_time = before_env_info.get("now_time")
        before_running_req = before_env_info.get("running_requests",[])
        before_waiting_req = before_env_info.get("waiting_requests",[])
        
        after_time = after_env_info.get("now_time")
        model_run_time = after_env_info.get("model_run_time",0.0)
        select_token_budget = after_env_info.get("select_token_budget", 0)

        ttft_scores, tpot_scores = [], []
        total_prompt_count = 0

        def ttft_score(req,prompt_count):
            ttft_remaining_time = (math.ceil(prompt_count/select_token_budget))*model_run_time/1000.0
            slack_ms = req.ttft_slo - (before_time-req.arrival_time)
            if ttft_remaining_time <= slack_ms:
                return 1.0
            else:
                return np.tanh((slack_ms - ttft_remaining_time)/req.ttft_slo)

        for req in before_running_req:
            # 运行的请求
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens>= 0:
                # decode 阶段
                if output_tokens > req.max_tokens*self.tpot_start:
                    tpot = (after_time -req.arrival_time)/(output_tokens+1)*1000.0
                    if tpot <= self.tpot_slo:
                        tpot_scores.append(1.0)
                    else:
                        tpot_scores.append(np.tanh((self.tpot_slo-tpot)/self.tpot_slo))
            else:
                # prefill 阶段
                total_prompt_count += req.num_prompt_tokens - req.num_computed_tokens
                ttft_scores.append(ttft_score(req,total_prompt_count))
                
        for req in before_waiting_req:
            # 等待的请求
            total_prompt_count += req.num_prompt_tokens
            ttft_scores.append(ttft_score(req,total_prompt_count))

        rew_decode = np.mean(tpot_scores) if tpot_scores else 0.0
        rew_prefill = np.mean(ttft_scores) if ttft_scores else 0.0

        rew_budget = 0
        # # 正向奖励
        if rew_decode == 0.0 or rew_decode == 1.0:
            rew_budget = select_token_budget/self.token_budget_norm
        
        # ---------- 综合 ----------
        reward = (self.lambda_decode * rew_decode) + \
                (self.lambda_prefill * rew_prefill) + \
                (self.lambda_budget * rew_budget)
        
        REW_DECODE.set(rew_decode)
        REW_PREFILL.set(rew_prefill)
        REWARD.set(reward)
        REW_BUDGET.set(rew_budget)

        return reward
    
    def _caculate_reward_v2(self, before_env_info,after_env_info):
        reward = 0.0

        # env 信息
        before_time = before_env_info.get("now_time")
        before_running_req = before_env_info.get("running_requests",[])
        before_waiting_req = before_env_info.get("waiting_requests",[])

        after_time = after_env_info.get("now_time")
        model_run_time = after_env_info.get("model_run_time",0.0)
        select_token_budget = after_env_info.get("select_token_budget", 0)

        ttft_scores, tpot_scores = [], []
        total_prompt_count = 0

        for req in before_running_req:
            # 运行的请求
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens>= 0:
                # decode 阶段
                if output_tokens > req.max_tokens*self.tpot_start:
                    tpot = (after_time -req.arrival_time)/(output_tokens+1)*1000.0
                    tpot_scores.append(np.tanh((self.tpot_slo-tpot)/self.tpot_slo))
            else:
                # prefill 阶段
                total_prompt_count += req.num_prompt_tokens - req.num_computed_tokens
                ttft_remaining_time = (math.ceil(total_prompt_count/select_token_budget))*model_run_time/1000.0
                slack_ms = req.ttft_slo - (before_time-req.arrival_time)
                if ttft_remaining_time <= slack_ms:
                    ttft_scores.append(1.0)
                else:
                    ttft_scores.append(np.tanh((slack_ms - ttft_remaining_time)/req.ttft_slo))
                
        for req in before_waiting_req:
            # 等待的请求
            total_prompt_count += req.num_prompt_tokens
            ttft_remaining_time = (math.ceil(total_prompt_count/select_token_budget))*model_run_time/1000
            slack_ms = req.ttft_slo - (before_time-req.arrival_time)
            if ttft_remaining_time <= slack_ms:
                ttft_scores.append(1.0)
            else:
                ttft_scores.append(np.tanh((slack_ms - ttft_remaining_time)/req.ttft_slo))

        rew_decode = np.mean(tpot_scores) if tpot_scores else 0.0
        rew_prefill = np.mean(ttft_scores) if ttft_scores else 0.0

        # 正向奖励
        rew_budget = select_token_budget/self.token_budget_norm

        # ---------- 综合 ----------
        reward = (self.lambda_decode * rew_decode) + \
                (self.lambda_prefill * rew_prefill) + \
                (self.lambda_budget * rew_budget)
        
        REW_DECODE.set(rew_decode)
        REW_PREFILL.set(rew_prefill)
        REWARD.set(reward)
        REW_BUDGET.set(rew_budget)

        return reward

    def _progess_penalty(self,progress):
        if progress <= self.progress_d1:   # 0.3
            return self.lambda_progress_d1  # 0.1
        elif progress <= self.progress_d2: # 0.5 -> 0.45
            return self.lambda_progress_d2  # 0.3 -> 0.5
        elif progress <= self.progress_d3: # 0.65 -> 0.6
            return self.lambda_progress_d3  # 0.5 -> 1
        elif progress <= self.progress_d4: # 0.7
            return self.lambda_progress_d4  # 1.4
        else:
            return self.lambda_progress_d5  # 1.7
        
    def _caculate_reward_v3(self, before_env_info,after_env_info):
        reward = 0.0

        # env 信息
        before_time = before_env_info.get("now_time")
        before_running_req = before_env_info.get("running_requests",[])
        before_waiting_req = before_env_info.get("waiting_requests",[])

        after_time = after_env_info.get("now_time")
        model_run_time = after_env_info.get("model_run_time",0.0)
        select_token_budget = after_env_info.get("select_token_budget", 0)

        ttft_scores, tpot_scores = [], []
        total_prompt_count = 0

        for req in before_running_req:
            # 运行的请求
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens>= 0:
                progress = output_tokens/req.max_tokens
                # decode 阶段
                if progress >= self.tpot_start:
                    tpot = (after_time -req.arrival_time)/(output_tokens+1)*1000.0
                    if tpot <= self.tpot_slo:
                        tpot_scores.append(1.0)
                    else:
                        tpot_scores.append(self._progess_penalty(progress)*np.tanh((self.tpot_slo-tpot)/self.tpot_slo))
            else:
                # prefill 阶段
                total_prompt_count += req.num_prompt_tokens - req.num_computed_tokens
                ttft_remaining_time = (math.ceil(total_prompt_count/select_token_budget))*model_run_time/1000.0
                slack_ms = req.ttft_slo - (before_time-req.arrival_time)
                if ttft_remaining_time <= slack_ms:
                    ttft_scores.append(1.0)
                else:
                    ttft_scores.append(np.tanh((slack_ms - ttft_remaining_time)/req.ttft_slo))
                
        for req in before_waiting_req:
            # 等待的请求
            total_prompt_count += req.num_prompt_tokens
            ttft_remaining_time = (math.ceil(total_prompt_count/select_token_budget))*model_run_time/1000
            slack_ms = req.ttft_slo - (before_time-req.arrival_time)
            if ttft_remaining_time <= slack_ms:
                ttft_scores.append(1.0)
            else:
                ttft_scores.append(np.tanh((slack_ms - ttft_remaining_time)/req.ttft_slo))

        rew_decode = np.mean(tpot_scores) if tpot_scores else 0.0
        rew_prefill = np.mean(ttft_scores) if ttft_scores else 0.0

        # 正向奖励
        rew_budget = select_token_budget/self.token_budget_norm

        # ---------- 综合 ----------
        reward = (self.lambda_decode * rew_decode) + \
                (self.lambda_prefill * rew_prefill) + \
                (self.lambda_budget * rew_budget)
        
        REW_DECODE.set(rew_decode)
        REW_PREFILL.set(rew_prefill)
        REWARD.set(reward)
        REW_BUDGET.set(rew_budget)

        return reward