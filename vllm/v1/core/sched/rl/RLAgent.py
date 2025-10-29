import torch
import torch.optim as optim
import numpy as np
import os
import logging
from datetime import datetime
import random
import time
import json
from typing import Dict

try:
    from .RLConfig import RLSchedulerConfig
    from .RLmodel import MLPNetwork, DualAttentionNetwork
except ImportError:
    from RLConfig import RLSchedulerConfig
    from RLmodel import MLPNetwork, DualAttentionNetwork

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class RLAgent:
    """
    智能体
    核心：ε-贪心策略（探索/利用）、目标网络（稳定训练）、异步学习（低开销）
    """
    def __init__(self,):

        self.config = RLSchedulerConfig.from_env()
        self.device = torch.device(self.config.device)

        # 是否进行训练
        self.train_enabled = self.config.train_enabled

        # 初始化模型
        self._initialize_model()

        # 训练组件（适配轻量网络）
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.config.lr)  # 低学习率，稳定更新
        self.criterion = torch.nn.MSELoss()  # 均方误差损失（拟合Q值）
        self.train_step = 0    # 训练步数计数器

        self.action_map = [
            (4, 256), (4, 512), (4, 1024), (4, 2048),
            (6, 256), (6, 512), (6, 1024), (6, 2048),
            (8, 256), (8, 512), (8, 1024), (8, 2048),
            (10, 256), (10, 512), (10, 1024), (10, 2048)
        ]

        self.is_ready = True
        self.last_save_model_time = 0
        self.log_frequency = self.config.log_frequency


    def _initialize_model(self):
        """初始化模型：尝试加载预训练模型或从0训练"""

        # 加载初始模型
        if self.config.rl_model == "MLPNetwork":  
            self.main_model = MLPNetwork(self.config.state_dim, self.config.action_dim).to(self.device)
            self.target_model = MLPNetwork(self.config.state_dim, self.config.action_dim).to(self.device)
        elif self.config.rl_model == "DualAttentionNetwork":
            
            self.main_model = DualAttentionNetwork(self.config.Global_state_dim, self.config.K_waiting, 
                                                   self.config.Feature_waiting, self.config.K_running, 
                                                   self.config.Feature_running, self.config.action_dim).to(self.device)
            self.target_model = DualAttentionNetwork(self.config.Global_state_dim, self.config.K_waiting,
                                                    self.config.Feature_waiting, self.config.K_running, 
                                                    self.config.Feature_running, self.config.action_dim).to(self.device)
            print("Initializing model DualAttentionNetwork successfully!")
            logger.info(f"Initializing model DualAttentionNetwork successfully!")
        else:
            raise ValueError(f"Unsupported rl_model: {self.config.rl_model}")
        
        if self.config.verbose_logging:
            logger.info(f"Initializing model: {self.config.rl_model}")
        
        # 尝试加载预训练模型参数
        if self.config.use_pretrained_model and self.config.pretrained_model_path:
            try:
                # 获取当前代码文件所在目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, self.config.pretrained_model_path)

                # 更新模型参数
                if os.path.exists(model_path):
                    self.load_model(self.main_model, model_path)
                    self.target_model.load_state_dict(self.main_model.state_dict())

                    if self.config.verbose_logging:
                        logger.info(f"Loaded pretrained model parameters from: {model_path}")

            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
        else:
            if self.config.verbose_logging:
                if not self.config.use_pretrained_model:
                    logger.info("Pretrained model disabled by configuration")
                elif not self.config.pretrained_model_path:
                    logger.info("No pretrained model path specified")
            
        if self.config.rl_model == "MLPNetwork":
            self.state_dim = self.config.state_dim
            self.action_dim = self.config.action_dim
        self.lr = self.config.lr

        # DQN超参数（对齐文档约束）
        self.gamma = self.config.gamma                                    # 折扣因子（长期奖励权重）
        self.epsilon = self.config.epsilon                                # 初始探索概率
        self.epsilon_decay = self.config.epsilon_decay                    # 探索概率衰减率
        self.epsilon_min = self.config.epsilon_min                        # 最小探索概率（保留少量试错）
        self.target_net_update_freq = self.config.target_net_update_freq  # 目标网络更新频率
        
        # 标记为已有模型（已加载预训练或初始化）
        self.is_ready = True
        self._update_count = 1  
    def _initialize_log_file(self):
        training_logs_str = "training_logs"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        training_logs_dir = os.path.join(current_dir, training_logs_str)
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(training_logs_dir, exist_ok=True)
        self.training_log_file_path = os.path.join(training_logs_dir, f"training_log_{formatted_time}.jsonl")
        logger.info(f"The training log file was successfully initialized in {self.training_log_file_path}")
            
    def select(self, env_info: Dict, running_requests_len: int):

        # === 1. 获取网络输入 ===
        global_vec, wait_arr, run_arr, wait_mask, run_mask = self.env_info_to_state(env_info)

        # 转为张量并放到设备
        global_vec = torch.tensor(global_vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, G]
        wait_arr = torch.tensor(wait_arr, dtype=torch.float32, device=self.device).unsqueeze(0)      # [1, K_wait, F_wait]
        run_arr = torch.tensor(run_arr, dtype=torch.float32, device=self.device).unsqueeze(0)        # [1, K_run, F_run]
        wait_mask = torch.tensor(wait_mask, dtype=torch.float32, device=self.device).unsqueeze(0)    # [1, K_wait]
        run_mask = torch.tensor(run_mask, dtype=torch.float32, device=self.device).unsqueeze(0)      # [1, K_run]

        # === 2. 进入评估模式（关闭dropout/bn） ===
        self.main_model.eval()
        if self.train_enabled:
            if np.random.rand() <= self.epsilon:
                # 优先从batch_size大于running_requests_len的动作中随机选择
                valid_actions = [(i, action) for i, action in enumerate(self.action_map) 
                             if action[0] > running_requests_len]
            
                # 如果有符合条件的动作，从中随机选择
                if valid_actions:
                    idx, action = random.choice(valid_actions)
                    self.action = action
                    return self.action
                # 否则回退到完全随机选择
                self.action = random.choice(self.action_map)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                return self.action
            
        with torch.no_grad():
            q_values = self.main_model(global_vec, wait_arr, run_arr, wait_mask, run_mask).squeeze(0).cpu().numpy()  # [1, action_dim]
             # 优先从batch_size大于running_requests_len的动作中选择Q值最高的
            valid_indices = [i for i, action in enumerate(self.action_map) 
                             if action[0] > running_requests_len]
            if valid_indices:
                # 在有效动作中选择Q值最高的
                valid_q_values = q_values[valid_indices]
                best_valid_idx = valid_indices[np.argmax(valid_q_values)]
                self.action = self.action_map[best_valid_idx]
            else:
                # 没有符合条件的动作，选择Q值最高的动作
                best_idx = np.argmax(q_values)
                self.action = self.action_map[best_idx]
            
            return self.action

    def learn(self, batch) -> float:
        # 1. 正确处理batch数据
        # 先将batch中的每个元素解包，然后将相同类型的数据收集到一起
        before_env_infoes = []
        after_env_infoes = []
        actions = []
        rewards = []
        
        # 1.从 batch 解包
        for experience in batch:
            before_env_info, after_env_info, action, reward = experience
            before_env_infoes.append(before_env_info)
            after_env_infoes.append(after_env_info)
            # 需要找到action在action_map中的索引
            action_idx = self.action_map.index(action) if action in self.action_map else 0
            actions.append(action_idx)
            rewards.append(reward)

        # 2.批量转换状态为张量 (global, wait, run)
        global_befores, wait_befores, run_befores, wait_mask_befores, run_mask_befores = [], [], [], [], []
        global_afters,  wait_afters,  run_afters,  wait_mask_afters,  run_mask_afters  = [], [], [], [], []

        for before_env_info in before_env_infoes:
            g, w, r, wm, rm = self.env_info_to_state(before_env_info)
            global_befores.append(g)
            wait_befores.append(w)
            run_befores.append(r)
            wait_mask_befores.append(wm)
            run_mask_befores.append(rm)

        for after_env_info in after_env_infoes:
            g, w, r, wm, rm = self.env_info_to_state(after_env_info)
            global_afters.append(g)
            wait_afters.append(w)
            run_afters.append(r)
            wait_mask_afters.append(wm)
            run_mask_afters.append(rm)

        # 3. 转换为 torch.Tensor
        device = self.device
        globals_before = torch.tensor(np.stack(global_befores), dtype=torch.float32).to(device)
        waits_before   = torch.tensor(np.stack(wait_befores), dtype=torch.float32).to(device)
        runs_before    = torch.tensor(np.stack(run_befores), dtype=torch.float32).to(device)
        wmask_before   = torch.tensor(np.stack(wait_mask_befores), dtype=torch.float32).to(device)
        rmask_before   = torch.tensor(np.stack(run_mask_befores), dtype=torch.float32).to(device)

        globals_after  = torch.tensor(np.stack(global_afters), dtype=torch.float32).to(device)
        waits_after    = torch.tensor(np.stack(wait_afters), dtype=torch.float32).to(device)
        runs_after     = torch.tensor(np.stack(run_afters), dtype=torch.float32).to(device)
        wmask_after    = torch.tensor(np.stack(wait_mask_afters), dtype=torch.float32).to(device)
        rmask_after    = torch.tensor(np.stack(run_mask_afters), dtype=torch.float32).to(device)

        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)   # [B, 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)   
        
        # 4. 计算预测Q值（主网络）
        current_q = self.main_model(globals_before, waits_before, runs_before, wmask_before, rmask_before).gather(1, actions).squeeze(1)
        
        # 5. 计算目标Q值（目标网络）
        with torch.no_grad():
            next_q_max = self.target_model(globals_after, waits_after, runs_after, wmask_after, rmask_after).max(1)[0]
            target_q = rewards + self.gamma * next_q_max   # 折扣奖励
        

        # 6. 反向传播优化
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), 1.0)  # 防止梯度爆炸
        self.optimizer.step()
        
        # 7. 更新目标网络 - 修复变量名
        self.train_step += 1
        if self.train_step % self.target_net_update_freq == 0:
            logger.info(f"update target model, step: {self.train_step}")
            self.target_model.load_state_dict(self.main_model.state_dict())
        
        # 8. 打印监控信息
        if self.train_step % self.log_frequency == 0:
            self.write_log_data(loss.detach().item(),target_q.mean().detach().item(),rewards.mean().detach().item())

        now = time.monotonic()
        if now - self.last_save_model_time >= self.config.save_model_frequency:
            self.save_model()
            self.last_save_model_time = now
        
        return loss.item()

    def save_model(self,) -> None:
        """保存模型权重"""
        model_str = "model"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, model_str)
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"model_{formatted_time}")
        torch.save(self.main_model.state_dict(), save_path)
        logger.info(f"save model successfully, to {save_path}")
    
    def load_model(self, model: torch.nn.Module, model_path: str) -> bool:
        """从模型文件加载网络参数"""
        try:
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查是否是完整的模型权重或者仅state_dict
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            logger.info(f"load model successfully,from {model_path} ")
            return True
        except Exception as e:
            logger.error(f"load model failed,from {model_path}, error: {e}")
            return False

    def env_info_to_state(self, env_info:Dict):

        # global queue data statistics
        now_time = env_info.get("now_time", 0.0)
        running = env_info.get("running_requests", [])
        waiting = env_info.get("waiting_requests", [])
        num_runing = len(running)
        num_waiting = len(waiting)
        total_reqs = num_runing + num_waiting
        running_ratio = num_runing / total_reqs if total_reqs > 0 else 0.0
        waiting_ratio = num_waiting / total_reqs if total_reqs > 0 else 0.0

        # running queue data statistics
        if num_runing > 0:
            r_need_computed = np.array([max(1,r.num_prompt_tokens - r.num_computed_tokens) for r in running], dtype=np.float32)
            r_need_computed_avg = float(r_need_computed.mean()) / self.config.llm_model_len
            r_need_computed_std = float(r_need_computed.std()) / self.config.llm_model_len
        else:
            r_need_computed_avg = r_need_computed_std = 0.0
        
        # waiting queue data statistics
        if num_waiting > 0:
            w_need_computed = np.array([w.num_prompt_tokens for w in waiting], dtype=np.float32)
            w_need_computed_avg = float(w_need_computed.mean()) / self.config.llm_model_len
            w_need_computed_std = float(w_need_computed.std()) / self.config.llm_model_len
        else:
            w_need_computed_avg = w_need_computed_std = 0.0

        # decode 比例
        is_decodes = np.array([1.0 if self._is_decode_phase(r) else 0.0 for r in running], dtype=np.float32)
        frac_decode = float(is_decodes.mean()) if len(is_decodes)>0 else 0.0

        # remaining SLO ratios: for waiting and running

        wait_remaining = []
        for r in waiting:
            elapsed = now_time - r.arrival_time
            remaining =  (r.slo - elapsed)/r.slo
            wait_remaining.append(remaining)
        run_remaining = []
        for r in running:
            elapsed = now_time - r.arrival_time
            remaining =  (r.slo - elapsed)/r.slo
            run_remaining.append(remaining)

        avg_wait_remaining = float(np.mean(wait_remaining)) if wait_remaining else 0.0
        min_wait_remaining = float(np.min(wait_remaining)) if wait_remaining else 0.0
        avg_run_remaining = float(np.mean(run_remaining)) if run_remaining else 0.0
        min_run_remaining = float(np.min(run_remaining)) if run_remaining else 0.0

        # recent aggregated metrics
        recent_throughput = float(env_info.get("recent_throughput", 0.0)) / self.config.throughput_norm
        recent_avg_latency = float(env_info.get("recent_avg_latency", 0.0)) / self.config.slo_norm
        recent_comform_slo_rate = float(env_info.get("recent_comform_slo_rate", 0.0))

        last_B = float(env_info.get("last_B", 0.0))/self.config.B_norm
        last_S = float(env_info.get("last_S", 0.0))/self.config.S_norm
        
        global_vec = np.array([
            num_runing,num_waiting,
            running_ratio,waiting_ratio,
            r_need_computed_avg,r_need_computed_std,
            w_need_computed_avg,w_need_computed_std,
            frac_decode,
            avg_wait_remaining,min_wait_remaining,
            avg_run_remaining,min_run_remaining,
            recent_throughput,recent_avg_latency,recent_comform_slo_rate,
            last_B,last_S,
        ], dtype=np.float32)

        waiting_feats, wait_mask = [], []
        for r in waiting[:self.config.K_waiting]:
            p_len = float(r.num_prompt_tokens) / self.config.prompt_norm
            elapsed = now_time - r.arrival_time
            remaining =  (r.slo - elapsed)/r.slo
            age = elapsed / r.slo
            waiting_feats.append([p_len,remaining, age])
            wait_mask.append(1.0)
        
        # pad waiting
        while len(waiting_feats) < self.config.K_waiting:
            waiting_feats.append([0.0]*3)
            wait_mask.append(0.0)
        wait_arr = np.array(waiting_feats, dtype=np.float32)  # shape (K_wait, 3)
        wait_mask = np.array(wait_mask, dtype=np.float32)

        running_feats, run_mask = [], []
        for r in running[:self.config.K_running]:
            p_len = float(r.num_prompt_tokens) / self.config.prompt_norm
            processed = float(r.num_computed_tokens) / self.config.prompt_norm
            is_dec = 1.0 if self._is_decode_phase(r) else 0.0
            elapsed = now_time - r.arrival_time
            remaining =  (r.slo - elapsed)/r.slo
            age = elapsed / r.slo
            # remaining_prefill_tokens: if available (how many prompt tokens left to prefill), normalize by S_max
            remaining_prefill = max(0.0, r.num_prompt_tokens - r.num_computed_tokens) / self.config.prompt_norm
            # pack: we normalize elapsed by slo as proxy (or by a fixed constant)
            elapsed_norm = min(1.0, elapsed / max(1.0, r.slo))
            running_feats.append([p_len, processed, is_dec, remaining, age, elapsed_norm, remaining_prefill])
            run_mask.append(1.0)
        
        # pad running
        while len(running_feats) < self.config.K_running:
            running_feats.append([0.0]*7)
            run_mask.append(0.0)
        run_arr = np.array(running_feats, dtype=np.float32)  # shape (K_run, 7)
        run_mask = np.array(run_mask, dtype=np.float32)
        
        return global_vec, wait_arr, run_arr, wait_mask, run_mask

    def _is_decode_phase(self, request) -> bool:
        """判断请求是否处于decode阶段"""
        try:
            return request.num_computed_tokens >= request.num_prompt_tokens
        except AttributeError:
            # 如果字段不存在，假设是prefill阶段
            return False
    def write_log_data(self, loss,target_q,rewards):
        """写入日志数据"""
        log_data = {"loss":f"{loss:.5f}",
                    "avg_target_q":f"{target_q:.5f}",
                    "avg_rewards":f"{rewards:.5f}"}
        # 写入日志文件
        try:
            with open(self.training_log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write the training log data: {e}")
        
    def reset(self):
        self._initialize_log_file()
        self.train_step = 0
