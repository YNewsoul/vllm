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
            (10, 256), (10, 512), (10, 1024), (10, 2048),
            (12,256), (12,512), (12,1024), (12,2048),
            (14,256), (14,512), (14,1024), (14,2048),
            (16,256), (16,512), (16,1024), (16,2048),
        ]

        self.is_ready = True
        self.last_save_model_time = 0
        self.log_frequency = self.config.log_frequency
        self.model_save_dir = None


    def _initialize_model(self):
        """初始化模型：尝试加载预训练模型或从0训练"""

        # 加载初始模型
        if self.config.rl_model == "MLPNetwork":  
            self.main_model = MLPNetwork(self.config.state_dim, self.config.action_dim).to(self.device)
            self.target_model = MLPNetwork(self.config.state_dim, self.config.action_dim).to(self.device)
            self.state_dim = self.config.state_dim
            self.action_dim = self.config.action_dim
            logger.info(f"Initializing model MLPNetwork successfully!")
        elif self.config.rl_model == "DualAttentionNetwork":
            
            self.main_model = DualAttentionNetwork(self.config.Global_state_dim, self.config.K_waiting, 
                                                   self.config.Feature_waiting, self.config.K_running, 
                                                   self.config.Feature_running, self.config.action_dim).to(self.device)
            self.target_model = DualAttentionNetwork(self.config.Global_state_dim, self.config.K_waiting,
                                                    self.config.Feature_waiting, self.config.K_running, 
                                                    self.config.Feature_running, self.config.action_dim).to(self.device)
            logger.info(f"Initializing model DualAttentionNetwork successfully!")
        else:
            raise ValueError(f"Unsupported rl_model: {self.config.rl_model}")
        
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

            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")

        self.lr = self.config.lr

        # DQN超参数（对齐文档约束）
        self.gamma = self.config.gamma                                    # 折扣因子（长期奖励权重）
        self.epsilon = self.config.epsilon                                # 初始探索概率
        self.epsilon_max_step = self.config.epsilon_max_step              # 最大探索步数
        self.epsilon_min = self.config.epsilon_min                        # 最小探索概率（保留少量试错）
        self.target_net_update_freq = self.config.target_net_update_freq  # 目标网络更新频率
        
        # 标记为已有模型（已加载预训练或初始化）
        self.is_ready = True
        self._update_count = 1  

    def _initialize_log_file(self):
        # 当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 创建存储日志目录
        training_logs_dir = os.path.join(current_dir, "training_logs")
        os.makedirs(training_logs_dir, exist_ok=True)
        # 创建当前日期目录
        current_time = datetime.now()
        date_dir = os.path.join(training_logs_dir, current_time.strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)
        # 创建具体时间目录
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        time_dir = os.path.join(date_dir, formatted_time)
        os.makedirs(time_dir, exist_ok=True)
        # 创建模型保存目录
        self.model_save_dir = os.path.join(time_dir, "model")
        os.makedirs(self.model_save_dir, exist_ok=True)
        # 写入训练的配置信息
        self._write_training_info(time_dir)
        self.training_log_file_path = os.path.join(time_dir, f"training_log_{formatted_time}.jsonl")
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
            epsilon = max(self.epsilon - self.train_step/self.epsilon_max_step, self.epsilon_min)
            if np.random.rand() < epsilon:
                # 优先从batch_size大于等于running_requests_len的动作中随机选择
                valid_actions = [(i, action) for i, action in enumerate(self.action_map) 
                             if action[0] >= running_requests_len]
            
                # 如果有符合条件的动作，从中随机选择
                if valid_actions:
                    idx, action = random.choice(valid_actions)
                    self.action = action
                    return self.action
                # 否则回退到完全随机选择
                self.action = random.choice(self.action_map)

                return self.action
            
        with torch.no_grad():
            q_values = self.main_model(global_vec, wait_arr, run_arr, wait_mask, run_mask).squeeze(0).cpu().numpy()  # [1, action_dim]
             # 优先从batch_size大于等于running_requests_len的动作中选择Q值最高的
            valid_indices = [i for i, action in enumerate(self.action_map) 
                             if action[0] >= running_requests_len]
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
            log_data = {"time":time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "train_step":self.train_step,
                    "loss":f"{loss.detach().item():.5f}",
                    "avg_target_q":f"{target_q.mean().detach().item():.5f}",
                    "avg_rewards":f"{rewards.mean().detach().item():.5f}"}
            self._write_log_data(log_data)

        now = time.monotonic()
        if now - self.last_save_model_time >= self.config.save_model_frequency:
            self.save_model()
            self.last_save_model_time = now
        
        return loss.item()

    def save_model(self,) -> None:
        """保存模型权重"""
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = os.path.join(self.model_save_dir, f"model_{formatted_time}")
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

    def _write_log_data(self, log_data):
        """写入日志数据"""
        try:
            with open(self.training_log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write the training log data: {e}")
        
    def reset(self):
        self._initialize_log_file()
        self.train_step = 0
        self.last_save_model_time = time.monotonic()
    
    def _write_training_info(self,time_dir):
        """写入训练的配置信息"""
        training_info = {
            "agent":{"device":self.config.device,
                    "rl_model":self.config.rl_model,
                    "use_pretrained_model":self.config.use_pretrained_model,
                    "pretrained_model_path":self.config.pretrained_model_path,
                    "save_model_frequency":self.config.save_model_frequency,
                    "log_frequency":self.config.log_frequency,
            },
            "Trainer":{"train_total_time":self.config.train_total_time,
                       "train_batch_size":self.config.train_batch_size,
                       "replay_buffer_size":self.config.replay_buffer_size,
            },
            "DQN":{"lr":self.config.lr,
                   "gamma":self.config.gamma,
                   "epsilon":self.config.epsilon,
                   "epsilon_max_step":self.config.epsilon_max_step,
                   "epsilon_min":self.config.epsilon_min,
                   "target_net_update_freq":self.config.target_net_update_freq,
            },
            "reward":{"lambda_recent_comform_slo":self.config.lambda_recent_comform_slo,
                      "lambda_recent_throughput":self.config.lambda_recent_throughput,
                      "lambda_R_match_penalty":self.config.lambda_R_match_penalty,
                      "lambda_R_comform_violate":self.config.lambda_R_comform_violate,
            },
            "MLPNetwork":{"state_dim":self.config.state_dim,
                          "action_dim":self.config.action_dim,
            },
            "DualAttentionNetwork":{"Global_state_dim":self.config.Global_state_dim,
                                     "K_waiting":self.config.K_waiting,
                                     "Feature_waiting":self.config.Feature_waiting,
                                     "K_running":self.config.K_running,
                                     "Feature_running":self.config.Feature_running
            },
        }
        train_info_path = os.path.join(time_dir, "training_info.jsonl")
        with open(train_info_path, "a", encoding="utf-8") as f:
            json.dump(training_info, f, ensure_ascii=False, indent=4)
