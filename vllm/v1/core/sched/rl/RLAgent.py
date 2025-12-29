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
from prometheus_client import start_http_server, Gauge

TRAIN_STEP = Gauge('train_step', 'Training step counter')
LOSS = Gauge('loss', 'Training loss')
AVG_TARGET_Q = Gauge('avg_target_q', 'Average target Q value')
AVG_REWARDS = Gauge('avg_rewards', 'Average rewards')


try:
    from .RLConfig import RLSchedulerConfig
    from .RLmodel import  DualAttentionNetwork
except ImportError:
    from RLConfig import RLSchedulerConfig
    from RLmodel import  DualAttentionNetwork

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

        self._initialize_config()
        self._initialize_model()

        # 其他参数
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.lr)  # 低学习率，稳定更新
        self.criterion = torch.nn.MSELoss()  # 均方误差损失（拟合Q值）
        self.train_step = 0    # 训练步数计数器
        self.action_map = [64,128,256,512,1024,2048]
        self.last_save_model_time = 0

        if self.train_enabled:
            start_http_server(8768)
            self._initialize_log_file()

    def _initialize_config(self):
        """初始化配置"""

        # 整体参数
        self.device = torch.device(self.config.device)
        self.train_enabled = self.config.train_enabled
        self.rl_model = self.config.rl_model
        self.use_pretrained_model = self.config.use_pretrained_model
        self.pretrained_model_path = self.config.pretrained_model_path
        self.model_save_frequency = self.config.model_save_frequency
        self.log_frequency = self.config.log_frequency
        self.action_dim = self.config.action_dim
        self.tpot_slo = self.config.tpot_slo
        self.tpot_start = self.config.tpot_start

        # DQN 参数
        self.lr = self.config.lr
        self.gamma = self.config.gamma                                    # 折扣因子（长期奖励权重）
        self.epsilon = self.config.epsilon                                # 初始探索概率
        self.epsilon_max_step = self.config.epsilon_max_step              # 最大探索步数
        self.epsilon_min = self.config.epsilon_min                        # 最小探索概率（保留少量试错）
        self.target_net_update_freq = self.config.target_net_update_freq  # 目标网络更新频率

        # DualAttentionNetwork 参数
        self.Global_state_dim = self.config.Global_state_dim
        self.K_waiting = self.config.K_waiting
        self.Feature_waiting = self.config.Feature_waiting
        self.K_running = self.config.K_running
        self.Feature_running = self.config.Feature_running

        # === env_to_state 参数 ===
        self.prompt_norm = self.config.prompt_norm              # 提示归一化因子
        self.token_budget_norm = self.config.token_budget_norm
        self.model_time_norm = self.config.model_time_norm

    def _initialize_model(self):
        """初始化模型：尝试加载预训练模型或从0训练"""

        # 加载初始模型
        if self.rl_model == "DualAttentionNetwork":
            
            self.main_model = DualAttentionNetwork(self.Global_state_dim, self.K_waiting, 
                                                   self.Feature_waiting, self.K_running, 
                                                   self.Feature_running, self.action_dim).to(self.device)
            self.target_model = DualAttentionNetwork(self.Global_state_dim, self.K_waiting, 
                                                   self.Feature_waiting, self.K_running, 
                                                   self.Feature_running, self.action_dim).to(self.device)
            logger.info(f"Initializing model DualAttentionNetwork successfully!")

        # 尝试加载预训练模型参数
        if self.use_pretrained_model and self.pretrained_model_path:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, self.pretrained_model_path)

                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    # 检查是否是完整的模型权重或者仅state_dict
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.main_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.main_model.load_state_dict(checkpoint)
                    self.target_model.load_state_dict(self.main_model.state_dict())
                    logger.info(f"load model successfully,from {model_path} ")
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")

    def _initialize_log_file(self):
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
            
    def select(self, env_info: Dict):

        # === 1. 获取网络输入 ===
        global_vec, wait_arr, run_arr, wait_mask, run_mask, scenario_id = self.env_info_to_state(env_info)

        # 转为张量并放到设备
        scenario_id = torch.tensor(scenario_id, dtype=torch.long, device=self.device)  # [1]
        global_vec = torch.from_numpy(global_vec).float().to(self.device).unsqueeze(0)  # [1, G]
        wait_arr = torch.from_numpy(wait_arr).float().to(self.device).unsqueeze(0)      # [1, K_wait, F_wait]
        run_arr = torch.from_numpy(run_arr).float().to(self.device).unsqueeze(0)        # [1, K_run, F_run]
        wait_mask = torch.from_numpy(wait_mask).float().to(self.device).unsqueeze(0)    # [1, K_wait]
        run_mask = torch.from_numpy(run_mask).float().to(self.device).unsqueeze(0)      # [1, K_run]

        # === 2. 进入评估模式（关闭dropout/bn） ===
        self.main_model.eval()
        if self.train_enabled:
            epsilon = max(self.epsilon - self.train_step/self.epsilon_max_step, self.epsilon_min)
            if np.random.rand() < epsilon:
                self.action = random.choice(self.action_map)
                return self.action
            
        with torch.inference_mode():
            q_values = self.main_model(global_vec, wait_arr, run_arr, wait_mask, run_mask, scenario_id).squeeze(0).cpu().numpy()  # [1, action_dim]
            
            best_idx = np.argmax(q_values)
            self.action = self.action_map[best_idx]
            
            return self.action

    def learn(self, batch) -> float:
        self.main_model.train()
        # 1. 正确处理batch数据
        # 先将batch中的每个元素解包，然后将相同类型的数据收集到一起
        before_env_infoes,after_env_infoes,actions,rewards = [],[],[],[]
        
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
        global_befores, wait_befores, run_befores, wait_mask_befores, run_mask_befores,scenario_befores  = [], [], [], [], [],[]
        global_afters,  wait_afters,  run_afters,  wait_mask_afters,  run_mask_afters ,scenario_afters  = [], [], [], [], [],[]

        for before_env_info in before_env_infoes:
            g, w, r, wm, rm ,sc = self.env_info_to_state(before_env_info)
            global_befores.append(g)
            wait_befores.append(w)
            run_befores.append(r)
            wait_mask_befores.append(wm)
            run_mask_befores.append(rm)
            scenario_befores.append(sc)

        for after_env_info in after_env_infoes:
            g, w, r, wm, rm ,sc= self.env_info_to_state(after_env_info)
            global_afters.append(g)
            wait_afters.append(w)
            run_afters.append(r)
            wait_mask_afters.append(wm)
            run_mask_afters.append(rm)
            scenario_afters.append(sc)

        # 3. 转换为 torch.Tensor
        device = self.device
        globals_before = torch.from_numpy(np.stack(global_befores)).float().to(device)
        waits_before   = torch.from_numpy(np.stack(wait_befores)).float().to(device)
        runs_before    = torch.from_numpy(np.stack(run_befores)).float().to(device)
        wmask_before   = torch.from_numpy(np.stack(wait_mask_befores)).float().to(device)
        rmask_before   = torch.from_numpy(np.stack(run_mask_befores)).float().to(device)
        scenarios_before = torch.from_numpy(np.array(scenario_befores)).long().to(device).view(-1)

        globals_after  = torch.from_numpy(np.stack(global_afters)).float().to(device)
        waits_after    = torch.from_numpy(np.stack(wait_afters)).float().to(device)
        runs_after     = torch.from_numpy(np.stack(run_afters)).float().to(device)
        wmask_after    = torch.from_numpy(np.stack(wait_mask_afters)).float().to(device)
        rmask_after    = torch.from_numpy(np.stack(run_mask_afters)).float().to(device)
        scenarios_after = torch.from_numpy(np.array(scenario_afters)).long().to(device).view(-1)

        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)   # [B, 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)   
        
        # 4. 计算预测Q值（主网络）
        current_q = self.main_model(
                globals_before, waits_before, runs_before, wmask_before, rmask_before ,scenarios_before
            ).gather(1, actions).squeeze(1)
        
        # 5. 计算目标Q值（目标网络）
        with torch.no_grad():
            next_q_max = self.target_model(
                globals_after, waits_after, runs_after, wmask_after, rmask_after ,scenarios_after
            ).max(1)[0]
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

        now = time.time()
        if now - self.last_save_model_time >= self.model_save_frequency:
            self.save_model()
            self.last_save_model_time = now
            
        # 更新Prometheus指标
        TRAIN_STEP.set(self.train_step)
        LOSS.set(loss.detach().item())
        AVG_TARGET_Q.set(target_q.mean().detach().item())
        AVG_REWARDS.set(rewards.mean().detach().item())
        
        return loss.item()

    def save_model(self,) -> None:
        """保存模型权重"""
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = os.path.join(self.model_save_dir, f"model_{formatted_time}")
        torch.save(self.main_model.state_dict(), save_path)
        logger.info(f"save model successfully, to {save_path}")
    
    def env_info_to_state(self, env_info: Dict):
        now_time = env_info.get("now_time", 0.0)
        running = env_info.get("running_requests", [])
        waiting = env_info.get("waiting_requests", [])
        scenario_id = np.array([env_info.get("scenario_id", 0)])

        # ---- Global 指标 ----
        recent_comform_slo_rate = float(env_info.get("recent_comform_slo_rate", 0.0))
        last_model_run_time = float(env_info.get("last_model_run_time", 0.0))/self.model_time_norm
        last_token_budget = float(env_info.get("last_token_budget", 0.0))/self.token_budget_norm

        # ---- 全局向量 ----
        global_vec = np.array([
            recent_comform_slo_rate, 
            last_model_run_time, 
            last_token_budget], dtype=np.float32)

        # ---- 运行队列 ----
        running_feats, run_mask = [], []
        remain_prefill_tokens = 0

        for r in running[:self.K_running]:
            output_tokens = r.num_computed_tokens - r.num_prompt_tokens
            if output_tokens >= 0:
                # decode请求
                # 1.当前 TPOT
                if output_tokens == 0:
                    tpot_status = 0.0
                else:
                    tpot = (now_time - r.ttft)/output_tokens*1000
                    tpot_status = np.tanh((self.tpot_slo-tpot)/self.tpot_slo)
                # 2.进度比例
                # tpot_start_tokens = (r.max_tokens*self.tpot_start)
                # progress = np.tanh((output_tokens - tpot_start_tokens) / tpot_start_tokens)
                progress = output_tokens/r.max_tokens
                running_feats.append([progress, tpot_status,1.0])
            else:
                remain_prefill_tokens -= output_tokens
                remaining_prefill  = np.tanh(remain_prefill_tokens/ self.prompt_norm)
                slack_ms = r.ttft_slo - (now_time - r.arrival_time)
                urgency = np.tanh((slack_ms) / r.ttft_slo)
                running_feats.append([remaining_prefill,urgency,-1.0])
            run_mask.append(1.0)
        while len(running_feats) < self.K_running:
            running_feats.append([0.0,0.0,0.0])
            run_mask.append(0.0)


        waiting_feats, wait_mask = [], []
        # ---- 等待队列 top-K ----
        for r in waiting[:self.K_waiting]:
            remain_prefill_tokens += r.num_prompt_tokens
            remaining_prefill = np.tanh(remain_prefill_tokens / self.prompt_norm)
            slack_ms = r.ttft_slo - (now_time - r.arrival_time)
            urgency = np.tanh(slack_ms / r.ttft_slo)
            waiting_feats.append([remaining_prefill, urgency,-1.0])
            wait_mask.append(1.0)
        while len(waiting_feats) < self.K_waiting:
            waiting_feats.append([0.0,0.0,0.0])
            wait_mask.append(0.0)

        wait_arr = np.array(waiting_feats, dtype=np.float32)
        wait_mask = np.array(wait_mask, dtype=np.float32)
        run_arr = np.array(running_feats, dtype=np.float32)
        run_mask = np.array(run_mask, dtype=np.float32)

        return global_vec, wait_arr, run_arr, wait_mask, run_mask, scenario_id

    def _write_log_data(self, log_data):
        """写入日志数据"""
        try:
            with open(self.training_log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write the training log data: {e}")
        
    def reset(self):
        self.train_step = 0
        self.last_save_model_time = time.time()
    
    def _write_training_info(self,time_dir):
        """写入训练的配置信息"""
        training_info = self.config.to_dict()
        train_info_path = os.path.join(time_dir, "training_info.jsonl")
        with open(train_info_path, "a", encoding="utf-8") as f:
            json.dump(training_info, f, ensure_ascii=False, indent=4)
