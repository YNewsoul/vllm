import torch
import torch.optim as optim
import numpy as np
import os
import logging
from datetime import datetime
import random
import time
from typing import Dict

from config import RLSchedulerConfig
from RLmodel import MLPNetwork

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
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.lr)  # 低学习率，稳定更新
        self.criterion = torch.nn.MSELoss()  # 均方误差损失（拟合Q值）
        self.train_step = 0    # 训练步数计数器

        self.action_map = [
            (4, 256), (4, 512), (4, 1024), (4, 2048),
            (6, 256), (6, 512), (6, 1024), (6, 2048),
            (8, 256), (8, 512), (8, 1024), (8, 2048),
            (10, 256), (10, 512), (10, 1024), (10, 2048)
        ]

        self.is_ready = True
        self.last_report_monitor_time = 0


    def _initialize_model(self):
        """初始化模型：尝试加载预训练模型或从0训练"""

        # 加载初始模型
        if self.config.rl_model == "MLPNetwork":  
            self.main_model = MLPNetwork(self.config.state_dim, self.config.action_dim).to(self.device)
            self.target_model = MLPNetwork(self.config.state_dim, self.config.action_dim).to(self.device)

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
            
    def select(self, env_info: Dict, running_requests_len: int):

        state = torch.FloatTensor(self.env_info_2_state(env_info)).unsqueeze(0).to(self.device)
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
                return self.action
            
        with torch.no_grad():
            q_values = self.main_model(state).squeeze(0).cpu().numpy()
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
            
            # q_values = self.main_model(state)
            # action = q_values.argmax().item()
            # self.action = self.action_map[action]
            return self.action

    def learn(self, batch) -> float:
        # 1. 正确处理batch数据
        # 先将batch中的每个元素解包，然后将相同类型的数据收集到一起
        before_env_infoes = []
        after_env_infoes = []
        actions = []
        rewards = []
        
        for experience in batch:
            before_env_info, after_env_info, action, reward = experience
            before_env_infoes.append(before_env_info)
            after_env_infoes.append(after_env_info)
            # 需要找到action在action_map中的索引
            action_idx = self.action_map.index(action) if action in self.action_map else 0
            actions.append(action_idx)
            rewards.append(reward)

        # 2. 转换为张量并移动到设备
        states = torch.tensor(np.array([self.env_info_2_state(before_env_info) for before_env_info in before_env_infoes]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([self.env_info_2_state(after_env_info) for after_env_info in after_env_infoes]), dtype=torch.float32).to(self.device)
        
        # 3. 计算预测Q值（主网络）- 修复变量名
        current_q = self.main_model(states).gather(1, actions).squeeze(1)
        
        # 4. 计算目标Q值（目标网络）- 修复变量名
        with torch.no_grad():
            next_q_max = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_max   # 折扣奖励
        

        # 5. 反向传播优化
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 6. 更新目标网络 - 修复变量名
        self.train_step += 1
        if self.train_step % self.target_net_update_freq == 0:
            logger.info(f"update target model, step: {self.train_step}")
            self.target_model.load_state_dict(self.main_model.state_dict())
        
        # 7. 衰减探索概率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        now = time.time()
        if now - self.last_report_monitor_time >= self.config.report_monitor_frequency:
            # 获取损失值
            value_loss = loss.detach().item()
            # 获取目标 Q 值的平均值
            q_value = target_q.mean().detach().item()
            # 获取奖励的平均值
            reward = rewards.mean().detach().item()
            # 打印监控信息
            logger.info(f"value_loss: {value_loss:.4f}, q_value: {q_value:.4f}, reward: {reward:.4f}")
            print(f"value_loss: {value_loss:.4f}, q_value: {q_value:.4f}, reward: {reward:.4f}")
            self.last_report_monitor_time = now
        
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

    def env_info_2_state(self, env_info:Dict):
        
        num_runing = len(env_info["running_requests"])
        num_waiting = len(env_info["waiting_requests"])
        running_ratio = num_runing / (num_runing + num_waiting)
        waiting_ratio = num_waiting / (num_runing + num_waiting)
    
        state_vec = np.array([
            num_runing,
            num_waiting,
            running_ratio,
            waiting_ratio,
        ], dtype=np.float32)

        return state_vec