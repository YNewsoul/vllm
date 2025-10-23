import threading
import logging
import random
from collections import deque


from vllm.logger import init_logger

from .RLConfig import RLSchedulerConfig
from .RLAgent import RLAgent

logger = init_logger(__name__)

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
        self.rl_replay_buffer.append((before_env_info, after_env_info, action, reward))

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
        """计算奖励（根据环境反馈）"""
        # 简单示例：奖励为动作执行后的奖励值
        reward_num_waiting_request = 0
        if len(after_env_info["waiting_requests"]) < len(before_env_info["waiting_requests"]):
            reward_num_waiting_request += 1
        reward = reward_num_waiting_request*1
        return reward