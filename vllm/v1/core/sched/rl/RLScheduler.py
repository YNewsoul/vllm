import time
from typing import Dict, Any
import logging

try:
    from .RLConfig import RLSchedulerConfig
    from .RLAgent import RLAgent
    from .RLOptimizer import RLOptimizer
    from .Trainer import Trainer
    from .Env import Env
except ImportError:
    from RLConfig import RLSchedulerConfig
    from RLAgent import RLAgent
    from RLOptimizer import RLOptimizer
    from Trainer import Trainer
    from Env import Env

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class RLScheduler:
    """RL调度器主接口"""
    
    def __init__(self,kv_cache_manager):
        """初始化RL调度器
        """
        # 加载配置
        self.config = RLSchedulerConfig.from_env()
        
        # 初始化组件
        self.env = Env()
        self.rl_agent = RLAgent()
        self.optimizer = RLOptimizer(kv_cache_manager)
        self.trainer = Trainer(self.rl_agent)

        # 状态管理
        self.enabled = self.config.enabled
        
        # 统计信息
        self.stats = {
            'total_schedule_calls': 0,
            'avg_optimization_time_ms': 0,
            'successful_optimizations': 0,
            'total_performance_records': 0,
        }
        
        if self.enabled:
            logger.info(f"RL Scheduler initialized successfully!")
        else:
            logger.info("RL Scheduler disabled by configuration")
    
    def compute_schedule_decision(self,env_info:Dict):
        """计算 RL 调度决策 """
        if not self.enabled:
            return None
        # 更新统计信息
        self.stats['total_schedule_calls'] += 1
        
        if self.rl_agent.is_ready:

            # Phase 1:从 RLAgent中选择 batch_size,token_budget
            if self.rl_agent.train_enabled:
                self.env.set_before_env_info(env_info)
            (batch_size, token_budget) = self.rl_agent.select(env_info,len(env_info["running_requests"]))

            # Phase 2:使用 optimizer 计算具体分配
            result = self.optimizer.optimize_schedule(
                running_requests=env_info["running_requests"],
                waiting_requests=env_info["waiting_requests"],
                batch_size=batch_size,
                token_budget=token_budget
            )
            
            if result:
                self.stats['successful_optimizations'] += 1
                
                # 更新优化时间统计
                # self._update_optimization_time_stats(result.optimization_time_ms)
                
                return {
                    'allocation': result.allocation,
                    'token_budget': result.select_S,
                    'prioritize_decode': result.decode_count > 0,
                    'actual_B': result.actual_B,
                    'actual_S': result.actual_S,
                    'select_B': result.select_B,
                    'decode_count': result.decode_count,
                    'prefill_count': result.prefill_count,
                }
            else:
                logger.info(f"RL Scheduler optimize schedule failed")
    
    def get_simple_status(self) -> Dict[str, Any]:
        """获取简化的状态信息，用于快速监控"""
        return {
            'enabled': self.enabled,
            'train':self.rl_agent.train_enabled,
        }
    
    def _update_optimization_time_stats(self, optimization_time_ms: float) -> None:
        """更新优化时间统计"""
        alpha = 0.1  # 指数移动平均权重
        if self.stats['avg_optimization_time_ms'] == 0:
            self.stats['avg_optimization_time_ms'] = optimization_time_ms
        else:
            self.stats['avg_optimization_time_ms'] = (
                alpha * optimization_time_ms + 
                (1 - alpha) * self.stats['avg_optimization_time_ms']
            )

    def record_performance(self, env_info: Dict[str, Any]) -> None:
        """记录RL调度器性能"""
        if self.rl_agent.train_enabled:
            # 训练模式开启时才需要存经验
            self.env.set_after_env_info(env_info)
            before_env_info, after_env_info = self.env.get_env_info()
            self.trainer.add_exp(before_env_info, after_env_info, self.rl_agent.action)
            
        self.stats['total_performance_records'] += 1