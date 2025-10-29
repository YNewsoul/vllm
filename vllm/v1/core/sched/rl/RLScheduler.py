"""SLA感知调度器主接口模块

该模块提供SLA感知调度器的统一对外接口，整合性能预测器和优化器，
为主调度器提供简洁易用的API。确保与现有vLLM调度器完全兼容。
"""

import time
from typing import Optional, Dict, Any, List, Tuple
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
    
    def __init__(self):
        """初始化RL调度器
        
        Args:
            config: RL调度器配置，None时从环境变量加载
        """
        # 加载配置
        self.config = RLSchedulerConfig.from_env()
        
        # 初始化组件
        self.env = Env()
        self.rl_agent = RLAgent()
        self.optimizer = RLOptimizer()
        self.trainer = Trainer(self.rl_agent)

        # 状态管理
        self.enabled = self.config.enabled
        self.initialization_time = time.time()
        
        # 统计信息
        self.stats = {
            'total_schedule_calls': 0,
            'avg_optimization_time_ms': 0,
            'successful_optimizations': 0,
            'last_optimization_result': None,
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
            self.env.set_before_env_info(env_info)
            logger.info(f"the running requests num is {len(env_info['running_requests'])}")
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
                self.stats['last_optimization_result'] = {
                    'select_B': result.select_B,
                    'token_budget': result.select_S,
                    'decode_count': result.decode_count,
                    'prefill_count': result.prefill_count,
                }
                
                # 更新优化时间统计
                self._update_optimization_time_stats(result.optimization_time_ms)
                
                if self.config.verbose_logging:
                    logger.debug(f"RL schedule decision: "
                                f"token_budget={result.select_S}, "
                                f"select_B={result.select_B}, "
                                f"allocation={len(result.allocation)} requests")
                
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
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器完整状态信息"""
        predictor_status = self.predictor.get_status()
        optimizer_stats = self.optimizer.get_stats()
        
        uptime_seconds = time.time() - self.initialization_time
        
        return {
            'enabled': self.enabled,
            'uptime_seconds': uptime_seconds,
            'config': self.config.to_dict(),
            'predictor': predictor_status,
            'optimizer': optimizer_stats,
            'stats': self.stats.copy(),
            'error_state': {
                'consecutive_errors': self._consecutive_errors,
                'in_recovery': self._is_in_error_recovery(),
                'recovery_time_remaining': max(0, self._error_recovery_time - time.time()),
            }
        }
    
    def get_simple_status(self) -> Dict[str, Any]:
        """获取简化的状态信息，用于快速监控"""
        return {
            'enabled': self.enabled,
            'avg_optimization_time_ms': self.stats['avg_optimization_time_ms'],
            'last_optimization_result': self.stats['last_optimization_result'],
            'train':self.rl_agent.train_enabled,
        }
    
    def reset(self) -> None:
        """重置调度器状态"""
        self.predictor.reset()
        self.optimizer.reset_stats()
        
        self.stats = {
            'total_schedule_calls': 0,
            'successful_optimizations': 0,
            'fallback_count': 0,
            'total_performance_records': 0,
            'avg_optimization_time_ms': 0,
            'last_optimization_result': None,
        }
        
        self._consecutive_errors = 0
        self._error_recovery_time = 0
        
        logger.info("SLA Scheduler reset")
    
    
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
    
    def enable(self) -> None:
        """启用RL调度器"""
        self.enabled = True
        logger.info("RL Scheduler enabled")
    
    def disable(self) -> None:
        """禁用RL调度器"""
        self.enabled = False
        logger.info("RL Scheduler disabled")
    
    def update_config(self, new_config: RLSchedulerConfig) -> None:
        """动态更新配置
        
        Args:
            new_config: 新的配置
        """
        old_enabled = self.enabled
        
        self.config = new_config
        self.enabled = new_config.enabled
        
        # 如果启用状态发生变化，记录日志
        if old_enabled != self.enabled:
            if self.enabled:
                logger.info("SLA Scheduler enabled by config update")
            else:
                logger.info("SLA Scheduler disabled by config update")
        
        logger.info(f"SLA Scheduler config updated: {new_config}")
    
    def record_performance(self, env_info: Dict[str, Any]) -> None:
        """记录RL调度器性能"""
        if self.rl_agent.train_enabled:
            # 训练模式开启时才需要存经验
            self.env.set_after_env_info(env_info)
            before_env_info, after_env_info = self.env.get_env_info()
            self.trainer.add_exp(before_env_info, after_env_info, self.rl_agent.action)
            
        self.stats['total_performance_records'] += 1