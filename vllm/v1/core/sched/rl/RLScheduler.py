import time
from typing import Dict, Any
import logging
import os
import time
import json
from datetime import datetime

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
            'total_performance_records': 0,
        }
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.time_out_info_path = os.path.join(current_dir, "timeout_info.jsonl")
        
        if self.enabled:
            logger.info(f"RL Scheduler initialized successfully!")
        else:
            logger.info("RL Scheduler disabled by configuration")
    
    def compute_schedule_decision(self,env_info:Dict):
        """计算 RL 调度决策 """
        
        # Phase 1:从 RLAgent中选择 token_budget
        time1 = time.monotonic()
        token_budget = self._RL_schedule_judge(env_info)
        use_rl_scheduler = False
        if token_budget is None:
            if self.rl_agent.train_enabled:
                self.env.set_before_env_info(env_info)
            token_budget = self.rl_agent.select(env_info)
            use_rl_scheduler = True
            self.stats['total_schedule_calls'] += 1
        time2 = time.monotonic()
        time_rl_agent_select = (time2 - time1)*1000

        # Phase 2:使用 optimizer 计算具体分配
        result = self.optimizer.optimize_schedule(
            running_requests=env_info["running_requests"],
            waiting_requests=env_info["waiting_requests"],
            token_budget=token_budget
        )
        time3 = time.monotonic()
        time_optimizer_optimize_schedule = (time3 - time2)*1000

        if time_optimizer_optimize_schedule >=5 or time_rl_agent_select>=2 :
            self._write_timeout_info(time_optimizer_optimize_schedule,time_rl_agent_select)
        
        if result:
            
            # 更新优化时间统计
            # self._update_optimization_time_stats(result.optimization_time_ms)
            
            return {
                'allocation': result.allocation,
                'token_budget': result.select_token_budget,
                'prioritize_decode': result.decode_count > 0,
                'actual_token_budget': result.actual_token_budget,
                'decode_count': result.decode_count,
                'prefill_count': result.prefill_count,
                'use_rl_scheduler': use_rl_scheduler,
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

    def _write_timeout_info(self,time_optimizer_optimize_schedule,time_rl_agent_select):
        
        timestamp = time.time()
        data = {
            "timestamp":f"{timestamp:.3f}",
            "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_optimizer_optimize_schedule": f"{time_optimizer_optimize_schedule:.3f}",
            "time_rl_agent_select": f"{time_rl_agent_select:.3f}",
        }
        try:
            with open(self.time_out_info_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write timeout info : {e}")

    def _RL_schedule_judge(self,env_info):
        running = env_info['running_requests']
        waiting = env_info['waiting_requests']
        max_num_scheduled_tokens = env_info['max_num_scheduled_tokens']
        num_running = len(running)
        num_waiting = len(waiting)
        if num_running == 0 and num_waiting == 0:
            # 1、 无请求运行，无等待请求
            return max_num_scheduled_tokens
        elif num_running == 0 and num_waiting != 0:
            # 2、新请求到来，且此时没有正在运行的请求
            return max_num_scheduled_tokens
        else:
            num_prefill,num_decode = self._count_running_types(running)
            
            if num_running!=0 and num_waiting == 0:
                # 3、 有请求运行，无等待请求
                if num_prefill != 0 and num_decode == 0:
                    # 3.1、 运行请求只含一个请求，且处于 prefill 阶段
                    return max_num_scheduled_tokens
                elif num_prefill == 0 and num_decode != 0:
                    # 3.2、 所有运行请求都处于 decode 阶段
                    return max_num_scheduled_tokens
                # 3.3、运行请求有 prefill 和 decode 请求，需要RL scheduler
                return None
            
            elif num_running != 0 and num_waiting != 0:
                # 4、 有请求运行，有等待请求
                if num_prefill != 0 and num_decode == 0:
                    # 4.1 运行的请求只含一个请求，且处于 prefill 阶段，且无 decode 阶段请求
                    return max_num_scheduled_tokens
                elif num_prefill == 0 and num_decode != 0:
                    # 4.2 所有运行请求都处于 decode 阶段
                    return max_num_scheduled_tokens
                # 4.3 运行的请求有 prefill 和 decode 请求，需要RL scheduler
                return None

    def _count_running_types(self,running):
        num_prefill = 0
        num_decode = 0
         
        for req in running:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                num_decode += 1
            else:
                num_prefill += 1
        return num_prefill,num_decode