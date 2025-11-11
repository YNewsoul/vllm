import numpy as np
import time
import logging
from typing import Dict, List, Any
from collections import namedtuple


try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class RequestSnapshot(namedtuple('RequestSnapshot', ['request_id', 'num_computed_tokens', 
                                'num_prompt_tokens', 'arrival_time', 'slo'])):
    """请求的轻量级快照，只包含RL决策所需的关键信息"""
    __slots__ = ()

class Env:
    """
    模拟文档中的LLM推理引擎环境
    核心适配：动态workload、GPU硬件约束（KV缓存/批大小）、迭代式调度（tens到hundreds毫秒）
    """
    def __init__(self):
        
        self.before_env_info = None # 上一个step的环境信息
        self.after_env_info =  None # 当前step的环境信息
    
    def _create_env_snapshot(self, env_info: Dict) -> Dict:
        """创建环境信息的轻量级快照，只复制必要信息"""
        snapshot = {
            'now_time': env_info['now_time'],
            'recent_throughput': env_info.get('recent_throughput', 0.0),
            'recent_avg_latency': env_info.get('recent_avg_latency', 0.0),
            'recent_comform_slo_rate': env_info.get('recent_comform_slo_rate', 0.0),
            'current_throughput': env_info.get('current_throughput', 0.0),
            'last_token_budget': env_info.get('last_token_budget', 0.0),
            'select_token_budget': env_info.get('select_token_budget', 0.0),
            'actual_token_budget': env_info.get('actual_token_budget', 0.0),
            # 仅创建请求的轻量级快照
            'running_requests': [
                RequestSnapshot(
                    req.request_id,
                    req.num_computed_tokens,
                    req.num_prompt_tokens,
                    req.arrival_time,
                    req.slo
                ) for req in env_info.get('running_requests', [])
            ],
            'waiting_requests': [
                RequestSnapshot(
                    req.request_id,
                    req.num_computed_tokens,
                    req.num_prompt_tokens,
                    req.arrival_time,
                    req.slo
                ) for req in env_info.get('waiting_requests', [])
            ]
        }
        return snapshot

    def set_before_env_info(self,before_env_info:Dict):
        self.before_env_info = None
        time_1 = time.monotonic()
        self.before_env_info = self._create_env_snapshot(before_env_info)
        time_2 = time.monotonic()
        
    def set_after_env_info(self,after_env_info:Dict):
        self.after_env_info = None
        self.after_env_info = self._create_env_snapshot(after_env_info)

    def get_env_info(self):
        return self.before_env_info, self.after_env_info

    def get_before_env_info(self):
        return self.before_env_info 
