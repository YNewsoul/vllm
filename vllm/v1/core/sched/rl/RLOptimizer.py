import time
from typing import List, Dict, Tuple, Optional, Any
import sys
import os
import logging
from dataclasses import dataclass

try:
    from .RLConfig import RLSchedulerConfig
except ImportError:
    from RLConfig import RLSchedulerConfig

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class RLOptimizationResult:
    """RL调度结果数据类
    
    输出结果，用于调度器进行资源分配决策。
    """
    select_token_budget: int     # 选择的 token预算
    allocation: Dict[str, int]   # request_id -> tokens的分配映射
    optimization_time_ms: float  # 调度执行时间
    actual_token_budget: int     # 实际分配的token预算


class RLOptimizer:
    """RL优化器"""
    
    def __init__(self, kv_cache_manager):
        """初始化优化器"""
        self.config = RLSchedulerConfig.from_env()
        
        # 用于获取 kv cache
        self.kv_cache_manager = kv_cache_manager
    
    def optimize_schedule(self,
                          running_requests,
                          waiting_requests,
                          token_budget: int) -> Optional[RLOptimizationResult]:

        # 贪心分配资源

        allocation = self._greedy_allocation(running_requests, waiting_requests, token_budget)

        # 验证分配结果
        actual_token_budget = sum(allocation.values())

        result = None
        if actual_token_budget <= token_budget:
            # decode_count, prefill_count = self._count_request_types(
            #                 running_requests, waiting_requests, allocation
            #             )
            result = RLOptimizationResult(
                            select_token_budget=token_budget,
                            allocation=allocation,
                            optimization_time_ms=0,  # 稍后设置
                            actual_token_budget=actual_token_budget,
                        )
        if result:
            return result
        else:
            logger.info(f"Optimization failed: no valid allocation found")
            return None
    
    def _greedy_allocation(self, 
                          running_requests,
                          waiting_requests,
                          token_budget: int) -> Dict[str, int]:
        """贪心分配算法"""

        allocation = {}
        remaining_budget = token_budget
        
        # Phase 1: 分类running请求
        decode_requests = []
        prefill_requests = []
        
        for req in running_requests:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                decode_requests.append(req)
            else:
                prefill_requests.append(req)
        
        if len(prefill_requests)>1:
            logger.info(f"The length of prefill_requests is {len(prefill_requests)}")
        
        # Phase 2: 优先分配decode请求（每个1 token）
        for req in decode_requests:
            if remaining_budget >= 1 :
                allocation[req.request_id] = 1
                remaining_budget -= 1
            else:
                allocation[req.request_id] = 0
        
        # Phase 3: 分配running prefill请求
        
        for req in prefill_requests:
            if remaining_budget <= 0 :
                allocation[req.request_id] = 0
                continue
            
            remaining_tokens = req.num_prompt_tokens - req.num_computed_tokens
            chunk_size = min(remaining_tokens, remaining_budget)
            
            allocation[req.request_id] = chunk_size
            remaining_budget -= chunk_size
        
        # Phase 4: 选择waiting请求
        if remaining_budget > 0 :
            selected_waiting = self._select_waiting_requests(waiting_requests, remaining_budget)
            
            for req, tokens in selected_waiting:
                allocation[req.request_id] = tokens
                remaining_budget -= tokens
        
        return allocation
    
    def _select_waiting_requests(self, waiting_requests,remaining_budget: int) :
        
        if not waiting_requests :
            return []
        
        # 按优先级排序（如果有优先级字段）
        try:
            sorted_waiting = sorted(waiting_requests, 
                                  key=lambda req: getattr(req, 'priority', 0), 
                                  reverse=True)
        except AttributeError:
            # 如果没有priority字段，按FIFO顺序
            sorted_waiting = waiting_requests
            logger.info("Sorting waiting requests by priority, FIFO order")
        
        selected = []
        
        for req in sorted_waiting:
            if remaining_budget <= 0:
                break
            
            # 新的分配方式，考虑kv cache
            if req.num_computed_tokens == 0:
                # 新的请求，获取本地缓存的token
                new_computed_blocks, num_computed_tokens = self.kv_cache_manager.get_computed_blocks(req)
            else:
                num_computed_tokens = req.num_computed_tokens
            num_new_tokens = req.num_tokens - num_computed_tokens
            chunk_size = min(num_new_tokens, remaining_budget)
            selected.append((req, chunk_size))
            remaining_budget -= chunk_size

        return selected
    
    def _count_request_types(self, 
                           running_requests,
                           waiting_requests,
                           allocation: Dict[str, int]) -> Tuple[int, int]:
        """统计分配结果中的decode和prefill请求数量"""
        decode_count = 0
        prefill_count = 0
        
        for req in running_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                if req.num_computed_tokens >= req.num_prompt_tokens:
                    decode_count += 1
                else:
                    prefill_count += 1
        
        for req in waiting_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                prefill_count += 1  # 新请求都是prefill
        
        return decode_count, prefill_count