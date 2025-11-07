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
    select_S: int                # 选择的 token预算
    allocation: Dict[str, int]   # request_id -> tokens的分配映射
    optimization_time_ms: float  # 调度执行时间
    actual_S: int                # 实际分配的token预算
    decode_count: int            # decode请求数量
    prefill_count: int           # prefill请求数量


class RLOptimizer:
    """RL优化器"""
    
    def __init__(self, kv_cache_manager):
        """初始化优化器"""
        self.config = RLSchedulerConfig.from_env()
        
        # 调度的统计信息
        self.stats = {
            'total_optimizations': 0,
            'avg_optimization_time_ms': 0.0,
            'timeout_count': 0,
        }
        # 用于获取 kv cache
        self.kv_cache_manager = kv_cache_manager
    
    def optimize_schedule(self,
                          running_requests,
                          waiting_requests,
                          token_budget: int) -> Optional[RLOptimizationResult]:

        start_time = time.perf_counter()
        self.stats['total_optimizations'] += 1

        # 贪心分配资源

        allocation = self._greedy_allocation(running_requests, waiting_requests, token_budget)

        # 验证分配结果
        scheduled_requests = [req_id for req_id, tokens in allocation.items() if tokens > 0]
        actual_tokens = sum(allocation.values())

        result = None
        if actual_tokens <= token_budget:
            decode_count, prefill_count = self._count_request_types(
                            running_requests, waiting_requests, allocation
                        )
            result = RLOptimizationResult(
                            select_S=token_budget,
                            allocation=allocation,
                            optimization_time_ms=0,  # 稍后设置
                            actual_S=actual_tokens,
                            decode_count=decode_count,
                            prefill_count=prefill_count
                        )
        if result:
            result.optimization_time_ms = (time.perf_counter() - start_time) * 1000
            # 判断是否超时
            if result.optimization_time_ms > self.config.optimization_timeout_ms:
                self.stats['timeout_count'] += 1
                logger.warning(f"Optimization timeout: {result.optimization_time_ms:.2f}ms "
                            f"for S={token_budget}")
            
            # self._update_stats(result.optimization_time_ms)
                
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
            if self._is_decode_phase(req):
                decode_requests.append(req)
            else:
                prefill_requests.append(req)
        
        # Phase 2: 优先分配decode请求（每个1 token）
        for req in decode_requests:
            if remaining_budget >= 1 :
                allocation[req.request_id] = 1
                remaining_budget -= 1
            else:
                allocation[req.request_id] = 0
        
        # Phase 3: 分配running prefill请求
        # 按剩余token数排序，优先处理即将完成的请求
        prefill_requests.sort(key=lambda req: self._get_remaining_prefill_tokens(req))
        
        for req in prefill_requests:
            if remaining_budget <= 0 :
                allocation[req.request_id] = 0
                continue
            
            remaining_tokens = self._get_remaining_prefill_tokens(req)
            chunk_size = min(remaining_tokens, remaining_budget)
            
            allocation[req.request_id] = chunk_size
            remaining_budget -= chunk_size
        
        # Phase 4: 选择waiting请求
        if remaining_budget > 0 :
            selected_waiting = self._select_waiting_requests(
                waiting_requests, remaining_budget
            )
            
            for req, tokens in selected_waiting:
                allocation[req.request_id] = tokens
                remaining_budget -= tokens
        
        return allocation
    
    def _select_waiting_requests(self, waiting_requests,remaining_budget: int) :
        
        if not waiting_requests :
            return []
        
        # 按优先级排序（如果有优先级字段）
        # 注意：vLLM Request可能没有priority字段，需要兼容处理
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

            # # 计算启动该请求需要的最小token数
            # logger.info(f"num_tokens:{req.num_tokens},num_prompt_tokens:{req.num_prompt_tokens}")
            # prompt_tokens = getattr(req, 'num_prompt_tokens', 0)
            # if prompt_tokens <= 0:
            #     # 如果无法获取prompt长度，使用默认最小值
            #     logger.info(f"can't get the prompt_tokens of req {req.request_id}, use default value 16")
            #     min_startup_tokens = 16
            # else:
            #     min_startup_tokens = min(16, prompt_tokens)
            
            # if remaining_budget < min_startup_tokens:
            #     # 剩余预算不足以启动新请求
            #     break
            
            # # 计算该请求的token分配
            # max_chunk = min(prompt_tokens, 512) if prompt_tokens > 0 else 256
            # chunk_size = min(max_chunk, remaining_budget)
            
            # if chunk_size >= min_startup_tokens:
            #     selected.append((req, chunk_size))
            #     remaining_budget -= chunk_size
        
        return selected
    
    def _is_decode_phase(self, request) -> bool:
        """判断请求是否处于decode阶段"""
        try:
            return request.num_computed_tokens >= request.num_prompt_tokens
        except AttributeError:
            # 如果字段不存在，假设是prefill阶段
            return False
    
    def _get_remaining_prefill_tokens(self, request) -> int:
        """获取prefill请求的剩余token数"""
        try:
            return max(0, request.num_prompt_tokens - request.num_computed_tokens)
        except AttributeError:
            # 如果字段不存在，返回默认值
            return 256
    
    def _count_request_types(self, 
                           running_requests,
                           waiting_requests,
                           allocation: Dict[str, int]) -> Tuple[int, int]:
        """统计分配结果中的decode和prefill请求数量"""
        decode_count = 0
        prefill_count = 0
        
        for req in running_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                if self._is_decode_phase(req):
                    decode_count += 1
                else:
                    prefill_count += 1
        
        for req in waiting_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                prefill_count += 1  # 新请求都是prefill
        
        return decode_count, prefill_count
    
    def _update_stats(self, optimization_time_ms: float) -> None:
        """更新优化器统计信息"""
        # 计算移动平均
        alpha = 0.1  # 指数移动平均的权重
        if self.stats['avg_optimization_time_ms'] == 0:
            self.stats['avg_optimization_time_ms'] = optimization_time_ms
        else:
            self.stats['avg_optimization_time_ms'] = (
                alpha * optimization_time_ms + 
                (1 - alpha) * self.stats['avg_optimization_time_ms']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        return {
            'total_optimizations': self.stats['total_optimizations'],
            'timeout_count': self.stats['timeout_count'],
            'avg_optimization_time_ms': self.stats['avg_optimization_time_ms'],
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_optimizations': 0,
            'timeout_count': 0,
            'avg_optimization_time_ms': 0.0,
        }
