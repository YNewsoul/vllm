import time
from typing import List, Dict, Tuple, Optional, Any
import sys
import os
from dataclasses import dataclass

from vllm.logger import init_logger

from .RLConfig import RLSchedulerConfig

logger = init_logger(__name__)


@dataclass
class RLOptimizationResult:
    """RL调度结果数据类
    
    输出结果，用于调度器进行资源分配决策。
    """
    optimal_batch_size: int                # 最优batch size
    optimal_token_budget: int              # 最优token预算
    allocation: Dict[str, int]             # request_id -> tokens的分配映射
    optimization_time_ms: float            # 调度执行时间
    actual_batch_size: int                 # 实际分配的batch size
    decode_count: int                      # decode请求数量
    prefill_count: int                     # prefill请求数量


class RLOptimizer:
    """RL优化器
    
    实现论文中描述的延迟导向统一调度算法，通过三个阶段的优化
    实现最佳的资源分配和SLA保证。
    """
    
    def __init__(self, ):
        """初始化优化器"""
        self.config = RLSchedulerConfig.from_env()
        
        # 调度的统计信息
        self.stats = {
            'total_optimizations': 0,
            'avg_optimization_time_ms': 0.0,
        }
    
    def optimize_schedule(self,
                          running_requests,
                          waiting_requests,
                          batch_size: int,
                          token_budget: int) -> Optional[RLOptimizationResult]:

        start_time = time.perf_counter()
        self.stats['total_optimizations'] += 1

        # 贪心分配资源

        allocation = self._greedy_allocation(
            running_requests, waiting_requests, 
            batch_size, token_budget
        )

        # 验证分配结果
        scheduled_requests = [req_id for req_id, tokens in allocation.items() if tokens > 0]
        actual_batch_size = len(scheduled_requests)
        actual_tokens = sum(allocation.values())

        if actual_batch_size <= batch_size and actual_tokens <= token_budget:
            decode_count, prefill_count = self._count_request_types(
                            running_requests, waiting_requests, allocation
                        )
            result = RLOptimizationResult(
                            optimal_batch_size=batch_size,
                            optimal_token_budget=token_budget,
                            allocation=allocation,
                            optimization_time_ms=0,  # 稍后设置
                            actual_batch_size=actual_batch_size,
                            decode_count=decode_count,
                            prefill_count=prefill_count
                        )
        if result:
            result.optimization_time_ms = (time.perf_counter() - start_time) * 1000
            # 判断是否超时
            if result.optimization_time_ms > self.config.optimization_timeout_ms:
                self.stats['timeout_count'] += 1
                logger.warning(f"Optimization timeout: {result.optimization_time_ms:.2f}ms "
                            f"for B={batch_size}, S={token_budget}")
            
            self._update_stats(result.optimization_time_ms)
                
            if self.config.verbose_logging:
                logger.debug(f"Optimization success: B={result.actual_batch_size}, "
                            f"S={sum(result.allocation.values())}, ")
                
            return result
        else:
            if self.config.verbose_logging:
                logger.info(f"Optimization failed: no valid allocation found")
            return None
    
    def _greedy_allocation(self, 
                          running_requests,
                          waiting_requests,
                          batch_size: int,
                          token_budget: int) -> Dict[str, int]:
        """贪心分配算法
        
        实现论文中描述的三阶段贪心策略：
        1. Running中的decode请求（每个需要1 token）
        2. Running中的prefill请求
        3. Waiting中的新请求（按优先级排序）
        
        Args:
            running_requests: 运行中的请求
            waiting_requests: 等待中的请求
            batch_size: 目标batch size
            token_budget: token预算
            
        Returns:
            request_id -> tokens的分配字典
        """
        allocation = {}
        remaining_budget = token_budget
        remaining_slots = batch_size
        
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
            if remaining_budget >= 1 and remaining_slots > 0:
                allocation[req.request_id] = 1
                remaining_budget -= 1
                remaining_slots -= 1
            else:
                allocation[req.request_id] = 0
        
        # Phase 3: 分配running prefill请求
        # 按剩余token数排序，优先处理即将完成的请求
        prefill_requests.sort(key=lambda req: self._get_remaining_prefill_tokens(req))
        
        for req in prefill_requests:
            if remaining_budget <= 0 or remaining_slots <= 0:
                allocation[req.request_id] = 0
                continue
            
            remaining_tokens = self._get_remaining_prefill_tokens(req)
            # 限制chunk大小以避免过度占用资源
            max_chunk = min(remaining_tokens, 2048)  # 最大chunk限制
            chunk_size = min(max_chunk, remaining_budget)
            
            allocation[req.request_id] = chunk_size
            remaining_budget -= chunk_size
            remaining_slots -= 1
        
        # Phase 4: 选择waiting请求
        if remaining_budget > 0 and remaining_slots > 0:
            selected_waiting = self._select_waiting_requests(
                waiting_requests, remaining_slots, remaining_budget
            )
            
            for req, tokens in selected_waiting:
                allocation[req.request_id] = tokens
                remaining_budget -= tokens
                remaining_slots -= 1
        
        return allocation
    
    def _select_waiting_requests(self, 
                               waiting_requests,
                               remaining_slots: int,
                               remaining_budget: int) :
        """选择等待队列中的请求
        
        Args:
            waiting_requests: 等待中的请求列表
            remaining_slots: 剩余slot数
            remaining_budget: 剩余token预算
            
        Returns:
            选中的请求及其token分配列表
        """
        if not waiting_requests or remaining_slots <= 0 or remaining_budget <= 0:
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
        
        selected = []
        
        for req in sorted_waiting:
            if remaining_slots <= 0 or remaining_budget <= 0:
                break
            
            # 计算启动该请求需要的最小token数
            prompt_tokens = getattr(req, 'num_prompt_tokens', 0)
            if prompt_tokens <= 0:
                # 如果无法获取prompt长度，使用默认最小值
                min_startup_tokens = 16
            else:
                min_startup_tokens = min(16, prompt_tokens)
            
            if remaining_budget < min_startup_tokens:
                # 剩余预算不足以启动新请求
                break
            
            # 计算该请求的token分配
            max_chunk = min(prompt_tokens, 512) if prompt_tokens > 0 else 256
            chunk_size = min(max_chunk, remaining_budget)
            
            if chunk_size >= min_startup_tokens:
                selected.append((req, chunk_size))
                remaining_budget -= chunk_size
                remaining_slots -= 1
        
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
