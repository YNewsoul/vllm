from collections import deque
from typing import List

try:
    from .utils import ReqSnapshot
    from .multislo_predictor import MulsloPredictor
except ImportError:
    from utils import ReqSnapshot
    from multislo_predictor import MulsloPredictor

class BatchForwarder:
    """
    基于预测模型的批量请求前向器
    """
    def __init__(
        self,
        predictor: MulsloPredictor = None,
    ):
        self.predictor = predictor

    def run(
        self,
        decoding_reqs: List[ReqSnapshot],
        prefilling_reqs: List[ReqSnapshot],
        waiting_reqs: deque[ReqSnapshot],
        token_budget: int = 2048,
    ) -> float:
        """
        运行一次iteration，返回 iteration 执行时长（毫秒）
        """
            
        # 1.初始化分配列表
        chunk_sizes: List[int] = []
        computed_tokens: List[int] = []
        cached_tokens: List[int] = []

        # 2.分配tokens
        # 2.1 第一优先级：分配给 running 队列的 decode 请求

        for req in decoding_reqs:
            chunk_sizes.append(1)
            computed_tokens.append(req.num_computed_tokens)
            cached_tokens.append(req.num_cached_tokens)
            token_budget -= 1
       
        # 2.2 第二优先级：分配给 running 队列的 prefill 请求（FCFS）
        for req in prefilling_reqs:
            if token_budget <= 0:
                break
            chunk = min(
                req.num_prompt_tokens - req.num_computed_tokens, token_budget
            )
            chunk_sizes.append(chunk)
            computed_tokens.append(req.num_computed_tokens)
            cached_tokens.append(req.num_cached_tokens)
            token_budget -= chunk

        # 2.3 第三优先级：分配给 waiting 队列的请求（FCFS）
        while waiting_reqs and token_budget > 0:
            req = waiting_reqs.popleft()

            chunk = min(
                req.num_prompt_tokens - req.num_computed_tokens,
                token_budget,
            )

            chunk_sizes.append(chunk)
            computed_tokens.append(req.num_computed_tokens)
            cached_tokens.append(req.num_cached_tokens)
            token_budget -= chunk
        
        sched_tokens = sum(chunk_sizes)

        # 调用预测器进行时长预测
        iter_ms = float(
            self.predictor.predict(
                chunk_sizes=chunk_sizes,
                cached_tokens=cached_tokens,
                computed_tokens=computed_tokens,
                sched_tokens=sched_tokens,
            )
        )
        return iter_ms

__all__ = [
    "BatchForwarder"
]
