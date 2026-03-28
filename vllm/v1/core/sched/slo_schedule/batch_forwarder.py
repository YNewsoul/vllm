from typing import List

try:
    from .utils import ReqSnapshot
except ImportError:
    from utils import ReqSnapshot

class BatchForwarder:
    """
    基于预测模型的批量请求前向器
    """
    def __init__(
        self,
        predictor,
    ):
        self.predictor = predictor

    def forward(
        self,
        decoding: List[ReqSnapshot],
        prefilling: List[ReqSnapshot],
        waiting: List[ReqSnapshot],
        token_budget: int = 2048,
    ) -> tuple[float, dict[str, int] | None]:
        """
        运行一次 iteration，返回执行时长（毫秒）和分配结果。
        waiting 只读遍历，不会被修改。
        """

        # 1.初始化分配列表
        chunk_sizes: List[int] = []
        computed_tokens: List[int] = []
        cached_tokens: List[int] = []
        assigned: dict[str, int] = {}
        sched_tokens = 0

        # 2.分配tokens
        # 2.1 第一优先级：分配给 running 队列的 decode 请求

        for req in decoding:
            chunk_sizes.append(1)
            sched_tokens += 1
            computed_tokens.append(req.num_computed_tokens)
            cached_tokens.append(req.num_cached_tokens)
            assigned[req.request_id] = 1
            token_budget -= 1
       
        # 2.2 第二优先级：分配给 running 队列的 prefill 请求（FCFS）
        for req in prefilling:
            if token_budget <= 0:
                break
            chunk = min(
                req.num_prompt_tokens - req.num_computed_tokens, token_budget
            )
            chunk_sizes.append(chunk)
            sched_tokens += chunk
            computed_tokens.append(req.num_computed_tokens + chunk)
            cached_tokens.append(req.num_cached_tokens)
            assigned[req.request_id] = chunk
            token_budget -= chunk

        # 2.3 第三优先级：分配给 waiting 队列的请求（FCFS）
        for req in waiting:
            if token_budget <= 0:
                break

            chunk = min(
                req.num_prompt_tokens - req.num_computed_tokens,
                token_budget,
            )

            chunk_sizes.append(chunk)
            sched_tokens += chunk
            computed_tokens.append(req.num_computed_tokens)
            cached_tokens.append(req.num_cached_tokens)
            assigned[req.request_id] = chunk
            token_budget -= chunk

        # 调用预测器进行时长预测
        iter_ms = float(
            self.predictor.predict(
                chunk_sizes=chunk_sizes,
                cached_tokens=cached_tokens,
                computed_tokens=computed_tokens,
                sched_tokens=sched_tokens,
            )
        )
        return iter_ms, assigned

    def time_to_token_budget(
        self,
        decoding: List[ReqSnapshot],
        prefilling: List[ReqSnapshot],
        waiting: List[ReqSnapshot],
        target_iter_ms: float = 50,
        max_iters: int = 10,
        tolerance_pct: float = 0.05,
        lowest_budget: int = 60,
    ) -> tuple[int, dict[str, int] | None]:
        """
        给定目标 iteration 时长，二分搜索近似对应的 token budget
        """
        low = lowest_budget
        high = self._get_high(target_iter_ms)

        best_budget = low
        best_diff = float("inf")
        best_assigned = None

        while low <= high and max_iters > 0:
            mid = (low + high) // 2
            pred_ms, assigned = self.forward(
                decoding=decoding,
                prefilling=prefilling,
                waiting=waiting,
                token_budget=mid,
            )
            diff = abs(pred_ms - target_iter_ms)
            if diff < best_diff:
                best_budget = mid
                best_diff = diff
                best_assigned = assigned

            relative_error = diff / target_iter_ms

            if relative_error <= tolerance_pct:
                return mid, assigned

            if pred_ms < target_iter_ms:
                low = mid + 1
            else:
                high = mid - 1
            max_iters -= 1
        return best_budget, best_assigned
    
    def _get_high(self, target_iter_ms: float) -> int:
        """
        获取最高的 token budget，用于二分搜索
        """
        if target_iter_ms <= 50:
            return 300
        if target_iter_ms <= 100:
            return 600
        if target_iter_ms <= 200:
            return 1300
        return 2048
