import os

from typing import List

try:
    from .config import SloSchedulerConfig
    from .batch_forwarder import BatchForwarder
    from .multislo_predictor import MultiSloPredictor
    from .utils import convert_req_to_snapshot, ReqSnapshot
except ImportError:
    from config import SloSchedulerConfig
    from batch_forwarder import BatchForwarder
    from multislo_predictor import MultiSloPredictor
    from utils import convert_req_to_snapshot, ReqSnapshot


class SlidingScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()

        self.predictor_model = self.config.predictor_model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", self.predictor_model)
        self.__alpha = self.config.multislo_urgency_threshold
        self.batch_forwarder = BatchForwarder(predictor=MultiSloPredictor.load(model_path))
    
    def schedule(self, sched_state: dict) -> dict:
        """
        调用调度器执行调度决策
        """
        decoding = sched_state['decoding']
        prefilling = sched_state['prefilling']
        waiting = sched_state['waiting']
        token_budget = sched_state['token_budget']
        
        # Step 1: 将 running 和 waiting 请求转换为 ReqSnapshot
        decoding_snapshots = [
            convert_req_to_snapshot(req)
            for req in decoding
        ]
        prefilling_snapshots = [
            convert_req_to_snapshot(req)
            for req in prefilling
        ]
        waiting_snapshots = [
            convert_req_to_snapshot(req)
            for req in waiting
        ]

        current_time = sched_state['current_time']

        # Step 2:优先级排序
        if len(prefilling_snapshots) + len(waiting_snapshots) >=2:
            prefilling_snapshots, waiting_snapshots = (
                self._reorder_prefill_waiting_by_priority(
                    prefilling_snapshots=prefilling_snapshots,
                    waiting_snapshots=waiting_snapshots,
                    now=current_time,
                    throughput=sched_state['throughput']))

        # Step 3: 先进行一次最大tokenbudget的PD融合计算
        iter_ms, assigned = self.batch_forwarder.forward(decoding_snapshots,prefilling_snapshots,
                                                         waiting_snapshots,token_budget)

        # Step 4:进行sliding window调度
        # 4.1 记录当前迭代允许的最大迭代时间和下一个迭代允许的最大时间
        max_iter_time = next_max_iter_time = 10
        for req in decoding:
            # 只对需要safeguard的请求进行判断
            if not req.safeguard:
                continue
            decoded_tokens = max(0, req.num_computed_tokens - req.num_prompt_tokens)
            token_slack = req.arrival_time + req.ttft_slo + decoded_tokens*req.tbt - current_time
            max_iter_time = min(max_iter_time, token_slack)
            # 当前请求的 token_slack 可能不是当前最大迭代时间，但其tbt也可能影响下一个最大迭代时间
            next_max_iter_time = min(next_max_iter_time, token_slack - max_iter_time + req.tbt)
        
        # 能进行最大tokenbudget融合计算
        if max_iter_time*1000 > iter_ms:
            return {
                "decode_only": False,
                "token_budget": token_budget,
                "slo_sched": True,
                "assigned": assigned,
            }
        return self._compare_iter(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            token_budget=token_budget,
            max_iter_time=max_iter_time,
            next_max_iter_time=next_max_iter_time,
        )

    def _reorder_prefill_waiting_by_priority(
        self,
        prefilling_snapshots: list,
        waiting_snapshots: list,
        now: float,
        throughput: float,
    ) -> tuple[list, list]:
        all_reqs = prefilling_snapshots + waiting_snapshots

        def priority_key(req):
            remaining_tokens = req.num_prompt_tokens - req.num_computed_tokens
            prefill_time = remaining_tokens / throughput
            slack = req.ttft_slo + req.arrival_time - now
            ratio = prefill_time / slack

            safeguard_rank = 0 if bool(req.safeguard) else 1
            urgent_rank = 0 if ratio > self.__alpha else 1
            return (safeguard_rank, urgent_rank, remaining_tokens)

        ordered_waiting = sorted(all_reqs, key=priority_key)
        return [], ordered_waiting
    
    def _compare_iter(
        self, 
        decoding: List[ReqSnapshot],
        prefilling: List[ReqSnapshot],
        waiting: List[ReqSnapshot],
        token_budget: int = 2048,
        max_iter_time: float = 1, 
        next_max_iter_time: float = 1
    ) -> dict:
        cur_max_ms = max_iter_time * 1000.0
        nxt_max_ms = next_max_iter_time * 1000.0

        cur_max_budget, _ = self.batch_forwarder.time_to_token_budget(
            decoding=decoding,
            prefilling=prefilling,
            waiting=waiting,
            target_iter_ms=cur_max_ms,
        )
        nxt_max_budget, _ = self.batch_forwarder.time_to_token_budget(
            decoding=decoding,
            prefilling=prefilling,
            waiting=waiting,
            target_iter_ms=nxt_max_ms,
        )

        total_budget = cur_max_budget + nxt_max_budget

        left = len(decoding)
        right = token_budget

        def _internal_eval_budget(budget: int) -> tuple[float, dict[str, int] | None]:
            iter_ms, assigned = self.batch_forwarder.forward(
                decoding=decoding,
                prefilling=prefilling,
                waiting=waiting,
                token_budget=budget,
            )
            return iter_ms, assigned

        def _internal_total_ms(cur_budget: int) -> tuple[float, dict[str, int] | None]:
            nxt_budget = total_budget - cur_budget
            cur_ms, cur_assigned = _internal_eval_budget(cur_budget)
            nxt_ms, _ = _internal_eval_budget(nxt_budget)
            return cur_ms + nxt_ms, cur_assigned

        # 离散三分：在预算空间上搜索最小总耗时
        # 同时保留边界检查，覆盖单调情况下“最优在边界”的情形。
        origin_left = left
        origin_right = right
        while right - left > 30:
            mid_left = left + (right - left) // 3
            mid_right = right - (right - left) // 3
            total_left, _ = _internal_total_ms(mid_left)
            total_right, _ = _internal_total_ms(mid_right)
            if total_left <= total_right:
                right = mid_right - 1
            else:
                left = mid_left + 1


        # 边界条件对比：覆盖单调函数时最优在边界的情况
        best_budget = origin_left
        best_total_ms, best_assigned = _internal_total_ms(origin_left)

        right_total_ms, right_assigned = _internal_total_ms(origin_right)
        if (
            right_total_ms < best_total_ms
            or (right_total_ms == best_total_ms and origin_right > best_budget)
        ):
            best_total_ms = right_total_ms
            best_budget = origin_right
            best_assigned = right_assigned

        # 取三分搜索的中间值
        ternary_budget = (left + right) // 2
        ternary_total_ms, ternary_assigned = _internal_total_ms(ternary_budget)
        if (
            ternary_total_ms < best_total_ms
            or (ternary_total_ms == best_total_ms and ternary_budget > best_budget)
        ):
            best_total_ms = ternary_total_ms
            best_budget = ternary_budget
            best_assigned = ternary_assigned

        return {
            "decode_only": False,
            "token_budget": best_budget,
            "slo_sched": True,
            "assigned": best_assigned,
        }
