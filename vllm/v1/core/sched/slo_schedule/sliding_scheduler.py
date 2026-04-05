import os

from typing import List

try:
    from .config import SloSchedulerConfig
    from .batch_forwarder import BatchForwarder
    from .multislo_predictor import MultiSloPredictor
    from .sliding_utils import select_dp_batch_decision
    from .utils import convert_req_to_snapshot, ReqSnapshot
except ImportError:
    from config import SloSchedulerConfig
    from batch_forwarder import BatchForwarder
    from multislo_predictor import MultiSloPredictor
    from sliding_utils import select_dp_batch_decision
    from utils import convert_req_to_snapshot, ReqSnapshot


class SlidingScheduler:
    def __init__(self):
        # 从环境变量加载调度器参数（模型名、阈值等）。
        self.config = SloSchedulerConfig.from_env()

        self.predictor_model = self.config.predictor_model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", self.predictor_model)
        # urgency 阈值用于 prefill/waiting 重排时的紧急度判断。
        self.__alpha = self.config.multislo_urgency_threshold
        # 统一通过 BatchForwarder + predictor 评估不同预算下的迭代时长。
        self.batch_forwarder = BatchForwarder(predictor=MultiSloPredictor.load(model_path))
    
    def schedule(self, sched_state: dict) -> dict:
        """
        Sliding 调度主流程：
        1) 组装请求快照并重排 prefill/waiting；
        2) 先尝试“直接用当前 token_budget”；
        3) 若超出时延约束，计算 max/next 窗口预算；
        4) 先走 DP 修正（处理 TTFT 风险），否则走两窗口总时长最优搜索
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

        # Step 2: prefill/waiting 优先级排序（将两队列合并后统一排序）。
        if len(prefilling_snapshots) + len(waiting_snapshots) >=2:
            prefilling_snapshots, waiting_snapshots = (
                self._reorder_prefill_waiting_by_priority(
                    prefilling_snapshots=prefilling_snapshots,
                    waiting_snapshots=waiting_snapshots,
                    now=current_time,
                    throughput=sched_state['throughput']))

        # Step 3: 先在给定 token_budget 下评估一次，判断是否已满足窗口约束。
        iter_ms, assigned = self.batch_forwarder.forward(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            token_budget=token_budget)

        # Step 4: sliding window 调度
        # 4.1 计算当前窗口与下一窗口允许的最大迭代时长（秒）。
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

        # 4.2 若“当前预算下的实际时长”仍在约束内，直接返回当前分配。
        if max_iter_time*1000 > iter_ms:
            return {
                "decode_only": False,
                "token_budget": token_budget,
                "slo_sched": True,
                "assigned": assigned,
            }
        
        # 4.3 将窗口时长约束映射为预算（前置计算，避免后续重复调用）。
        cur_max_ms = max_iter_time * 1000.0
        nxt_max_ms = next_max_iter_time * 1000.0
        cur_max_budget, _ = self.batch_forwarder.time_to_token_budget(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            target_iter_ms=cur_max_ms,
        )
        nxt_max_budget, _ = self.batch_forwarder.time_to_token_budget(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            target_iter_ms=nxt_max_ms,
        )

        # 4.4 对“当前窗口预算”做 TTFT 风险检查：
        # 若存在风险，走 DP 方案返回显式 assigned。
        dp_decision = select_dp_batch_decision(
            batch_forwarder=self.batch_forwarder,
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            token_budget=token_budget,
            cur_max_budget=cur_max_budget,
            current_time=current_time,
            max_iter_time=max_iter_time,
        )
        if dp_decision is not None:
            return {
                "decode_only": False,
                "token_budget": dp_decision["token_budget"],
                "slo_sched": True,
                "assigned": dp_decision["assigned"],
            }
        # 4.5 无显式风险时，执行两窗口总时长最优预算搜索。
        return self.select_best_batch_decision(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            token_budget=token_budget,
            cur_max_budget=cur_max_budget,
            nxt_max_budget=nxt_max_budget,
        )

    def _reorder_prefill_waiting_by_priority(
        self,
        prefilling_snapshots: list,
        waiting_snapshots: list,
        now: float,
        throughput: float,
    ) -> tuple[list, list]:
        # 将 running 中 prefill 与 waiting 合并后统一排序，
        # 返回格式保持 (prefilling, waiting)；这里全部放回 waiting。
        all_reqs = prefilling_snapshots + waiting_snapshots

        def priority_key(req):
            remaining_tokens = req.num_prompt_tokens - req.num_computed_tokens
            prefill_time = remaining_tokens / throughput
            slack = req.ttft_slo + req.arrival_time - now
            ratio = prefill_time / slack

            safeguard_rank = 0 if bool(req.safeguard) else 1
            urgent_rank = 0 if ratio > self.__alpha else 1
            # 优先级：safeguard > urgent > remaining_tokens 少者优先。
            return (safeguard_rank, urgent_rank, remaining_tokens)

        ordered_waiting = sorted(all_reqs, key=priority_key)
        return [], ordered_waiting
    
    def select_best_batch_decision(
        self, 
        decoding: List[ReqSnapshot],
        prefilling: List[ReqSnapshot],
        waiting: List[ReqSnapshot],
        token_budget: int = 2048,
        cur_max_budget: int = 1,
        nxt_max_budget: int = 1,
    ) -> dict:
        """
        在 [len(decoding), token_budget] 的预算区间内，
        近似寻找“当前窗口 + 下一窗口”总时长最小的预算。
        """
        # 两个窗口预算之和固定，搜索的是当前窗口分到多少预算。
        total_budget = cur_max_budget + nxt_max_budget

        # 当前窗口预算下界至少覆盖 decoding（每个 decode 1 token）。
        left = len(decoding)
        right = token_budget

        def _internal_eval_budget(budget: int) -> tuple[float, dict[str, int] | None]:
            # 评估单个预算下的一次迭代时长与分配结果。
            iter_ms, assigned = self.batch_forwarder.forward(
                decoding=decoding,
                prefilling=prefilling,
                waiting=waiting,
                token_budget=budget,
            )
            return iter_ms, assigned

        def _internal_total_ms(cur_budget: int) -> tuple[float, dict[str, int] | None]:
            # 由总预算推导下一窗口预算，目标最小化两窗口总时长。
            nxt_budget = total_budget - cur_budget
            cur_ms, cur_assigned = _internal_eval_budget(cur_budget)
            nxt_ms, _ = _internal_eval_budget(nxt_budget)
            return cur_ms + nxt_ms, cur_assigned

        # 离散三分：在预算空间上快速逼近最小总耗时点。
        # 后续仍会做边界检查，避免“最优在边界”被漏掉。
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


        # 边界条件对比：覆盖单调或近单调情况下“最优在边界”的情形。
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

        # 取三分收敛后区间中点作为候选，再和边界做比较。
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
