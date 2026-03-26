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


class MultiSloScheduler:
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
        iter_ms, assigned = self.batch_forwarder.forward(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            token_budget=token_budget,
        )

        # Step 4:判断是否需要 decode only
        # 记录当前最小迭代时间和下一个最小迭代时间
        min_iter_time = next_min_iter_time = 1
        for req in decoding:
            # 只对需要safeguard的请求进行判断
            if not req.safeguard:
                continue
            decoded_tokens = max(0, req.num_computed_tokens - req.num_prompt_tokens)
            token_slack = req.arrival_time + req.ttft_slo + decoded_tokens*req.tbt - current_time
            min_iter_time = min(min_iter_time, token_slack)
            next_min_iter_time = min(next_min_iter_time, token_slack - min_iter_time + req.tbt)
            
        if min_iter_time*1000 > iter_ms:
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
            min_iter_time=min_iter_time,
            next_min_iter_time=next_min_iter_time,
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
        min_iter_time: float = 1, 
        next_min_iter_time: float = 1
    ) -> dict:
        # 当前的token budget
        cur_token_budget, _ = self.batch_forwarder.time_to_token_budget(
            decoding=decoding,
            prefilling=prefilling,
            waiting=waiting,
            target_iter_ms=min_iter_time*1000,
        )
        # 下一个迭代时间的token budget
        next_token_budget, _ = self.batch_forwarder.time_to_token_budget(
            decoding=decoding,
            prefilling=prefilling,
            waiting=waiting,
            target_iter_ms=next_min_iter_time*1000,
        )
        # 纯解码的迭代时间
        num_decoding = len(decoding)
        decode_only_iter_ms, _ = self.batch_forwarder.forward(
            decoding=decoding,
            prefilling=prefilling,
            waiting=waiting,
            token_budget=num_decoding,
        )
        # 下一个大的迭代时间
        next_iter_ms, _ = self.batch_forwarder.forward(
            decoding=decoding,
            prefilling=prefilling,
            waiting=waiting,
            token_budget=cur_token_budget + next_token_budget - num_decoding,
            return_assigned=False,
        )
        if decode_only_iter_ms + next_iter_ms < (min_iter_time + next_min_iter_time)*1000:
            return {
            "decode_only": True,
            "token_budget": token_budget,
            "slo_sched": True,
            "assigned": None,
        }
        return {
            "decode_only": False,
            "token_budget": cur_token_budget,
            "slo_sched": True,
            "assigned": None,
        }