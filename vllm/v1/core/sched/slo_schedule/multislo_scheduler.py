import os

from collections import deque

try:
    from .config import SloSchedulerConfig
    from .batch_forwarder import BatchForwarder
    from .multislo_predictor import MultiSloPredictor
    from .utils import convert_req_to_snapshot
except ImportError:
    from config import SloSchedulerConfig
    from batch_forwarder import BatchForwarder
    from multislo_predictor import MultiSloPredictor
    from utils import convert_req_to_snapshot


class MultiSloScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()

        self.multislo_model = self.config.multislo_model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", self.multislo_model)
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
        waiting_snapshots = deque([
            convert_req_to_snapshot(req)
            for req in waiting
        ])

        current_time = sched_state['current_time']

        # Step 2: 先进行一次最大tokenbudget的PD融合计算
        iter_ms = self.batch_forwarder.run(
            decoding_snapshots,
            prefilling_snapshots,
            waiting_snapshots,
            token_budget=token_budget,
        )

        # Step 3:判断是否需要 decode only
        min_iter_time = 1
        for req in decoding:
            # 只对需要safeguard的请求进行判断
            if not req.safeguard:
                continue
            decoded_tokens = max(0, req.num_computed_tokens - req.num_prompt_tokens)
            next_token_slack = req.arrival_time + req.ttft_slo + decoded_tokens*req.tbt - current_time
            min_iter_time = min(min_iter_time, next_token_slack)

        if min_iter_time < iter_ms:
            return {
            "decode_only": True,
            "token_budget": token_budget,
            "slo_sched": True,
            }
        return {
            "decode_only": False,
            "token_budget": token_budget,
            "slo_sched": True,
        }


__all__ = [
    "MultiSloScheduler",
]