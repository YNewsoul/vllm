import os

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

class SarathiScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()
        self.predictor_model = self.config.predictor_model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", self.predictor_model)
        self.sarathi_mode = self.config.sarathi_mode
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
        # 改成list
        waiting_snapshots = [
            convert_req_to_snapshot(req)
            for req in waiting
        ]
        
        # Step 2: 从decoing请求中获取最严格tbt
        min_iter_time = 1
        for req in decoding_snapshots:
            min_iter_time = min(min_iter_time, req.tbt)

        # Step 3:实行EDF或者SRPF调度策略
        if self.sarathi_mode == "edf":
            prefilling_snapshots, waiting_snapshots = (
                self._sarathi_edf(prefilling_snapshots, waiting_snapshots))
        elif self.sarathi_mode == "srpf":
            prefilling_snapshots, waiting_snapshots = (
                self._sarathi_srpf(prefilling_snapshots, waiting_snapshots))

        # Step 4:调用 time_to_token_budget 计算token budget
        token_budget, assigned_tokens = self.batch_forwarder.time_to_token_budget(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            target_iter_ms=min_iter_time*1000,
        )

        return {
            "decode_only": False,
            "token_budget": token_budget,
            "slo_sched": True,
            "assigned": assigned_tokens,
        }


    def _sarathi_edf(self, prefilling_snapshots: list,
                            waiting_snapshots: list) -> tuple[list, list]:
        all_reqs = prefilling_snapshots + waiting_snapshots
        if len(all_reqs) < 2:
            return [], all_reqs

        def edf_key(req):
            deadline = req.arrival_time + (req.ttft_slo or float("inf"))
            remaining_prefill = max(0,
                                    req.num_prompt_tokens -
                                    req.num_computed_tokens)
            return (deadline, remaining_prefill, req.arrival_time)

        ordered = sorted(all_reqs, key=edf_key)
        return [], ordered


    def _sarathi_srpf(self, prefilling_snapshots: list,
                            waiting_snapshots: list) -> tuple[list, list]:
        all_reqs = prefilling_snapshots + waiting_snapshots
        if len(all_reqs) < 2:
            return [], all_reqs

        def srpf_key(req):
            remaining_prefill = max(0,
                                    req.num_prompt_tokens -
                                    req.num_computed_tokens)
            deadline = req.arrival_time + (req.ttft_slo or float("inf"))
            return (remaining_prefill, deadline, req.arrival_time)

        ordered = sorted(all_reqs, key=srpf_key)
        return [], ordered