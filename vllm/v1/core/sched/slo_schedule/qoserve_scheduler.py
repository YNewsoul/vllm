import os

from vllm.logger import init_logger

try:
    from .config import SloSchedulerConfig
    from .batch_forwarder import BatchForwarder
    from .multislo_predictor import MultiSloPredictor
    from .multislo_online_trainer import (MultiSloOnlineTrainConfig,
                                          MultiSloOnlineTrainer)
    from .utils import convert_req_to_snapshot
except ImportError:
    from config import SloSchedulerConfig
    from batch_forwarder import BatchForwarder
    from multislo_predictor import MultiSloPredictor
    from multislo_online_trainer import (MultiSloOnlineTrainConfig,
                                         MultiSloOnlineTrainer)
    from utils import convert_req_to_snapshot

logger = init_logger(__name__)


class QoServeScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()
        self.predictor_model = self.config.predictor_model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", self.predictor_model)
        self.__alpha = self.config.qoserve_alpha
        predictor = MultiSloPredictor.load(model_path)

        self.online_trainer: MultiSloOnlineTrainer | None = None
        if self.config.online_train_enabled:
            online_cfg = MultiSloOnlineTrainConfig(
                enabled=True,
                buffer_size=self.config.online_buffer_size,
                warmup_samples=self.config.online_warmup_samples,
                retrain_interval=self.config.online_retrain_interval,
                l2=self.config.online_l2,
                use_scene_models=self.config.online_use_scene_models,
                min_scene_samples=self.config.online_min_scene_samples,
                min_ms=self.config.online_min_ms,
                save_path=self.config.online_save_path,
                ingest_queue_size=self.config.online_ingest_queue_size,
            )
            self.online_trainer = MultiSloOnlineTrainer(
                initial_predictor=predictor,
                config=online_cfg,
            )
            predictor_for_forward = self.online_trainer
            logger.info(
                "QoServe online training enabled: warmup=%d interval=%d buffer=%d",
                online_cfg.warmup_samples,
                online_cfg.retrain_interval,
                online_cfg.buffer_size,
            )
        else:
            predictor_for_forward = predictor

        self.batch_forwarder = BatchForwarder(predictor=predictor_for_forward)

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

        # Step 2:从decoding中获取最大可支持的iter_time
        current_time = sched_state['current_time']
        min_iter_time = 1
        for req in decoding:
            # 只对需要safeguard的请求进行判断
            if not req.safeguard:
                continue
            decoded_tokens = max(0, req.num_computed_tokens - req.num_prompt_tokens)
            next_token_slack = req.arrival_time + req.ttft_slo + decoded_tokens*req.tbt - current_time
            min_iter_time = min(min_iter_time, next_token_slack)

        # Step3:采用qoserve优先级排序
        prefilling_snapshots, waiting_snapshots = (
            self._qoserve_priority(
                prefilling_snapshots, waiting_snapshots))
        
        # Step 4:调用 time_to_token_budget 计算token budget
        token_budget, assigned = self.batch_forwarder.time_to_token_budget(
            decoding=decoding_snapshots,
            prefilling=prefilling_snapshots,
            waiting=waiting_snapshots,
            target_iter_ms=min_iter_time*1000,
        )

        return {
            "decode_only": False,
            "token_budget": token_budget,
            "slo_sched": True,
            "sched_method":"qoserve",
            "assigned": assigned,
            "max_iter_time": min_iter_time,
        }

    # qoserve 优先级排序
    def _qoserve_priority(
        self,
        prefilling_snapshots: list,
        waiting_snapshots: list,
    ) -> tuple[list, list]:
        all_reqs = prefilling_snapshots + waiting_snapshots
        if len(all_reqs) < 2:
            return [], all_reqs

        def priority_key(req):
            remaining = req.num_prompt_tokens - req.num_computed_tokens
            priority = req.arrival_time + req.ttft_slo + self.__alpha * remaining
            is_degraded = 1 if bool(req.safeguard) else 0
            return (is_degraded, priority, req.arrival_time)

        ordered_waiting = sorted(all_reqs, key=priority_key)
        return [], ordered_waiting

    def should_capture_runtime_record(self) -> bool:
        if self.online_trainer is None:
            return False
        return self.online_trainer.should_capture_runtime_record()

    def observe_record(self, record: dict) -> None:
        if self.online_trainer is None:
            return
        self.online_trainer.observe_record(record)
