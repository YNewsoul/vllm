"""
SLO 感知调度器模块。

本模块提供 SLOScheduler 类，用于根据调度器的 running 和 waiting 队列信息，
动态调整 token budget，以优化 TTFT SLO 达标率。
"""

from vllm.logger import init_logger
logger = init_logger(__name__)

try:
    from .fixed_scheduler import FixedScheduler
    from .random_scheduler import RandomScheduler
    from .multislo_scheduler import MultiSloScheduler
    from .sarathi_scheduler import SarathiScheduler
    from .qoserve_scheduler import QoServeScheduler
    from .sliding_scheduler import SlidingScheduler
    from .config import SloSchedulerConfig
except ImportError as e:
    from fixed_scheduler import FixedScheduler
    from random_scheduler import RandomScheduler
    from multislo_scheduler import MultiSloScheduler
    from sarathi_scheduler import SarathiScheduler
    from qoserve_scheduler import QoServeScheduler
    from sliding_scheduler import SlidingScheduler
    from config import SloSchedulerConfig
    logger.warning("Slo scheduler not available aaa : %s", e)

# 调度器映射
scheduler_cls = {
        "random-chunk": RandomScheduler,
        "sarathi": SarathiScheduler,
        "multislo": MultiSloScheduler,
        "fixed-chunk": FixedScheduler,
        "qoserve": QoServeScheduler,
        "sliding-chunk": SlidingScheduler,
    }

class SloScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()

        self.sched_mode = self.config.sched_mode
        try:
            self.scheduler = scheduler_cls.get(self.sched_mode)()
        except TypeError:
            logger.error("Scheduler mode %s init failed", self.sched_mode)
            self.sched_mode = "fixed-chunk"
            self.scheduler = scheduler_cls.get(self.sched_mode)()
            
    def get_status(self) -> dict:
        return {
            "schedule_mode": self.sched_mode
        }
    
    def sched_decision(self, sched_state: dict) -> dict:

        running = sched_state['running']
        waiting = sched_state['waiting']

        # Step 1: 判断是否需要调度
        sched = self._sched_estimate(running, waiting)

        if not sched["slo_sched"]:
            return {
                "decode_only": False,
                "token_budget": sched_state["token_budget"],
                "slo_sched": False,
                "assigned":None}
        
        # Step 2: 更新sched_state,包含更细的划分
        sched_state.update(sched)

        # Step 3:调用调度器执行调度决策
        return self.scheduler.schedule(sched_state)

    def _sched_estimate(self, running: list, waiting: list):
        """判断是否需要进行chunk size调整调度"""
        num_running = len(running)
        num_waiting = len(waiting)

        decoding, prefilling = self._classify_running(running)
        num_decode = len(decoding)
        num_prefill = len(prefilling)
        
        slo_sched = False
        if num_running != 0 and num_waiting == 0:
            # 1、有请求运行，无请求等待
            if num_prefill != 0 and num_decode != 0:
                # 1.1、运行请求包括 prefill 和 decode 请求
                slo_sched = True
            else:
                # 1.2、运行请求只包括 prefill 请求
                slo_sched = False

        elif num_running != 0 and num_waiting != 0:
            # 2、 有请求运行，有请求等待
            # 2.1、运行请求中包含 decode 请求
            slo_sched = num_decode != 0

        return {
                "decoding": decoding,
                "prefilling": prefilling, 
                "slo_sched": slo_sched
        }

    def _classify_running(self, running):
        """对running队列中的请求进行分类，返回prefill和decode请求"""

        decoding = []
        prefilling = []

        for req in running:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                decoding.append(req)
            else:
                prefilling.append(req)
        return decoding,prefilling
