try:
    from .config import SloSchedulerConfig
except ImportError:
    from config import SloSchedulerConfig


class FixedScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()
        self.fixed_chunk_size = self.config.fixed_chunk_size
    
    def schedule(self, sched_state: dict) -> dict:
        if self.fixed_chunk_size == 0:
            token_budget = sched_state['token_budget']
        else:
            token_budget = min(self.fixed_chunk_size, sched_state['token_budget'])

        return {
            "decode_only": False,
            "token_budget": token_budget,
            "slo_sched": True,
            "sched_method": "fixed-chunk",
            "assigned": None,
        }