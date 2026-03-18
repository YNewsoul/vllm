import random

try:
    from .config import SloSchedulerConfig
except ImportError:
    from config import SloSchedulerConfig

class RandomScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()
        self.chunk_sizes = (
            32, 64, 128, 256, 512, 1024, 2048
        )
        self.min_chunk = self.config.min_chunk
        self.max_chunk = self.config.max_chunk

        self._choice = random.choice
        self._randint = random.randint
        self._decode_threshold = 32

    def schedule(self, sched_state: dict) -> dict:
        return self._random_from_list()
    
    def _random_from_list(self) -> dict:
        token_budget = self._choice(self.chunk_sizes)
        return {
            "decode_only": token_budget == self._decode_threshold,
            "token_budget": token_budget,
            "slo_sched": True,
            "assigned": None,
        }

    def _random_in_range(self) -> dict:
        token_budget = self._randint(self.min_chunk, self.max_chunk)
        return {
            "decode_only": token_budget <= self._decode_threshold,
            "token_budget": token_budget,
            "slo_sched": True,
            "assigned": None,
        }