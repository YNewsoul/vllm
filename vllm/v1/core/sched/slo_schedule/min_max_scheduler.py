import queue

try:
    from .config import SloSchedulerConfig
except ImportError:
    from config import SloSchedulerConfig


class MinMaxScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()
        self.chunk_queue = queue.Queue()
        self.init_chunk_size = 64

    def _internal_refill_chunk_queue(self, sched_state: dict) -> None:
        decoding = sched_state.get("decodeing")
        if decoding is None:
            decoding = sched_state.get("decoding", [])
        decoding_len = len(decoding)

        chunk_a = self.init_chunk_size
        chunk_b = self.init_chunk_size * 2 - decoding_len

        self.chunk_queue.put(chunk_a)
        self.chunk_queue.put(chunk_a)
        self.chunk_queue.put(decoding_len)
        self.chunk_queue.put(chunk_b)

        self.init_chunk_size += 30
        if self.init_chunk_size > 1024:
            self.init_chunk_size = 64

    def schedule(self, sched_state: dict) -> dict:
        if self.chunk_queue.empty():
            self._internal_refill_chunk_queue(sched_state)

        token_budget = self.chunk_queue.get()

        return {
            "decode_only": False,
            "token_budget": token_budget,
            "slo_sched": True,
            "assigned": None,
        }
