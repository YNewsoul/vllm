try:
    from .config import SloSchedulerConfig
except ImportError:
    from config import SloSchedulerConfig

class SarathiScheduler:
    def __init__(self):
        self.config = SloSchedulerConfig.from_env()

__all__ = [
    "SarathiScheduler",
]