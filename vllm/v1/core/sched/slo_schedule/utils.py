import time
from typing import Any
import logging
logger = logging.getLogger(__name__)

from collections import namedtuple


# ReqSnapshot - 请求快照（不可变的命名元组）
ReqSnapshot = namedtuple(
    "ReqSnapshot",
    [
        "request_id",  # 请求唯一标识符
        "num_computed_tokens",  # 已计算的 token 数量（包括 prefill 和 decode）
        "num_cached_tokens",  # 已缓存的 kv cache token 数量（系统计算）
        "num_prompt_tokens",  # prompt 的总 token 数量（prefill 目标）
        "arrival_time",  # 请求到达时间（本仿真器中不使用，保留字段）
        "ttft_slo",  # TTFT (Time To First Token) SLO 约束
        "max_tokens",  # 允许生成的最大 token 数量（不含 prompt）
        "tbt",
        "safeguard",
    ],
)

def convert_req_to_snapshot(req: Any) -> ReqSnapshot:
    """
    将 vllm request 对象转换为仿真器需要的 ReqSnapshot
    """
    return ReqSnapshot(
        request_id=req.request_id,
        num_computed_tokens=req.num_computed_tokens,
        num_cached_tokens=getattr(req, 'num_cached_tokens', 0),
        num_prompt_tokens=req.num_prompt_tokens,
        arrival_time=req.arrival_time,
        ttft_slo=getattr(req, 'ttft_slo', None),
        max_tokens=req.max_tokens,
        tbt=getattr(req, 'tbt', None),
        safeguard=getattr(req, 'safeguard', None),
    )

class SloLogger:
    def __init__(self):
        self.num_tokens = 0
        self.last_log_time = None
        self.throughput = 2000.0 # 初始值设置为2000
    
    def _reset(self):
        self.num_tokens = 0

    def add_tokens(self, num_tokens: int):
        if self.last_log_time is None:
            self.last_log_time = time.monotonic()
        self.num_tokens += num_tokens
        now = time.monotonic()
        elapsed = now - self.last_log_time
        if elapsed >= 1.0:
            self.throughput = self.num_tokens / elapsed
            logger.info("Throughput: %.2f tokens/s", self.throughput)
            self.last_log_time = now
            self._reset()
            
    def get_throughput(self) -> float:
        return self.throughput 
