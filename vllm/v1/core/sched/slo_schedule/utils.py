
from typing import Any

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