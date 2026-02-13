"""
LLM chunk prefill/decoding 调度仿真
"""

from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

try:
    from chunk_predictor import DurationPredictor
except ImportError:
    from .chunk_predictor import DurationPredictor

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
        "ttft",  # 实际的 TTFT 值（首 token 生成时间）
        "finish_time",  # 请求完成时间（保留字段）
        "tpot_type",
        "tpot_slo",
        "accept",
        "ttft_time",
    ],
)


# ReqState - 请求运行时状态（可变的数据类）
@dataclass(slots=True)
class ReqState:
    """
    请求运行时状态
    """

    request_id: str  # 请求唯一标识符
    num_prompt_tokens: int  # prompt 总 token 数
    max_tokens: int  # 允许生成的最大 token 数
    arrival_time: float  # 到达时间（保留字段）
    ttft_slo: float  # TTFT SLO 约束
    num_computed_tokens: int = 0  # 已计算的 token 数
    num_cached_tokens: int = 0  # 已缓存的 token 数
    ttft: Optional[float] = None  # 实际 TTFT 值
    finish_time: Optional[float] = None  # 请求完成时间（保留字段）
    tpot_type: Optional[str] = None  # TPOT 类型
    tpot_slo: Optional[float] = None  # TPOT SLO 约束
    accept: Optional[bool] = None  # 是否被接受
    ttft_time: Optional[float] = None  # 首token输出时间

    @classmethod
    def from_snapshot(cls, snap: ReqSnapshot) -> "ReqState":
        """
        从快照创建运行时状态
        """
        return cls(
            request_id=snap.request_id,
            num_prompt_tokens=snap.num_prompt_tokens,
            max_tokens=snap.max_tokens,
            arrival_time=snap.arrival_time,
            ttft_slo=snap.ttft_slo,
            num_computed_tokens=snap.num_computed_tokens,
            num_cached_tokens=snap.num_cached_tokens,
            ttft=snap.ttft,
            finish_time=snap.finish_time,
            tpot_type=snap.tpot_type,
            tpot_slo=snap.tpot_slo,
            accept=snap.accept,
            ttft_time=snap.ttft_time,
        )

    def to_snapshot(self) -> ReqSnapshot:
        """
        将运行时状态转换为快照
        """
        return ReqSnapshot(
            self.request_id,
            self.num_computed_tokens,
            self.num_cached_tokens,
            self.num_prompt_tokens,
            self.arrival_time,
            self.ttft_slo,
            self.max_tokens,
            self.ttft,
            self.finish_time,
            self.tpot_type,
            self.tpot_slo,
            self.accept,
            self.ttft_time,
        )

    def phase(self) -> int:
        if self.is_finished():
            return 3
        return (
            0 if self.num_computed_tokens < self.num_prompt_tokens else 1
        )

    def is_finished(self) -> bool:
        target_total = self.num_prompt_tokens + self.max_tokens
        return self.num_computed_tokens >= target_total


# IteraSnapshot - 单次 iteration 的快照记录
@dataclass
class IteraSnapshot:
    """
    记录单次 iteration（调度周期）的详细信息
    """

    iteration: int  # iteration 序号（从 0 开始）
    current_time: float  # 当前仿真时间（秒）
    total_scheduled_tokens: int  # 本次调度的总 token 数
    chunk_sizes: List[int]  # 每个被调度请求分配的 chunk 大小
    all_cached_tokens: List[int]  # 每个请求的已缓存 token 数（调度前）
    all_computed_tokens: List[int]  # 每个请求的已计算 token 数（调度前）
    num_waiting_reqs: int  # 未被调度的等待请求数量
    num_unscheduled_running: int  # 未被调度的 running 请求数量
    duration_ms: float  # 预测的 iteration 执行时长（毫秒）
    scheduled_request_ids: List[str]  # 本次被调度的请求 ID 列表


# SimulationResult - 仿真结果（封装所有返回值）
@dataclass
class SimulationResult:
    """
    仿真运行结果
    """

    history: List[IteraSnapshot]  # iteration 历史快照列表
    final_running: List[ReqSnapshot]  # 最终 running 队列快照
    final_waiting: List[ReqSnapshot]  # 最终 waiting 队列快照
    finished: List[ReqSnapshot]  # 已完成请求快照列表
    end_time: float  # 最终仿真时间（秒）


class ChunkSimulator:
    """
    基于 chunk prefill/decode 的离线仿真器
    """

    def __init__(
        self,
        predictor: DurationPredictor = None,
        decode_cache: bool = True,
        profiling: bool = False,
    ):
        self.predictor = predictor
        self.decode_cache = decode_cache
        self.profiling = profiling

    def run(
        self,
        running_reqs: List[ReqSnapshot],
        waiting_reqs: List[ReqSnapshot],
        max_iters: int = 1000,
        start_time: float = 0.0,
        token_budget: int = 2048,
        decode_only: bool = False,
    ) -> "SimulationResult":
        """
        运行仿真，返回 SimulationResult 对象。

        ========================================================================
        仿真流程：
        ========================================================================
        1. 初始化：将输入快照转换为运行时状态
        2. 迭代循环（最多 max_iters 次）：
           a. 区分 running 队列中的 decode 和 prefill 请求
           b. 按优先级分配 token budget
           c. 预测本次 iteration 的执行时间
           d. 更新请求状态和仿真时间
           e. 清理已完成的请求
        3. 返回：iteration 历史记录和最终状态快照

        """
        current_time = start_time
        history: List[IteraSnapshot] = []

        # 用于缓存纯 decode iteration 的预测结果（优化性能）
        _last_decode_ms: Optional[float] = None
        _last_decode_reqs: Optional[int] = None

        # waiting 队列
        waiting: Deque[ReqSnapshot] = deque(waiting_reqs)
        # running 队列直接转换为 RequestState（可变状态）
        running: List[ReqState] = [
            ReqState.from_snapshot(snap) for snap in running_reqs
        ]
        # 已完成请求列表
        finished: List[ReqSnapshot] = []
        # =====================================================================
        # 主循环：每次循环代表一次 iteration
        # =====================================================================
        for iteration in range(max_iters):

            # =================================================================
            # 步骤 1：一次性区分 running 队列中的 decode 和 prefill 请求
            # =================================================================
            decode_reqs: List[ReqState] = []  # decode 阶段的请求
            prefill_reqs: List[ReqState] = []  # prefill 阶段的请求
            for req in running:
                phase = req.phase()
                if phase == 1:
                    decode_reqs.append(req)
                elif phase == 0:
                    prefill_reqs.append(req)

            remaining_budget = token_budget
            assignments: List[Tuple[ReqState, int]] = []  # 记录调度分配

            # =================================================================
            # 步骤 2：按优先级分配 token budget
            # =================================================================

            # 第一优先级：分配给 running 队列的 decode 请求
            for req in decode_reqs:
                assignments.append((req, 1))
                remaining_budget -= 1

            # 第二优先级：分配给 running 队列的 prefill 请求（FCFS）
            for req in prefill_reqs:
                if remaining_budget <= 0 or decode_only:
                    break
                chunk = min(
                    req.num_prompt_tokens - req.num_computed_tokens, remaining_budget
                )
                assignments.append((req, chunk))
                remaining_budget -= chunk

            # 第三优先级：分配给 waiting 队列的请求（FCFS）
            while waiting and remaining_budget > 0:
                if decode_only:
                    break
                snap = waiting.popleft()
                new_req = ReqState.from_snapshot(snap)

                chunk = min(
                    new_req.num_prompt_tokens - new_req.num_computed_tokens,
                    remaining_budget,
                )

                assignments.append((new_req, chunk))
                remaining_budget -= chunk

                running.append(new_req)

            if not assignments:
                break
            # =================================================================
            # 步骤 3：记录调度信息，用于时长预测
            # =================================================================
            chunk_sizes: List[int] = []
            all_computed_tokens: List[int] = []
            all_cached_tokens: List[int] = []
            for req, chunk in assignments:
                chunk_sizes.append(chunk)
                all_computed_tokens.append(req.num_computed_tokens)
                all_cached_tokens.append(req.num_cached_tokens)

            # 统计running队列中未被调度的请求数量
            num_unscheduled_running = max(0, len(decode_reqs) + len(prefill_reqs) - len(assignments))
            num_waiting_reqs = len(waiting)

            # 计算总调度 token 数
            total_scheduled_tokens = sum(chunk_sizes)
            num_running = len(assignments)

            # =================================================================
            # 步骤 4：预测 iteration 执行时长
            # =================================================================
            pure_decode = (total_scheduled_tokens == num_running)
            if (
                self.decode_cache
                and pure_decode
                and _last_decode_ms is not None
                and _last_decode_reqs == num_running
            ):
                # 复用缓存的预测结果（纯 decode 且运行请求数相同）
                duration_ms = _last_decode_ms
            else:
                # 调用预测器进行预测（使用完全内联的极速方法）
                duration_ms = float(
                    self.predictor.predict_ultrafast(
                        chunk_sizes=chunk_sizes,
                        all_cached_tokens=all_cached_tokens,
                        all_computed_tokens=all_computed_tokens,
                        total_scheduled_tokens=total_scheduled_tokens,
                    )
                )
                # 更新缓存
                if self.decode_cache and pure_decode:
                    _last_decode_ms = duration_ms
                    _last_decode_reqs = num_running
                else:
                    _last_decode_ms = None
                    _last_decode_reqs = None

            # 记录 iteration 快照（仅在 profiling 模式下记录）
            if self.profiling:
                history.append(
                    IteraSnapshot(
                        iteration=iteration,
                        current_time=current_time,
                        total_scheduled_tokens=total_scheduled_tokens,
                        chunk_sizes=chunk_sizes,
                        all_cached_tokens=all_cached_tokens,
                        all_computed_tokens=all_computed_tokens,
                        num_waiting_reqs=num_waiting_reqs,
                        duration_ms=duration_ms,
                        num_unscheduled_running=num_unscheduled_running,
                        scheduled_request_ids=[req.request_id for req, _ in assignments],
                    )
                )

            # =================================================================
            # 步骤 5：更新仿真时间轴
            # =================================================================
            current_time += duration_ms / 1000.0  # 毫秒转秒

            # =================================================================
            # 步骤 6：更新请求状态
            # =================================================================
            for req, chunk in assignments:
                req.num_computed_tokens += chunk
                # 关键：当 prefill 完成（进入 decode 阶段）时记录 TTFT
                if req.ttft is None and req.num_computed_tokens > req.num_prompt_tokens:
                    req.ttft = current_time - req.arrival_time
                    req.ttft_time = current_time

            # 清理已完成请求，并记录完成时间
            still_running = []
            for req in running:
                if req.is_finished():
                    if req.finish_time is None:
                        req.finish_time = current_time
                    finished.append(req.to_snapshot())
                else:
                    still_running.append(req)
            running = still_running

        # =====================================================================
        # 返回最终状态
        # =====================================================================
        return SimulationResult(
            history=history,
            final_running=[req.to_snapshot() for req in running],
            final_waiting=list(waiting),
            finished=finished,
            end_time=current_time,
        )


# ============================================================================
# 模块导出
# ============================================================================
__all__ = [
    "ReqSnapshot",  # 请求快照（不可变）
    "ReqState",  # 请求运行时状态（可变）
    "IteraSnapshot",  # iteration 快照
    "SimulationResult",  # 仿真结果
    "ChunkSimulator",  # 仿真器主类
]
