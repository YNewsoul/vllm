# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LLM chunk prefill/decoding 调度仿真
"""

import time
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Optional

# 导入时长预测器，用于预测每次 iteration 的执行时间
from chunk_predictor import DurationPredictor

# RequestSnapshot - 请求快照（不可变的命名元组）
RequestSnapshot = namedtuple(
    "RequestSnapshot",
    [
        "request_id",  # 请求唯一标识符
        "num_computed_tokens",  # 已计算的 token 数量（包括 prefill 和 decode）
        "num_prompt_tokens",  # prompt 的总 token 数量（prefill 目标）
        "arrival_time",  # 请求到达时间（本仿真器中不使用，保留字段）
        "ttft_slo",  # TTFT (Time To First Token) SLO 约束
        "max_tokens",  # 允许生成的最大 token 数量（不含 prompt）
        "ttft",  # 实际的 TTFT 值（首 token 生成时间）
        "finish_time",  # 请求完成时间（保留字段）
    ],
)


# RequestState - 请求运行时状态（可变的数据类）
@dataclass(slots=True)
class RequestState:
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

    @classmethod
    def from_snapshot(cls, snap: RequestSnapshot) -> "RequestState":
        """
        从快照创建运行时状态

        用于将不可变的 RequestSnapshot 转换为可变的 RequestState，
        以便在仿真过程中更新状态。
        """
        return cls(
            request_id=snap.request_id,
            num_prompt_tokens=snap.num_prompt_tokens,
            max_tokens=snap.max_tokens,
            arrival_time=snap.arrival_time,
            ttft_slo=snap.ttft_slo,
            num_computed_tokens=snap.num_computed_tokens,
            num_cached_tokens=snap.num_computed_tokens,
            ttft=snap.ttft,
            finish_time=snap.finish_time,
        )

    def to_snapshot(self) -> RequestSnapshot:
        """
        将运行时状态转换为快照

        用于仿真结束后返回最终状态
        """
        return RequestSnapshot(
            self.request_id,
            self.num_computed_tokens,
            self.num_prompt_tokens,
            self.arrival_time,
            self.ttft_slo,
            self.max_tokens,
            self.ttft,
            self.finish_time,
        )

    def phase(self) -> str:
        if self.is_finished():
            return "finished"
        return ("prefill" if self.num_computed_tokens < self.num_prompt_tokens
                else "decode")

    def is_finished(self) -> bool:
        target_total = self.num_prompt_tokens + self.max_tokens
        return self.num_computed_tokens >= target_total


# IterationTiming - 单次 iteration 的耗时记录
@dataclass
class IterationTiming:
    """
    记录单次 iteration 各阶段的耗时信息（毫秒）
    """
    iteration: int  # iteration 序号
    phase_classify_ms: float  # 阶段分类耗时
    decode_select_ms: float  # decode 请求分配耗时
    prefill_select_ms: float  # prefill 请求分配耗时
    waiting_select_ms: float  # waiting 请求分配耗时
    record_build_ms: float  # 记录构建耗时
    predict_ms: float  # 预测耗时
    predict_cache_hit: bool  # 预测是否命中缓存
    snapshot_build_ms: float  # 快照构建耗时
    state_update_ms: float  # 状态更新耗时
    cleanup_ms: float  # 清理耗时
    iter_total_ms: float  # 本次 iteration 总耗时
    num_scheduled: int  # 本次调度的请求数
    num_decode: int  # decode 阶段请求数
    num_prefill: int  # prefill 阶段请求数
    num_waiting: int  # waiting 队列剩余数


# IterationSnapshot - 单次 iteration 的快照记录
@dataclass
class IterationSnapshot:
    """
    记录单次 iteration（调度周期）的详细信息
    """
    iteration: int  # iteration 序号（从 0 开始）
    current_time: float  # 当前仿真时间（秒）
    total_scheduled_tokens: int  # 本次调度的总 token 数
    chunk_sizes: list[int]  # 每个被调度请求分配的 chunk 大小
    all_cached_tokens: list[int]  # 每个请求的已缓存 token 数（调度前）
    all_computed_tokens: list[int]  # 每个请求的已计算 token 数（调度前）
    num_running_reqs: int  # 本次参与调度的请求数量
    num_waiting_reqs: int  # 未被调度的等待请求数量
    num_unscheduled_running: int  # 未被调度的 running 请求数量
    duration_ms: float  # 预测的 iteration 执行时长（毫秒）
    scheduled_request_ids: list[str]  # 本次被调度的请求 ID 列表


class ChunkSimulator:
    """
    基于 chunk prefill/decode 的离线仿真器
    """

    def __init__(
        self,
        predictor: DurationPredictor,
        enable_decode_cache: bool = True,
    ):
        self.predictor = predictor
        self.enable_decode_cache = enable_decode_cache

        # 用于缓存纯 decode iteration 的预测结果（优化性能）
        self._last_decode_ms: Optional[float] = None
        self._last_decode_reqs: Optional[int] = None

    def run(
        self,
        running_requests: list[RequestSnapshot],
        waiting_requests: list[RequestSnapshot],
        max_iters: int,
        start_time: float = 0.0,
        token_budget: int = 2048,
        limit_token_budget: Optional[int] = None,
        compare_mode: bool = False,
    ) -> tuple[
            list[IterationSnapshot],
            list[RequestSnapshot],
            list[RequestSnapshot],
            list[IterationTiming],
            list[RequestSnapshot],
    ]:
        """
        运行仿真，返回 iteration 快照列表、最终 running 请求快照、
        waiting 请求快照、耗时记录以及已完成请求快照。

        ====================================================================
        仿真流程：
        ====================================================================
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
        history: list[IterationSnapshot] = []
        timing_history: list[IterationTiming] = []  # 耗时记录列表

        # =================================================================
        # 初始化队列
        # =================================================================
        # waiting 队列
        waiting: deque[RequestSnapshot] = deque(waiting_requests)
        # running 队列直接转换为 RequestState（可变状态）
        running: list[RequestState] = [
            RequestState.from_snapshot(snap) for snap in running_requests
        ]
        # 已完成请求列表
        finished: list[RequestSnapshot] = []
        # =================================================================
        # 主循环：每次循环代表一次 iteration
        # =================================================================
        final_iteration = 0
        for iteration in range(max_iters):
            iter_t0 = time.perf_counter()

            if compare_mode and limit_token_budget <= 0:
                # 对比模式下，prefill tokens有约束
                break
            # =============================================================
            # 步骤 1：一次性区分 running 队列中的 decode 和 prefill 请求
            # 优化：避免两次遍历 running 队列
            # =============================================================
            phase_classify_t0 = time.perf_counter()
            decode_reqs: list[RequestState] = []  # decode 阶段的请求
            prefill_reqs: list[RequestState] = []  # prefill 阶段的请求
            for req in running:
                phase = req.phase()
                if phase == "decode":
                    decode_reqs.append(req)
                elif phase == "prefill":
                    prefill_reqs.append(req)
            phase_classify_ms = round(
                (time.perf_counter() - phase_classify_t0) * 1000, 4)

            remaining_budget = token_budget
            assignments: list[tuple[RequestState, int]] = []  # 记录调度分配

            # =============================================================
            # 步骤 2：按优先级分配 token budget
            # =============================================================

            # -------------------------------------------------------------
            # 第一优先级：分配给 running 队列的 decode 请求
            # -------------------------------------------------------------
            decode_t0 = time.perf_counter()
            for req in decode_reqs:
                assignments.append((req, 1))
                remaining_budget -= 1
            decode_select_ms = round((time.perf_counter() - decode_t0) * 1000,
                                     4)

            # -------------------------------------------------------------
            # 第二优先级：分配给 running 队列的 prefill 请求（FCFS）
            # -------------------------------------------------------------

            if limit_token_budget is not None:
                if iteration == 0:
                    # 第一次 iteration，prefill 预算减去 decode 请求的数量
                    limit_token_budget -= len(decode_reqs)
                remaining_budget = min(remaining_budget, limit_token_budget)
            prefill_t0 = time.perf_counter()
            for req in prefill_reqs:
                if remaining_budget <= 0:
                    break
                chunk = min(req.num_prompt_tokens - req.num_computed_tokens,
                            remaining_budget)
                assignments.append((req, chunk))
                remaining_budget -= chunk
                if limit_token_budget is not None:
                    limit_token_budget -= chunk
            prefill_select_ms = round(
                (time.perf_counter() - prefill_t0) * 1000, 4)

            # -------------------------------------------------------------
            # 第三优先级：分配给 waiting 队列的请求（FCFS）
            # -------------------------------------------------------------
            waiting_t0 = time.perf_counter()
            newly_scheduled_from_waiting: list[RequestState] = []
            while waiting and remaining_budget > 0:

                snap = waiting.popleft()
                new_req = RequestState.from_snapshot(snap)

                chunk = min(
                    new_req.num_prompt_tokens - new_req.num_computed_tokens,
                    remaining_budget)

                assignments.append((new_req, chunk))
                remaining_budget -= chunk
                if limit_token_budget is not None:
                    limit_token_budget -= chunk

                # 将新请求加入 running 队列
                running.append(new_req)
                newly_scheduled_from_waiting.append(new_req)
            waiting_select_ms = round(
                (time.perf_counter() - waiting_t0) * 1000, 4)

            if not assignments:
                # 无法调度任何请求，结束仿真
                break
            # =============================================================
            # 步骤 3：记录调度信息，用于时长预测
            # 优化：一次遍历收集所有信息，避免重复列表推导
            # =============================================================
            record_build_t0 = time.perf_counter()
            chunk_sizes: list[int] = []
            all_computed_tokens: list[int] = []
            all_cached_tokens: list[int] = []
            for req, chunk in assignments:
                chunk_sizes.append(chunk)
                all_computed_tokens.append(req.num_computed_tokens)
                all_cached_tokens.append(req.num_cached_tokens)

            # 统计running队列中未被调度的请求数量
            num_unscheduled_running = (len(decode_reqs) + len(prefill_reqs) -
                                       len(assignments))
            num_waiting_reqs = len(waiting)

            # 构建预测器输入记录
            total_scheduled_tokens = sum(chunk_sizes)
            record: dict = {
                "chunk_sizes": chunk_sizes,
                "all_cached_tokens": all_cached_tokens,
                "all_computed_tokens": all_computed_tokens,
                "total_scheduled_tokens": total_scheduled_tokens,
                "num_running_reqs": len(assignments),
                "num_waiting_reqs": num_waiting_reqs,
            }
            record_build_ms = round(
                (time.perf_counter() - record_build_t0) * 1000, 4)

            # =============================================================
            # 步骤 4：预测 iteration 执行时长
            # 优化：对于纯 decode iteration，可以复用上次的预测结果
            # =============================================================

            predict_t0 = time.perf_counter()
            pure_decode = (total_scheduled_tokens == len(chunk_sizes))
            num_running = len(assignments)
            if (self.enable_decode_cache and pure_decode
                    and self._last_decode_ms is not None
                    and self._last_decode_reqs == num_running):
                # 复用缓存的预测结果（纯 decode 且运行请求数相同）
                duration_ms = self._last_decode_ms
                predict_ms = 0.0
                cache_hit = True
            else:
                # 调用预测器进行预测
                duration_ms = float(self.predictor.predict_record(record))
                cache_hit = False
                # 更新缓存
                if self.enable_decode_cache and pure_decode:
                    self._last_decode_ms = duration_ms
                    self._last_decode_reqs = num_running
                else:
                    self._last_decode_ms = None
                    self._last_decode_reqs = None
            predict_ms = round((time.perf_counter() - predict_t0) * 1000, 4)

            # 记录 iteration 快照
            snapshot_build_t0 = time.perf_counter()
            history.append(
                IterationSnapshot(
                    iteration=iteration,
                    current_time=current_time,
                    total_scheduled_tokens=record["total_scheduled_tokens"],
                    chunk_sizes=chunk_sizes,
                    all_cached_tokens=all_cached_tokens,
                    all_computed_tokens=all_computed_tokens,
                    num_running_reqs=len(assignments),
                    num_waiting_reqs=num_waiting_reqs,
                    duration_ms=duration_ms,
                    num_unscheduled_running=num_unscheduled_running,
                    scheduled_request_ids=[
                        req.request_id for req, _ in assignments
                    ],
                ))
            snapshot_build_ms = round(
                (time.perf_counter() - snapshot_build_t0) * 1000, 4)

            # =============================================================
            # 步骤 5：更新仿真时间轴
            # =============================================================
            current_time += duration_ms / 1000.0  # 毫秒转秒

            # =============================================================
            # 步骤 6：更新请求状态
            # =============================================================
            state_update_t0 = time.perf_counter()
            for req, chunk in assignments:
                req.num_computed_tokens += chunk
                # 关键：当 prefill 完成（进入 decode 阶段）时记录 TTFT
                if chunk == 1 and req.ttft is None:
                    req.ttft = current_time - req.arrival_time
            state_update_ms = round(
                (time.perf_counter() - state_update_t0) * 1000, 4)

            # 清理已完成请求，并记录完成时间
            cleanup_t0 = time.perf_counter()
            still_running = []
            for req in running:
                if req.is_finished():
                    if req.finish_time is None:
                        req.finish_time = current_time
                    finished.append(req.to_snapshot())
                else:
                    still_running.append(req)
            running = still_running
            cleanup_ms = round((time.perf_counter() - cleanup_t0) * 1000, 4)

            iter_total_ms = round((time.perf_counter() - iter_t0) * 1000, 4)

            # 记录耗时信息
            timing_history.append(
                IterationTiming(
                    iteration=iteration,
                    phase_classify_ms=phase_classify_ms,
                    decode_select_ms=decode_select_ms,
                    prefill_select_ms=prefill_select_ms,
                    waiting_select_ms=waiting_select_ms,
                    record_build_ms=record_build_ms,
                    predict_ms=predict_ms,
                    predict_cache_hit=cache_hit,
                    snapshot_build_ms=snapshot_build_ms,
                    state_update_ms=state_update_ms,
                    cleanup_ms=cleanup_ms,
                    iter_total_ms=iter_total_ms,
                    num_scheduled=len(assignments),
                    num_decode=len(decode_reqs),
                    num_prefill=len(prefill_reqs),
                    num_waiting=len(waiting),
                ))

            final_iteration = iteration

        # =================================================================
        # 返回最终状态
        # =================================================================
        final_running_snapshots = [req.to_snapshot() for req in running]
        final_waiting_snapshots = list(waiting)
        return (
            history,
            final_running_snapshots,
            final_waiting_snapshots,
            timing_history,
            finished,
            final_iteration,
            current_time,
        )


# ========================================================================
# 模块导出
# ========================================================================
__all__ = [
    "RequestSnapshot",  # 请求快照（不可变）
    "RequestState",  # 请求运行时状态（可变）
    "IterationSnapshot",  # iteration 快照
    "IterationTiming",  # iteration 耗时记录
    "ChunkSimulator",  # 仿真器主类
]
