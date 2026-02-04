# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SLO 感知调度器模块。

本模块提供 SLOScheduler 类，用于根据调度器的 running 和 waiting 队列信息，
动态调整 token budget，以优化 TTFT SLO 达标率。
"""

from collections import deque
from typing import Any

from .chunk_predictor import DurationPredictor
from .chunk_simulator import ChunkSimulator, RequestSnapshot


def convert_request_to_snapshot(request: Any) -> RequestSnapshot:
    """
    将 vllm Request 对象转换为仿真器需要的 RequestSnapshot
    """
    return RequestSnapshot(
        request_id=request.request_id,
        num_computed_tokens=request.num_computed_tokens,
        num_prompt_tokens=request.num_prompt_tokens,
        arrival_time=request.arrival_time,
        ttft_slo=getattr(request, 'ttft_slo', None),
        max_tokens=request.max_tokens,
        ttft=getattr(request, 'ttft', None),
        finish_time=None,
    )


class SLOScheduler:
    """
    SLO 感知调度器。
    """

    def __init__(
        self,
        max_num_scheduled_tokens: int = 2048,
    ):

        # chunk 队列，用于存储待分配的 token budget（整数队列）
        self.chunk_queue: deque[int] = deque(maxlen=1000)
        self.max_iterations = 1000
        self.max_num_scheduled_tokens = max_num_scheduled_tokens

        # 缓存最近一次转换的 RequestSnapshot
        self.running_snapshots: list[RequestSnapshot] = []
        self.waiting_snapshots: list[RequestSnapshot] = []

        self.predictor = DurationPredictor.load(
            "models/profiling_result_a6000_model_predictor.joblib")
        self.simulator = ChunkSimulator(predictor=self.predictor)

    def compute_sched_decision(self, schedule_state: dict) -> int:
        sched_decision = {"pure_decode": False}

        if not self._chunk_schedule_estimate(schedule_state):
            return sched_decision

        if (not schedule_state['finish_request']
                and not schedule_state['new_request']):
            if self.chunk_queue:
                self.chunk_queue.popleft()
                sched_decision["pure_decode"] = True
                return sched_decision

        # 转换队列为 RequestSnapshot 格式
        self.running_snapshots = [
            convert_request_to_snapshot(req)
            for req in schedule_state['running_requests']
        ]
        self.waiting_snapshots = [
            convert_request_to_snapshot(req)
            for req in schedule_state['waiting_requests']
        ]

        pure_decode = self.batch_forward(schedule_state['current_time'])
        sched_decision["pure_decode"] = pure_decode

        return sched_decision

    def _chunk_schedule_estimate(self, schedule_state: dict):
        running = schedule_state['running_requests']
        waiting = schedule_state['waiting_requests']
        num_running = len(running)
        num_waiting = len(waiting)

        num_prefill, num_decode = self._count_running_types(running)

        if num_running != 0 and num_waiting == 0:
            # 1、有请求运行，无请求等待
            if num_prefill != 0 and num_decode != 0:
                # 1.1、运行请求包括 prefill 和 decode 请求
                return True
            return False

        elif num_running != 0 and num_waiting != 0:
            # 2、 有请求运行，有请求等待
            return num_decode != 0
        return False

    def _count_running_types(self, running):
        num_prefill = 0
        num_decode = 0

        for req in running:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                num_decode += 1
            else:
                num_prefill += 1
        return num_prefill, num_decode

    def batch_forward(self, current_time: float) -> bool:
        # 判断能否以最大chunk size 进行调度
        result = self.simulator.run(
            running_requests=self.running_snapshots,
            waiting_requests=self.waiting_snapshots,
            max_iters=self.max_iterations,
            start_time=current_time,
            token_budget=self.max_num_scheduled_tokens,
            num_prefill_tokens=self.max_num_scheduled_tokens,
        )

        min_num_prue_decode = self.max_num_scheduled_tokens
        prue_decode = False
        for req in result['running_snapshots']:
            topt = (req.estimated_finish_time -
                    req.ttft) / req.max_tokens * 1000
            if topt > req.tpot_slo:
                min_num_prue_decode = min(min_num_prue_decode,
                                          req.sim_decode_tokens)
                prue_decode = True
        if prue_decode:
            self.chunk_queue.append([1] * min_num_prue_decode)
            return prue_decode
        return prue_decode


# 模块导出
__all__ = [
    'SLOScheduler',
    'convert_request_to_snapshot',
    'RequestSnapshot',
]
