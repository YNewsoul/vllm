# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SLO 感知调度器模块。

本模块提供 SLOScheduler 类，用于根据调度器的 running 和 waiting 队列信息，
动态调整 token budget，以优化 TTFT SLO 达标率。
"""

import os
import time
import logging

from typing import Any

logger = logging.getLogger(__name__)

try:
    from .chunk_predictor import DurationPredictor
    from .chunk_simulator import ChunkSimulator, ReqSnapshot
    from .config import SloSchedulerConfig
except ImportError:
    from chunk_predictor import DurationPredictor
    from chunk_simulator import ChunkSimulator, ReqSnapshot
    from config import SloSchedulerConfig

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
        ttft=getattr(req, 'ttft', None),
        finish_time=None,
        tpot_type=getattr(req, 'tpot_type', None),
        tpot_slo=getattr(req, 'tpot_slo', None),
        accept=getattr(req, 'accept', None),
        ttft_time=getattr(req, 'ttft_time', None),
    )


class SLOScheduler:
    """
    SLO 感知调度器
    """
    def __init__(
        self,
        max_num_scheduled_tokens: int = 2048,
    ):
        self.config = SloSchedulerConfig.from_env()

        self.model = self.config.model
        self.max_num_scheduled_tokens = max_num_scheduled_tokens

        model_path = os.path.join("models", self.model)
        self.predictor = DurationPredictor.load(model_path)
        self.simulator = ChunkSimulator(predictor=self.predictor)

    def sched_decision(self, sched_state: dict) -> dict:
        timing = {}  # record elapsed time for each step
        total_start = time.perf_counter()

        sched_decision = {"decode_only": False,
                          "token_budget": self.max_num_scheduled_tokens}

        running = sched_state['running']
        waiting = sched_state['waiting']

        # Step 1: schedule estimate
        t0 = time.perf_counter()
        need_schedule = self._schedule_estimate(running, waiting)
        timing['schedule_estimate'] = time.perf_counter() - t0

        if not need_schedule:
            timing['total'] = time.perf_counter() - total_start
            self._print_timing(timing, sched_decision)
            return sched_decision

        # Step 2: convert requests to snapshots
        t0 = time.perf_counter()
        running_snapshots = [
            convert_req_to_snapshot(req)
            for req in running
        ]
        waiting_snapshots = [
            convert_req_to_snapshot(req)
            for req in waiting
        ]
        timing['convert_snapshot'] = time.perf_counter() - t0

        current_time = sched_state['current_time']

        # Step 3: batch_forward (with internal timing)
        t0 = time.perf_counter()
        sched_decision["decode_only"], forward_timing = self.batch_forward(
            running_snapshots, waiting_snapshots, current_time
        )
        timing['batch_forward'] = time.perf_counter() - t0
        timing.update(forward_timing)

        timing['total'] = time.perf_counter() - total_start
        self._print_timing(timing, sched_decision)
        return sched_decision

    @staticmethod
    def _print_timing(timing: dict, decision: dict):
        """Print elapsed time for each step in sched_decision."""
        lines = ["[SLOScheduler Timing] decision=%s" % decision]
        for key, value in timing.items():
            lines.append("  %-30s: %.6f s (%.3f ms)" % (key, value, value * 1000))
        logger.info("\n".join(lines))

    def _schedule_estimate(self, running: list, waiting: list):
        """判断是否需要进行chunk size调整调度"""
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
        """计算running队列中prefill和decode请求的数量"""
        num_prefill = 0
        num_decode = 0

        for req in running:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                num_decode += 1
            else:
                num_prefill += 1
        return num_prefill, num_decode

    def _predict_decode_duration(self, final_running: list[ReqSnapshot]) -> float:
        """
        直接用 predictor 预测纯 decode 一次 iteration 的耗时（毫秒转秒）。
        等价于对 final_running 中所有 decode 请求做一次 decode_only 仿真，
        但跳过了仿真器的快照转换、状态管理等开销。
        """
        chunk_sizes = []
        all_cached_tokens = []
        all_computed_tokens = []

        for req in final_running:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                # decode request: chunk_size = 1
                chunk_sizes.append(1)
                all_cached_tokens.append(req.num_cached_tokens)
                all_computed_tokens.append(req.num_computed_tokens)

        if not chunk_sizes:
            return 0.0

        duration_ms = self.predictor.predict_ultrafast(
            chunk_sizes=chunk_sizes,
            all_cached_tokens=all_cached_tokens,
            all_computed_tokens=all_computed_tokens,
            total_scheduled_tokens=len(chunk_sizes),
        )
        return float(duration_ms) / 1000.0

    def batch_forward(self, running_snapshots: list[ReqSnapshot],
                      waiting_snapshots: list[ReqSnapshot],
                      current_time: float) -> tuple[bool, dict]:
        """判断是否需要decode only，同时返回内部各步耗时"""
        forward_timing = {}

        # 1、先进行一次PD融合计算
        t0 = time.perf_counter()
        result_pd = self.simulator.run(
            running_reqs=running_snapshots,
            waiting_reqs=waiting_snapshots,
            max_iters=1,
            start_time=current_time,
            token_budget=self.max_num_scheduled_tokens,
        )
        forward_timing['  simulator_run'] = time.perf_counter() - t0
        dura_time_pd = result_pd.end_time - current_time

        # 2、直接用predictor预测纯decode一次iteration的耗时，
        #    避免第二次仿真的快照转换和状态管理开销
        t0 = time.perf_counter()
        dura_time_d = self._predict_decode_duration(result_pd.final_running)
        forward_timing['  predict_decode'] = time.perf_counter() - t0

        decode_only = False
        # 3、判断是否需要decode only
        t0 = time.perf_counter()
        for req in result_pd.final_running:
            if not req.accept:
                continue
            if req.num_computed_tokens > req.num_prompt_tokens + 1:
                # "> + 1" 是因为不考虑刚从prefill转到decode的请求，因为输出从0->1是属于prefill的时间
                # decode请求，非 > 表明是已经有token输出的decode请求
                # 而不是从 prefill 转到decode的请求
                if req.tpot_type == 0:
                    # 0 表示coding 类型的请求,这类请求只要最终请求完成时计算的tpot满足slo即可
                    # 剩余要生成的token数
                    remain_tokens = req.max_tokens + req.num_prompt_tokens - req.num_computed_tokens
                    tpot = (dura_time_pd + dura_time_d * remain_tokens + current_time - req.ttft_time) / req.max_tokens * 1000
                    if tpot > req.tpot_slo:
                        decode_only = True
                        break
                elif req.tpot_type == 1:
                    # 1 表示对话类型的请求,这类请求要求每输出一个token时计算的tpot都要满足slo
                    decoded_tokens = req.num_computed_tokens - req.num_prompt_tokens
                    tpot_temp = (result_pd.end_time - req.ttft_time)/decoded_tokens * 1000
                    if tpot_temp > req.tpot_slo:
                        decode_only = True
                        break
        forward_timing['  tpot_check'] = time.perf_counter() - t0

        return decode_only, forward_timing


# 模块导出
__all__ = [
    'SLOScheduler',
    'convert_req_to_snapshot',
    'ReqSnapshot',
]
