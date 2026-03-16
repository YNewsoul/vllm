# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# profiling相关的导入
import json
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

# RL/SLA感知调度器导入
try:
    from .slo_schedule import SLOScheduler
    SLO_SCHEDULER_AVAILABLE = True
except ImportError as e:
    logger.warning("SLO调度器不可用: %s", e)
    SLO_SCHEDULER_AVAILABLE = False

class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        # include_finished_set控制是否在update_from_outputs()返回的
        # EngineCoreOutputs中包含一个单独的已完成请求ID集合。
        # 目前用于多引擎场景下高效追踪请求生命周期。
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # 调度约束条件
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # 为调度器创建KVConnector。注意每个Worker都会有一个
        # 对应的KVConnector（Role=WORKER）。
        # KV Connector用于推送/拉取远程KV，支持P/D和卸载功能。
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector_v1(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            vllm_config.parallel_config.data_parallel_rank,
        )
        # 初始化GPU块信息
        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # req_id -> Request 请求存储结构
        self.requests: dict[str, Request] = {}
        # 请求优先级队列
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # 在上一步和当前步之间完成的请求ID。
        # 用于通知工作节点释放这些请求的缓存状态。
        # 在每个调度步骤结束时清空。
        # 已完成请求的ID集合，用于通知工作节点释放缓存
        self.finished_req_ids: set[str] = set()

        # KV Connector: 正在进行异步KV加载或接收的请求
        self.finished_recving_kv_req_ids: set[str] = set()

        # 优化：缓存CachedRequestData对象，避免在每个调度步骤重新创建。
        # Request id -> CachedRequestData队列
        self._cached_reqs_data: dict[
            str, deque[CachedRequestData]] = defaultdict(deque)

        # 编码器相关
        # 如果适用，计算编码器缓存大小
        # 注意：目前我们对计算和存储使用相同的预算。
        # 当我们为跨请求嵌入缓存实现编码器缓存时，可以改变这一点。
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # 注意：此处的"编码器"包括视觉编码器（以及必要时的投影器）。
        # 目前，我们假设编码器也使用Transformer架构（例如ViT）。
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # 注意：对于没有编码器的模型（例如纯文本模型），
        # 编码器缓存不会初始化，因为这些模型的缓存大小为0。
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)
        # 投机解码相关
        speculative_config = vllm_config.speculative_config

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # 创建KV缓存管理器
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )

        # 初始化
        self._initialize()

    def _initialize(self):
        # Batch跟踪
        self.batch_id = 0
        self.last_sched_end_time: Optional[float] = None
        self.batch_profiling_data: Optional[dict] = None
        
        # Profiling设置 - 仅在启用时创建目录
        self.enable_profiling = os.getenv(
            'VLLM_ENABLE_SCHEDULER_PROFILING', 'false').lower() == 'true'
        self.profiling_log_file: Optional[str] = None
        self._profiling_queue: Optional[Queue[Optional[str]]] = None
        self._profiling_writer_stop: Optional[threading.Event] = None
        self._profiling_writer_thread: Optional[threading.Thread] = None
        
        if self.enable_profiling:
            now = datetime.now()
            profiling_log_dir = os.getenv(
                'VLLM_SCHEDULER_PROFILING_LOG', 'profiling')
            profiling_log_filename = os.getenv(
                'VLLM_SCHEDULER_PROFILING_LOG_FILENAME', f"profiling_{now.strftime('%Y-%m-%d %H_%M_%S')}.jsonl")
            date_dir = os.path.join(profiling_log_dir, now.strftime("%Y-%m-%d"))
            os.makedirs(date_dir, exist_ok=True)
            self.profiling_log_file = os.path.join(
                date_dir, profiling_log_filename)
            logger.info("The profiling log file: %s", self.profiling_log_file)
            self._profiling_queue = Queue()
            self._profiling_writer_stop = threading.Event()
            self._start_profiling_writer_thread()

        # SLO调度器设置
        self.slo_scheduler = None
        logger.info("SLO Scheduler available: %s, enabled: %s", SLO_SCHEDULER_AVAILABLE, os.getenv(
                'VLLM_SLO_SCHEDULER_ENABLED', 'false').lower() == 'true')
        if SLO_SCHEDULER_AVAILABLE and os.getenv(
                'VLLM_SLO_SCHEDULER_ENABLED', 'false').lower() == 'true':
            try:
                self.slo_scheduler = SLOScheduler()
                logger.info("SLO Scheduler initialized successfully!")
            except Exception as e:
                logger.warning("SLO Scheduler initialization failed: %s", e)
        self.slo_sched = False

    def schedule(self) -> SchedulerOutput:
        # 注意(woosuk)关于调度算法：
        # 调度器中没有"解码阶段"或"预填充阶段"的区分。
        # 每个请求只有num_computed_tokens和num_tokens_with_spec。
        # num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)。
        # 在每个步骤中，调度器尝试为请求分配token，
        # 使每个请求的num_computed_tokens能够跟上其num_tokens_with_spec。
        # 这种设计足够通用，可以涵盖分块预填充、前缀缓存、
        # 投机解码，以及未来的"跳跃解码"优化。

        sched_start_time = time.time()  # Profiling: 记录调度开始时间

        scheduled_new_reqs: list[Request] = []  # 新调度的请求
        scheduled_resumed_reqs: list[Request] = []  # 恢复的请求（从抢占状态）
        scheduled_running_reqs: list[Request] = []  # 正在运行的请求
        preempted_reqs: list[Request] = []  # 被抢占的请求

        # 注意：structured_output_request_ids将使用结构化输出的请求的
        # request_id映射到运行请求索引。
        # 这将帮助我们确定如何切片语法位掩码，
        # 并仅对使用结构化解码的请求应用有效掩码。
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}  # 记录每个请求已调度的token数

        token_budget = self.max_num_scheduled_tokens
        # 使用SLO调度器
        if self.slo_scheduler:
            schedule_decision = self.slo_scheduler.sched_decision(
                self.get_schedule_state())
            token_budget = schedule_decision['token_budget']
            self.slo_sched = schedule_decision['slo_sched']

        init_token_budget = token_budget

        # 编码器相关
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # 投机解码相关
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        scheduled_timestamp = time.monotonic()  # 用于日志记录

        # 首先，调度RUNNING状态的请求
        req_index = 0
        # 遍历RUNNING队列，分配token预算
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # 计算该请求需要的新token数量
            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)

            if self.slo_scheduler and num_new_tokens > 1 and schedule_decision.get(
                    'decode_only', False):
                # 纯解码，跳过prefill请求
                req_index += 1
                continue

            # 长预填充截断处理
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)

            num_new_tokens = min(num_new_tokens, token_budget)
            # 确保输入位置不超过最大模型长度。
            # 使用投机解码时必要。
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)

            # 调度编码器输入
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # 请求无法调度，可能的原因：
                # 1. 没有新token需要调度。这可能在PP>1时发生，
                #    我们已经调度了所有提示符token但它们还未完成。
                # 2. 编码器预算已耗尽。
                # 3. 编码器缓存已耗尽。
                # 注意(woosuk)：此处使用`continue`而不是`break`，
                # 我们不严格遵循FCFS调度策略，
                # 允许较低优先级的请求被调度。
                req_index += 1
                continue

            num_draft_tokens = max(
                num_new_tokens + request.num_computed_tokens -
                request.num_tokens, 0)

            # 尝试分配KV cache空间，如果失败则抢占低优先级请求
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_draft_tokens=num_draft_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # 请求无法调度。
                    # 抢占最低优先级请求。
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # 没有更多请求可以抢占
                        can_schedule = False
                        break
                else:
                    # 请求可以被调度
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # 调度请求。
            # 将请求添加进已调度running请求列表
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # 性能注意：在分块预填充情况下，
                # 请求可能不包含任何新token。
                # 因此，我们可能会引入一些额外的循环来填充位掩码，
                # 这可能是一个很大的空操作。
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = (
                new_blocks.get_block_ids())
            # 将新调度的token添加进num_scheduled_tokens字典
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

            req_index += 1

            # 投机解码相关
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # 将spec_token_ids列表截断为num_scheduled_spec_tokens长度
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # 编码器相关
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # 分配编码器缓存
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # 记录scheduled_running_reqs中的LoRA
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # 使用临时队列收集需要跳过的请求，
        # 稍后将它们放回等待队列的头部
        skipped_waiting_requests: deque[Request] = deque()

        # 接下来，调度WAITING请求
        if not preempted_reqs:
            # 只有在没有抢占发生时才调度新请求
            while self.waiting and token_budget > 0:

                if self.slo_scheduler and schedule_decision.get(
                        'decode_only', False):
                    # 纯解码，跳过prefill请求
                    break
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting[0]

                # KVTransfer：如果仍在等待远程KV，则跳过请求
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # 如果结构化输出请求仍在等待FSM编译，则跳过
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.popleft()
                        skipped_waiting_requests.appendleft(request)
                        continue

                # 检查添加请求是否仍然满足max_loras约束
                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    # 调度将超过max_loras，跳过
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # 获取已缓存的token
                if request.num_computed_tokens == 0:
                    # 获取本地缓存的token
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # 如果使用KVConnector，获取外部缓存的token
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                    # 总计算token数（本地 + 外部）
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)

                # KVTransfer：WAITING请求在异步KV接收完成后
                # num_computed_tokens > 0
                else:
                    # 处理之前已经启动异步KV传输但现在完成的请求
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget

                # KVTransfer：正在加载远程KV，不为新工作分配资源
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # 要调度的token数量
                else:
                    # 我们使用`request.num_tokens`而不是
                    # `request.num_prompt_tokens`来考虑恢复的请求，
                    # 这些请求有输出token。
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    # 长预填充截断处理
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # 调度编码器输入
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_budget)
                        if num_new_tokens == 0:
                            # 请求无法调度
                            break

                # 尝试分配KV cache空间
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                )
                if new_blocks is None:
                    # 请求无法调度
                    break

                # KVTransfer：connector使用此信息来确定是否需要加载。
                # 注意：此信息用于确定该请求是否需要加载。
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )
                # 从等待队列中弹出请求
                self.waiting.popleft()
                if load_kv_async:
                    # 如果异步加载，分配内存并将请求
                    # 置为WAITING_FOR_REMOTE_KV状态
                    skipped_waiting_requests.appendleft(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                # 将请求加入running队列
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                # 根据请求的状态将请求加入对应队列
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = (
                    self.kv_cache_manager.get_block_ids(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # 计算前缀缓存token数量
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # 编码器相关
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # 分配编码器缓存
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # 将任何跳过的请求放回等待队列头部
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # 检查调度约束是否满足
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        # assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # 由于RUNNING队列中的某些请求可能不会在此步骤中被调度，
        # 因此调度的请求总数可能小于len(self.running)。
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # 获取running队列中所有请求的最长公共前缀。
        # 这可以潜在地用于级联注意力。
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )
        # 构建调度器输出
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids是调度器中的现有状态，
            # 而不是在此步骤中新调度的。
            # 它包含在上一步和当前步之间完成的请求ID。
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # 注意(Kuntai)：此函数设计用于多种用途：
        # 1. 规划KV缓存存储
        # 2. 将所有KV缓存加载/保存操作封装为不透明对象
        # 3. 清理connector的内部状态
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # 在请求被调度后推进已计算的token数量。
        # 1. 当前步骤的scheduler_output必须包含
        #    原始调度的token数以确定输入ID。
        # 2. 在此处推进已计算的token数，允许我们
        #    在下一个调度步骤中立即再次调度预填充请求。
        # 3. 如果某些token（例如投机token）稍后被拒绝，
        #    已计算的token数将在update_from_output中调整。
        # 更新每个req的num_computed_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()

        # Profiling: 记录调度统计信息，但不立即写入文件（等待model run完成）
        sched_end_time = time.time()
        self.last_sched_end_time = sched_end_time
        if self.enable_profiling:
            self._prepare_schedule_profiling(
                schedule_duration=sched_end_time - sched_start_time,
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                token_budget=init_token_budget)

        return scheduler_output

    def _make_cached_request_data(
        self,
        request: Request,
        num_scheduled_tokens: int,
        num_scheduled_spec_tokens: int,
        new_block_ids: tuple[list[int], ...],
        resumed_from_preemption: bool,
    ) -> CachedRequestData:
        # 优化：缓存CachedRequestData对象，避免在每个调度步骤重新创建。
        num_computed_tokens = request.num_computed_tokens
        num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens
        new_token_ids = request.all_token_ids[
            num_computed_tokens:num_computed_tokens + num_regular_tokens]

        req_data_queue = self._cached_reqs_data.get(request.request_id)
        if req_data_queue:
            req_data = req_data_queue.popleft()
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_token_ids = new_token_ids
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            # 没有缓存的请求数据，或所有缓存的请求数据已被
            # 调度的请求使用。
            req_data = CachedRequestData.from_request(request,
                                                      resumed_from_preemption,
                                                      new_token_ids,
                                                      new_block_ids)
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        确定在当前步骤中需要调度哪些编码器输入，
        并相应地更新`num_new_tokens`和编码器token预算。

        编码器输入将在以下情况下被调度：
        - 其输出token与此步骤中计算的token范围重叠，即
          [num_computed_tokens, num_computed_tokens + num_new_tokens)。
        - 它尚未计算并存储在编码器缓存中。
        - 有足够的编码器token预算来处理它。
        - 编码器缓存有空间存储它。

        如果编码器输入由于缓存或预算限制而无法调度，
        此方法会调整`num_new_tokens`，仅调度在不可调度的
        编码器输入之前的解码器token。

        注意：num_computed_tokens包括本地缓存的块
        和外部缓存的块（通过KVConnector）。
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # 编码器输出在两个范围重叠时需要：
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) 和
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # 此步骤不需要编码器输入
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # 编码器输入已经计算并存储
                # 在解码器的KV缓存中
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # 编码器输入已经计算并缓存
                continue

            # 如果不允许编码器输入分块，我们不想部分调度
            # 多模态项。如果调度范围只能覆盖多模态输入的一部分，
            # 则回滚到多模态项之前。
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # 编码器缓存已满或编码器预算已耗尽。
                # 注意(woosuk)：我们假设编码器输入token应该
                # 一起处理，因为编码器通常使用双向注意力。
                if num_computed_tokens < start_pos:
                    # 我们只调度编码器输入之前的解码器token
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # 由于前缀缓存，num_computed_tokens大于
                    # start_pos，即使其编码器输入不可用。
                    # 在这种情况下，我们无法为该请求调度任何token。
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """处理LLM运行一个iteration后的数据"""

        model_run_duration = time.time() - self.last_sched_end_time

        # Profiling数据记录
        # if self.enable_profiling and self.slo_sched:
        if self.enable_profiling:
            self._finalize_and_log_profiling(model_run_duration)
        self.slo_sched = False

        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        # logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens  # 每个请求调度的token数量

        # 初始化用于存储处理结果的数据结构
        new_running: list[Request] = []  # 存储处理后仍在运行的请求
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # 注意(woosuk)：由于len(self.running)可能达到1K或更多，
        # 下面的循环可能成为性能瓶颈。我们应该尽最大努力
        # 避免在循环内执行昂贵的操作。
        for request in self.running:
            req_id = request.request_id
            # 获取该请求在本步骤被调度的token数量
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # 请求在此步骤未被调度。
                # 如果请求在本步骤未被调度，直接添加到新的运行列表
                new_running.append(request)
                continue

            # 记录ttft时间
            if request.ttft is None and num_tokens_scheduled == 1:
                now_time = time.time()
                request.ttft = now_time - request.arrival_time
                if request.ttft < request.ttft_slo:
                    request.safeguard = True

            # 获取请求在模型输出中的索引位置
            req_index = model_runner_output.req_id_to_index[req_id]
            # 获取模型为该请求生成的token ID
            generated_token_ids = sampled_token_ids[req_index]

            # 处理投机解码的情况
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens表示当前步骤中处理的token数，
                # 考虑已调度的token和拒绝。如果某些token被拒绝，
                # num_computed_tokens会减少被拒绝token的数量，
                # 即：len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)。
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            # 处理编码器缓存（如果有）
            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # 优化：如果集合为空，避免list(set)
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    mm_positions = request.mm_positions[input_id]
                    start_pos = mm_positions.offset
                    num_tokens = mm_positions.length
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # 编码器输出已经处理并存储
                        # 在解码器的KV缓存中
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            # 初始化变量
            stopped = False  # 请求是否已停止
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None

            # 添加生成的token并检查停止条件
            # 注意：如果请求仍在prefill阶段，模型运行器
            # 应该为该请求返回空的token ID
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                # 将生成的token添加到请求的输出token列表中
                request.append_output_token_ids(output_token_id)

                # 检查停止条件并更新请求状态。
                # 这必须在创建EngineCoreOutput之前调用。
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    # 如果请求已停止，释放请求资源
                    kv_transfer_params = self._free_request(request)
                    # 如有必要，修剪新生成的token
                    del new_token_ids[num_new:]
                    break

            # 处理结构化输出（如果启用）
            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # 注意：如果use_structured_output，
                # structured_output_request不应该为None，
                # 我们上面已经检查过，所以可以安全地忽略类型警告
                request.structured_output_request.grammar.accept_tokens(
                    req_id, new_token_ids)

            # 将新生成的投机token ID添加到请求中
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    # 需要在new_token_ids被接受后执行
                    request.spec_token_ids = metadata.grammar.validate_tokens(
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # 获取此请求的提示符logprobs
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or kv_transfer_params:

                # 为此请求添加EngineCoreOutput
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    ))

            else:
                # 不变量：EngineCore不返回部分预填充输出
                assert not prompt_logprobs_tensors

            if not stopped:
                # 如果请求未停止，将其添加到新的运行列表
                new_running.append(request)

        # KV Connector：更新已完成KV传输的状态
        self._update_from_kv_xfer_finished(model_runner_output)

        # 将缓存的请求数据返回到队列以便重用
        for req_data in scheduler_output.scheduled_cached_reqs:
            # 注意(rob)：由于我们在上面释放了已停止的请求，
            # 将已停止的请求添加到_cached_reqs_data会导致内存泄漏。
            if req_data.req_id not in self.finished_req_ids:
                self._cached_reqs_data[req_data.req_id].append(req_data)

        # 更新运行中的请求列表
        self.running = new_running

        # 为在此步骤中有输出的所有客户端创建EngineCoreOutputs
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # 包含自上次发送输出以来完成的请求ID
            for client_index, finished_set in finished_req_ids.items():
                # 为此客户端在EngineCoreOutputs中设置已完成的请求集合
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if engine_core_outputs:
            # 仅将统计信息返回给其中一个前端
            next(iter(engine_core_outputs.values())).scheduler_stats = (
                self.make_stats(spec_decoding_stats))

        return engine_core_outputs

    def get_request_counts(self) -> tuple[int, int]:
        """返回 (num_running_reqs, num_waiting_reqs)。"""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """处理来自调度器外部的完成信号。

        例如，当客户端断开连接时，API服务器可以中止请求。
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # 无效的请求ID
                continue

            if request.status == RequestStatus.RUNNING:

                self.running.remove(request)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:

        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self._cached_reqs_data.pop(request_id, None)
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        assert request.request_id not in self._cached_reqs_data
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()

    ########################################################################
    # KV Connector相关方法
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        如果适用，调用KV connector的request_finished()方法。

        返回可选的KV传输参数，以包含在请求输出中。
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector：检查request_id是否已完成接收。

        finished_recving_kv_req_ids列表在上一步骤的
        update_from_output中基于工作器端connector填充。

        当KV传输准备就绪时，我们缓存块，
        请求状态将从WAITING_FOR_REMOTE_KV移回WAITING。
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # 现在块已准备就绪，实际缓存它们
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # 处理请求token数小于一个块的情况
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # 更新请求状态用于调度
        request.num_computed_tokens = num_computed_tokens

        # 返回表示已准备就绪
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      model_runner_output: ModelRunnerOutput):
        """
        KV Connector：根据输出更新调度器状态。

        工作器端connector将finished_recving和
        finished_sending请求添加到输出中。
        * 如果finished_sending：释放块
        * 如果finished_recving：添加到状态，
          以便我们可以在下一步骤中调度请求。
        """
        # KV Connector:: 更新上一步的接收和发送状态
        for req_id in (model_runner_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (model_runner_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])

    def _start_profiling_writer_thread(self) -> None:
        """启动后台线程，异步写入profiling日志。"""
        if self._profiling_writer_thread is not None:
            return
        self._profiling_writer_thread = threading.Thread(
            target=self._profiling_writer_loop,
            name="SchedulerProfilingWriter",
            daemon=True,
        )
        self._profiling_writer_thread.start()

    def _profiling_writer_loop(self) -> None:
        """后台线程：从队列取出profiling数据并写入文件。"""
        while not self._profiling_writer_stop.is_set():
            try:
                payload = self._profiling_queue.get(timeout=1.0)
            except Empty:
                continue
            if payload is None:
                break
            try:
                with open(self.profiling_log_file, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")
            except Exception as e:
                logger.warning(
                    "Failed to write profiling data asynchronously: %s", e)

    def _prepare_schedule_profiling(self, schedule_duration: float,
                                    num_scheduled_tokens: dict[str, int],
                                    total_num_scheduled_tokens: int,
                                    token_budget: float) -> None:
        """准备调度profiling信息，不写入文件。"""
        now_time = time.time()
        # 仅记录当前step中被调度的请求，避免running中未调度请求触发KeyError。
        scheduled_running = [
            req for req in self.running if req.request_id in num_scheduled_tokens
        ]

        chunk_sizes: list[int] = []
        computed_tokens: list[int] = []
        cached_tokens: list[int] = []
        ttft_list: list[Optional[str]] = []
        ttft_slo_list: list[float] = []
        remaining_ttft: list[Optional[str]] = []
        meet_ttft: list[Optional[str]] = []
        tbt: list[Optional[str]] = []
        decode_tokens: list[Optional[int]] = []
        max_tokens_list: list[int] = []
        request_data_id: list[Optional[int]] = []

        # 单次遍历本轮被调度请求
        for req in scheduled_running:
            chunk_sizes.append(num_scheduled_tokens[req.request_id])
            computed_tokens.append(req.num_computed_tokens)
            cached_tokens.append(req.num_cached_tokens)
            ttft_slo_list.append(req.ttft_slo)
            max_tokens_list.append(req.max_tokens)
            request_data_id.append(req.request_data_id)
            tbt.append(req.tbt)

            req_ttft = req.ttft
            if req_ttft is not None:
                ttft_list.append(f"{req_ttft:.3f}")
                meet_ttft.append("T" if req_ttft <= req.ttft_slo else "F")
                decode_tokens.append(req.num_computed_tokens - req.num_prompt_tokens)
                remaining_ttft.append(None)
            else:
                ttft_list.append(None)
                remaining_ttft.append(
                    f"{req.ttft_slo - (now_time - req.arrival_time):.3f}")
                meet_ttft.append(None)
                decode_tokens.append(None)

        self.batch_profiling_data = {
            "batch_id": self.batch_id,
            "time": f"{now_time:.3f}",
            "token_budget": token_budget,
            "sched_tokens": total_num_scheduled_tokens,
            "slo_sched": self.slo_sched,
            "schedule_ms": f"{schedule_duration * 1000:.3f}",
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_scheduled": len(chunk_sizes),
            "req_data_id": request_data_id,
            "chunk_sizes": chunk_sizes,
            "computed_tokens": computed_tokens,
            "cached_tokens": cached_tokens,
            "ttft_slo": ttft_slo_list,
            "ttft": ttft_list,
            "remaining_ttft": remaining_ttft,
            "meet_ttft": meet_ttft,
            "tbt": tbt,
            "decode_tokens": decode_tokens,
            "max_tokens": max_tokens_list,
        }

    def _finalize_and_log_profiling(self, model_run_duration: float) -> None:
        """完成profiling数据并写入文件"""

        # 添加model run时间
        self.batch_profiling_data[
            "model_run_ms"] = f"{model_run_duration * 1000:.3f}"

        # 写入日志文件
        try:
            payload = json.dumps(self.batch_profiling_data,
                                    ensure_ascii=False)
            if self._profiling_writer_thread:
                self._profiling_queue.put(payload)
            else:
                logger.warning("Failed to write profiling data")
        except Exception as e:
            logger.warning("Failed to write profiling data: %s", e)

        self.batch_id += 1

    def get_schedule_state(self) -> dict:
        """
        获取running和waiting请求队列，用于仿真。
        """
        return {
            'running': self.running,
            'waiting': self.waiting,
            'current_time': time.time(),
            'token_budget': self.max_num_scheduled_tokens,
        }
