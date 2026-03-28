

from collections import deque

from vllm.v1.request import Request
import math
import random
from collections import deque
from typing import Deque, Tuple
from typing import Dict

try:
    from .RLConfig import RLSchedulerConfig
except ImportError:
    from RLConfig import RLSchedulerConfig

class RLDataCollection:
    """ RL 数据收集的类"""
    def __init__(self):
        self.config = RLSchedulerConfig.from_env()

        # slo 相关
        self.rl_finished_reqs_buffer= deque(maxlen=self.config.rl_finished_reqs_buffer_size)
        self.comform_slo_count = 0

        # token_budget 相关
        self.select_token_budget = 0.0
        self.last_token_budget = self.config.max_num_scheduled_tokens
        
    def add_rl_finished_req(self,comform_slo:bool):
        if comform_slo:
            self.comform_slo_count += 1
        if len(self.rl_finished_reqs_buffer) == self.rl_finished_reqs_buffer.maxlen:
            pop_rl_req = self.rl_finished_reqs_buffer.popleft()
            if pop_rl_req.comform_slo:
                self.comform_slo_count -= 1
        self.rl_finished_reqs_buffer.append(comform_slo)

    def get_comform_slo_ratio(self):
        return self.comform_slo_count / len(self.rl_finished_reqs_buffer) if len(self.rl_finished_reqs_buffer) > 0 else 0.0
    
    def set_select_token_budget(self,select_token_budget:float):
        self.last_token_budget = self.select_token_budget
        self.select_token_budget = select_token_budget

    def get_select_token_budget(self):
        return self.select_token_budget
        
    def get_last_token_budget(self):
        return self.last_token_budget

def _eval_ttft_goodput(
        waiting: Deque,
        select_token_budget: int,
        model_run_time: float) -> Tuple[float, list[int]]:
    """返回 (总代价, 违约索引列表)"""
    violators = []
    acc_time = 0.0
    acc_tokens = 0
    for idx, req in enumerate(waiting):
        acc_tokens += req.num_prompt_tokens
        p_time = math.ceil(acc_tokens / select_token_budget) * model_run_time
        if p_time > req.ttft_slo:
            violators.append(idx)
        acc_time += p_time
    goodput = (len(waiting) - len(violators))/acc_time
    return goodput, violators

def adjust_waiting_seq(
    waiting: Deque,
    select_token_budget: int,
    model_run_time: float,
) -> Deque:
    """
    对 waiting 队列做模拟退火重排序。邻域：将随机选中的不满足 TTFT 的请求前移一格。
    """
    iters = 10
    init_temp = 5.0
    temp_threshold = 0.1
    cooling = 0.9
    cur_seq = deque(waiting)
    cur_goodput, cur_violators = _eval_ttft_goodput(cur_seq, select_token_budget, model_run_time)
    best_seq, best_goodput = deque(cur_seq), cur_goodput
    temp = init_temp

    for _ in range(iters):
        # if temp < temp_threshold:
        #     break
        
        # 1.选邻域：优先选非队头的违约请求前移，否则随机交换两个请求
        candidates = [i for i in cur_violators if i > 0]
        new_seq = deque(cur_seq)
        if candidates:
            # 随机选一个违约请求前移，生成新序列
            idx = random.choice(candidates)
            req = new_seq[idx]
            new_seq.remove(req)
            new_seq.insert(idx - 1, req)
        else:
            # 随机交换两个请求，生成新序列
            i, j = random.sample(range(len(new_seq)), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        # 2.计算新序列的 goodput
        new_goodput, new_violators = _eval_ttft_goodput(new_seq, select_token_budget, model_run_time)

        # 3.判断是否接受新序列
        accept = False
        delta = new_goodput - cur_goodput
        if delta > 0:
            accept = True
        # elif math.exp(delta / max(temp, 1e-6)) > random.random():
        #     accept = True
        
        # 4.接受新序列
        if accept:
            cur_seq, cur_goodput, cur_violators = new_seq, new_goodput, new_violators
            if cur_goodput > best_goodput:
                best_seq, best_goodput = deque(cur_seq), cur_goodput
        # temp *= cooling

    return best_seq

def get_chunk_size(cs_2048:int,prompt_count:int,d_token:int,time_slo:float):
    """
    根据 time_slo 动态调整 chunk_size
    """
    cs_64 = math.ceil(max(0,prompt_count - 2048*cs_2048)/64)
    predict_time = min(d_token,cs_2048)*450.0 \
                    + min(max(0,d_token-cs_2048),cs_64)*40.0 \
                    + max(0,d_token-cs_2048-cs_64)*15.0

    if cs_2048 == 0:
        if predict_time <= time_slo:
            return 64
        return 2048
    
    if predict_time <= time_slo:
        return 2048
    else:
        return get_chunk_size(cs_2048-1,prompt_count,d_token,time_slo)

def heuristic_algorithm(env_info:Dict,tpot_slo:float):
        """
        启发式算法，根据环境信息，返回一个 token_budget
        """
        running = env_info['running_requests']
        waiting = env_info['waiting_requests']
        now_time = env_info["now_time"]

        prompt_count = 0
        for req in reversed(running):
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens <= 0:
                # prefill 请求
                prompt_count -= output_tokens
                ttft_remaining_time = (math.ceil(prompt_count/2048))*0.450
                slack_ms = req.ttft_slo - (now_time-req.arrival_time)
                if ttft_remaining_time > slack_ms:
                    # 满足ttft为第一要义
                    return 2048
                break
        for req in reversed(waiting):
            prompt_count += req.num_prompt_tokens
            ttft_remaining_time = (math.ceil(prompt_count/2048))*0.450
            slack_ms = req.ttft_slo - (now_time-req.arrival_time)
            if ttft_remaining_time > slack_ms:
                # 满足ttft为第一要义
                return 2048
        
        cs_2048 = math.ceil(prompt_count/2048)

        for req in running:
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens > 0:
                d_token = req.max_tokens - output_tokens
                time_slo = tpot_slo*req.max_tokens - (now_time - req.ttft_time)
                chunk_size =  get_chunk_size(cs_2048,prompt_count,d_token,time_slo)
                if chunk_size != 2048:
                    return 64
        return 2048