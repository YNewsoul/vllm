

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

def heuristic_algorithm(env_info:Dict,tpot_slo:float):
        """
        启发式算法，根据环境信息，返回一个 token_budget
        """
        running = env_info['running_requests']
        now_time = env_info["now_time"]
        for req in running:
            output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            if output_tokens>= 2:
                # 解码阶段请求
                tpot = (now_time -req.ttft_time)/(output_tokens - 1)*1000.0
                if tpot > tpot_slo:
                    return 64
        return 2048