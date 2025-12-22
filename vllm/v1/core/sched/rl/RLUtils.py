

from collections import deque

from vllm.v1.request import Request

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

# def reconstruct_waiting_queue(running:list[Request],waiting:list[Request],prefill_ms:float):
#     for req in running:
#         if req.num_computed_tokens - req.num_prompt_tokens < 0: