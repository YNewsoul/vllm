

from collections import deque
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

        # throughput 相关
        self.throughput_buffer = deque(maxlen=self.config.throughput_buffer_size)
        self.total_token = 0
        self.total_model_time = 0.0
        
        # latency 相关
        self.latency_buffer = deque(maxlen=self.config.latency_buffer_size)
        self.total_latency = 0.0

        # token_budget 相关
        self.select_token_budget = 0.0
        self.actual_token_budget = 0.0
        self.last_token_budget = 0.0
        
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
    
    def add_throughput(self, actual_token: int, model_time: float):
        self.total_token += actual_token
        self.total_model_time += model_time
        self.current_throughput = actual_token / model_time
        if len(self.throughput_buffer)==self.throughput_buffer.maxlen:
            (pop_token,pop_model_time) = self.throughput_buffer.popleft()
            self.total_token -= pop_token
            self.total_model_time -= pop_model_time
        self.throughput_buffer.append((actual_token,model_time))

    def get_throughput(self):
        return self.total_token / self.total_model_time if self.total_model_time > 0 else 0.0
        
    def get_current_throughput(self):
        return self.current_throughput
    
    def add_latency(self, latency: float):
        self.total_latency += latency
        if len(self.latency_buffer)==self.latency_buffer.maxlen:
            pop_latency = self.latency_buffer.popleft()
            self.total_latency -= pop_latency
        self.latency_buffer.append(latency)
    def get_avg_latency(self):
        return self.total_latency / len(self.latency_buffer)
    
    def set_select_token_budget(self,select_token_budget:float):
        self.last_token_budget = self.select_token_budget
        self.select_token_budget = select_token_budget

    def get_select_token_budget(self):
        return self.select_token_budget
        
    def get_last_token_budget(self):
        return self.last_token_budget
