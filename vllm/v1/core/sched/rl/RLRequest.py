

from collections import deque

from .RLConfig import RLSchedulerConfig

class RLRequest:
    def __init__(
    self,
    request_id: str,
    comform_slo: bool,
    ):
        self.request_id = request_id
        self.comform_slo = comform_slo

class RLFinishedReqHandler:
    def __init__(self):
        self.config = RLSchedulerConfig.from_env()
        self.rl_finished_reqs_buffer= deque(maxlen=self.config.rl_finished_reqs_buffer_size)
        self.comform_slo_ratio = 0.0
        self.comform_slo_count = 0
        self.valid_slo_count = 0
        
    def add_rl_finished_req(self, rl_req: RLRequest):
        if rl_req.comform_slo:
            self.comform_slo_count += 1
        else:
            self.valid_slo_count += 1

        if len(self.rl_finished_reqs_buffer) >= self.config.rl_finished_reqs_buffer_size:
            pop_rl_req = self.rl_finished_reqs_buffer.popleft()
            if pop_rl_req.comform_slo:
                self.comform_slo_count -= 1
            else:
                self.valid_slo_count -= 1
        
        self.rl_finished_reqs_buffer.append(rl_req)
        self.comform_slo_ratio = self.comform_slo_count / (self.valid_slo_count + self.comform_slo_count)

    def get_comform_slo_ratio(self):
        return self.comform_slo_ratio