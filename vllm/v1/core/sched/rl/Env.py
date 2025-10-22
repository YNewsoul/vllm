
import numpy as np
from typing import Dict

class Env:
    """
    模拟文档中的LLM推理引擎环境
    核心适配：动态workload、GPU硬件约束（KV缓存/批大小）、迭代式调度（tens到hundreds毫秒）
    """
    def __init__(self):
        
        self.before_env_info = None # 上一个step的环境信息
        self.after_env_info =  None # 当前step的环境信息

    def set_before_env_info(self,before_env_info:Dict):
        self.before_env_info = None
        self.before_env_info = before_env_info
        
    def set_after_env_info(self,after_env_info:Dict):
        self.after_env_info = None
        self.after_env_info = after_env_info

    def get_env_info(self):
        return self.before_env_info, self.after_env_info
    
    def reset(self) -> np.ndarray:
        pass
        # """重置环境（训练初始化）"""
        # self.unfinished_reqs = []
        # self.waiting_queue = []
        # self.gpu_free_kv = 1.0
        # self.last_bs = (6, 512)
        # self.slo_history.clear()
        # self._init_workload()
        # return self._get_state()