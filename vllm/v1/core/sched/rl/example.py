#!/usr/bin/env python3
"""
SLA感知调度器使用示例

演示如何配置和使用SLA感知调度器，包括性能监控和调试。
"""

from copy import deepcopy
import os
import time
import sys
from typing import List, Dict, Any

try:
    from .RLConfig import RLSchedulerConfig
    from .RLScheduler import RLScheduler
except ImportError :
    from RLConfig import RLSchedulerConfig
    from RLScheduler import RLScheduler

# 设置示例配置
def setup_example_config():
    """设置RL调度器示例配置"""
    os.environ['VLLM_RL_SCHEDULER_ENABLED'] = 'true'
    os.environ['VLLM_RL_TRAIN_ENABLED'] = 'true'
    
    print("✅ RL 调度器配置已设置")


def demonstrate_config_loading():
    """演示配置加载"""
    print("\n📋 配置加载示例")
    print("=" * 50)
    
    try:
        
        # 从环境变量加载
        config = RLSchedulerConfig.from_env()
        print(f"从环境变量加载的配置: {config}")
        print("✅ 从环境变量加载配置成功")
        return config
        
    except ImportError as e:
        print(f"❌ 无法导入配置模块: {e}")
        return None


def demonstrate_rl_scheduler():
    """演示RL调度器主接口"""
    print("\n🎛️ RL调度器主接口示例")
    print("=" * 50)
    
    try:
        scheduler = RLScheduler()
        print(f"✅ SLA调度器已初始化: {scheduler.enabled}")
        
        # 模拟调度请求
        print("\n📋 模拟调度计算...")
        
        # 创建模拟请求（这里用简化的数据结构）
        class MockRequest:
            def __init__(self, req_id: str, num_computed_tokens: int = 0, num_prompt_tokens: int = 100,
                         slo: float = 10.0):
                self.request_id = req_id
                self.num_computed_tokens = num_computed_tokens
                self.num_prompt_tokens = num_prompt_tokens
                self.slo = slo
                self.arrival_time = time.monotonic()
        
        req_1 = MockRequest("req_1", 50, 100000, 10.0)
        req_2 = MockRequest("req_2", 100, 100, 10.0)
        req_3 = MockRequest("req_3", 100, 100, 10.0)
        req_4 = MockRequest("req_4", 0, 2000000, 10.0)
        req_5 = MockRequest("req_5", 0, 1500000, 10.0)

        running_requests = [
            req_1,   # prefill阶段
            req_2,  # decode阶段
            req_3,  # decode阶段
        ]
        
        waiting_requests = [
            req_4,
            req_5,
        ]
        
        sech_count = 0

        rl_env_info = {'running_requests': None,
        'waiting_requests': None,
        'now_time': None,
        'recent_throughput': 0.0,
        'recent_avg_latency': 0.0,
        'recent_comform_slo_rate': 0.0,
        'current_throughput': 0.0,
        'last_B':0.0,
        'last_S':0.0,
        'select_B':0.0,
        'select_S':0.0,
        'actual_B':0.0,
        'actual_S':0.0}
        
        while True:
            rl_env_info['running_requests'] = running_requests
            rl_env_info['waiting_requests'] = waiting_requests
            rl_env_info['now_time'] = time.monotonic()

            result = scheduler.compute_schedule_decision(rl_env_info)

            if sech_count <= 30:
                print(f"✅ 调度决策结果: {result}")
            for req in running_requests:
                if result['allocation'].get(req.request_id, 0) > 0 and req.num_computed_tokens < req.num_prompt_tokens:
                    req.num_computed_tokens += result['allocation'][req.request_id]
            for req in waiting_requests:
                if result['allocation'].get(req.request_id, 0) > 0 :
                    waiting_requests.remove(req)
                    running_requests.append(req)
                    req.num_computed_tokens += result['allocation'][req.request_id]
            sech_count += 1

            rl_env_info['running_requests'] = running_requests
            rl_env_info['waiting_requests'] = waiting_requests
            rl_env_info['now_time'] = time.monotonic()
            scheduler.record_performance(rl_env_info)
        
    except ImportError as e:
        print(f"❌ 无法导入RL调度器模块: {e}")
        return None


def demonstrate_integration():
    """演示与主调度器的集成"""
    print("\n🔗 调度器集成示例")
    print("=" * 50)
    
    print("SLA调度器与vLLM主调度器的集成点:")
    print("1. 在Scheduler.__init__()中初始化SLA调度器")
    print("2. 在schedule()方法中调用compute_token_budget_and_target()")
    print("3. 在_finalize_and_log_profiling()中记录性能数据")
    print("4. 通过get_sla_scheduler_status()监控状态")
    
    print("\n集成代码示例:")
    print("""
    # 在主调度器中
    if self.sla_scheduler and self.sla_scheduler.enabled:
        token_budget, target_latency = self.sla_scheduler.compute_token_budget_and_target(
            running_requests=self.running,
            waiting_requests=list(self.waiting),
            max_tokens=self.max_num_scheduled_tokens,
            max_batch_size=self.max_num_running_reqs
        )
    else:
        # 回退到原有逻辑
        token_budget = self.max_num_scheduled_tokens
        target_latency = self.slo_tpot_ms
    """)

def main():
    """主函数"""
    print("🚀 SLA感知调度器示例程序")
    print("=" * 60)
    
    # 1. 设置配置
    setup_example_config()
    
    # 2. 演示配置加载
    config = demonstrate_config_loading()
    
    # 4. 演示RL调度器
    scheduler = demonstrate_rl_scheduler()
    
    # # 5. 演示集成
    # demonstrate_integration()
    
    # print("\n🎉 示例程序执行完成！")
    # print("\n要在实际vLLM中使用SLA调度器，请设置相应的环境变量，")
    # print("或参考README.md中的详细配置说明。")


if __name__ == '__main__':
    main()
