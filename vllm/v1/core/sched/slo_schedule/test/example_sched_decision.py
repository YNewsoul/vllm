"""
SLOScheduler.sched_decision 调用示例

本文件演示了如何构造请求对象并调用 sched_decision 方法，
覆盖以下典型场景：

  场景 1：仅有 decode 请求在 running 队列（无需调度优化）
  场景 2：running 队列中混合 prefill + decode 请求（可能触发 decode_only）
  场景 3：running 有 decode 请求 + waiting 有新请求（可能触发 decode_only）
  场景 4：running 中 decode 请求的 TPOT 即将违反 SLO（触发 decode_only）

运行方式：
    cd vllm/v1/core/sched/slo_schedule
    python example_sched_decision.py
"""

import os
import sys
import time
import logging

# 设置日志级别，以便看到 SLOScheduler 的 timing 输出
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

# ---- 将当前目录加入 sys.path，方便直接运行 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slo_scheduler import SLOScheduler


# =============================================================================
# 辅助类：模拟 vllm 的 Request 对象
# =============================================================================
class FakeRequest:
    """
    模拟 vllm Request 对象，提供 SLOScheduler 所需的全部属性。
    """

    def __init__(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_prompt_tokens: int,
        num_cached_tokens: int = 0,
        arrival_time: float = 0.0,
        ttft_slo: float = 0.5,
        max_tokens: int = 128,
        ttft: float = None,
        tpot_type: int = None,
        tpot_slo: float = None,
        accept: bool = None,
        ttft_time: float = None,
    ):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.num_cached_tokens = num_cached_tokens
        self.arrival_time = arrival_time
        self.ttft_slo = ttft_slo
        self.max_tokens = max_tokens
        self.ttft = ttft
        self.tpot_type = tpot_type
        self.tpot_slo = tpot_slo
        self.accept = accept
        self.ttft_time = ttft_time

    def __repr__(self):
        phase = "decode" if self.num_computed_tokens >= self.num_prompt_tokens else "prefill"
        return (f"FakeRequest(id={self.request_id}, phase={phase}, "
                f"computed={self.num_computed_tokens}, prompt={self.num_prompt_tokens})")


# =============================================================================
# 初始化 SLOScheduler
# =============================================================================
def create_scheduler(max_num_scheduled_tokens: int = 2048) -> SLOScheduler:
    """
    创建 SLOScheduler 实例。
    确保工作目录在 slo_schedule 文件夹下，以便找到 models/ 目录中的模型文件。
    """
    # 切换工作目录到脚本所在目录（模型加载依赖相对路径 models/xxx.joblib）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    scheduler = SLOScheduler(max_num_scheduled_tokens=max_num_scheduled_tokens)
    print(f"[OK] SLOScheduler 初始化成功，token_budget={max_num_scheduled_tokens}")
    return scheduler


# =============================================================================
# 场景示例
# =============================================================================

def scenario_1_decode_only(scheduler: SLOScheduler):
    """
    场景 1：running 队列中仅有 decode 请求，waiting 队列为空。
    预期：_schedule_estimate 返回 False，直接跳过仿真，decode_only=False。
    """
    print("\n" + "=" * 70)
    print("场景 1：仅 decode 请求运行，无等待请求")
    print("=" * 70)

    now = time.time()
    # FakeRequest(id, computed, prompt, cached, arrival, ttft_slo, max_tokens, ttft, tpot_type, tpot_slo, accept, ttft_time)
    running = [
        FakeRequest("r1-1", 520, 512, 16, now - 1.0, 0.5, 128, 0.05, 1, 30.0, True, now - 0.95),
        FakeRequest("r1-2", 260, 256,  8, now - 0.8, 0.5,  64, 0.06, 0, 50.0, True, now - 0.74),
        FakeRequest("r1-3", 130, 128,  4, now - 0.6, 0.5, 200, 0.04, 1, 30.0, True, now - 0.56),
        FakeRequest("r1-4", 640, 512, 32, now - 2.0, 0.5, 256, 0.03, 0, 50.0, True, now - 1.97),
        FakeRequest("r1-5", 300, 256, 16, now - 1.5, 0.5, 100, 0.05, 1, 30.0, True, now - 1.45),
    ]
    waiting = []

    result = scheduler.sched_decision({"running": running, "waiting": waiting, "current_time": now})
    print(f"  调度结果: {result}")


def scenario_2_mixed_prefill_decode(scheduler: SLOScheduler):
    """
    场景 2：running 队列中混合 prefill + decode 请求，waiting 为空。
    预期：触发仿真，根据 TPOT 检查决定是否 decode_only。
    """
    print("\n" + "=" * 70)
    print("场景 2：running 中混合 prefill + decode 请求")
    print("=" * 70)

    now = time.time()
    running = [
        # decode 请求
        FakeRequest("r2-d1", 520, 512,  16, now - 1.0, 0.5, 128, 0.05, 1, 30.0, True, now - 0.95),
        FakeRequest("r2-d2", 300, 256,   8, now - 0.8, 0.5,  80, 0.06, 0, 50.0, True, now - 0.74),
        FakeRequest("r2-d3", 640, 512,  32, now - 2.0, 0.5, 256, 0.03, 1, 30.0, True, now - 1.97),
        # prefill 请求
        FakeRequest("r2-p1", 100, 1024,  0, now - 0.2, 0.5,  64, None, 0, 50.0, True, None),
        FakeRequest("r2-p2",  50,  800,  0, now - 0.1, 0.8, 128, None, 1, 30.0, True, None),
    ]
    waiting = []

    result = scheduler.sched_decision({"running": running, "waiting": waiting, "current_time": now})
    print(f"  调度结果: {result}")


def scenario_3_running_with_waiting(scheduler: SLOScheduler):
    """
    场景 3：running 有 decode 请求，waiting 有新请求。
    预期：触发仿真，可能因为新请求抢占 budget 导致 decode 请求 TPOT 违规。
    """
    print("\n" + "=" * 70)
    print("场景 3：running 有 decode + waiting 有新请求")
    print("=" * 70)

    now = time.time()
    running = [
        FakeRequest("r3-d1", 530, 512, 32, now - 2.0, 0.5, 200, 0.05, 0, 50.0, True, now - 1.95),
        FakeRequest("r3-d2", 520, 512, 16, now - 1.5, 0.5, 128, 0.04, 1, 30.0, True, now - 1.46),
        FakeRequest("r3-d3", 260, 256,  8, now - 1.0, 0.5,  64, 0.06, 0, 50.0, True, now - 0.94),
        FakeRequest("r3-d4", 140, 128,  4, now - 0.5, 0.5, 100, 0.03, 1, 30.0, True, now - 0.47),
        FakeRequest("r3-d5", 640, 512, 32, now - 3.0, 0.5, 300, 0.05, 0, 50.0, True, now - 2.95),
    ]
    waiting = [
        FakeRequest("r3-w1", 0, 2048, 0, now - 0.10, 1.0, 128, None, 1, 30.0, None, None),
        FakeRequest("r3-w2", 0, 1024, 0, now - 0.05, 0.8,  64, None, 0, 50.0, None, None),
        FakeRequest("r3-w3", 0,  512, 0, now - 0.02, 0.5, 200, None, 1, 30.0, None, None),
    ]

    result = scheduler.sched_decision({"running": running, "waiting": waiting, "current_time": now})
    print(f"  调度结果: {result}")


def scenario_4_tpot_violation(scheduler: SLOScheduler):
    """
    场景 4：构造一个 TPOT 一定会违反 SLO 的场景。
    - 多个对话类型请求 (tpot_type=1)，紧急 SLO=30ms
    - ttft_time 设置得较早，使得 tpot_temp 远大于 30ms
    - 混入大 prefill 请求抢占 budget
    预期：decode_only=True
    """
    print("\n" + "=" * 70)
    print("场景 4：TPOT 违反 SLO，触发 decode_only")
    print("=" * 70)

    now = time.time()
    running = [
        # decode 请求：TPOT 必然违规（decoded_tokens 少，累积时间长）
        FakeRequest("r4-v1", 520, 512, 16, now - 1.0, 0.5, 100, 0.05, 1, 30.0, True, now - 0.5),
        FakeRequest("r4-v2", 518, 512,  8, now - 0.8, 0.5,  80, 0.04, 1, 30.0, True, now - 0.4),
        FakeRequest("r4-d1", 640, 512, 32, now - 2.0, 0.5, 256, 0.03, 0, 50.0, True, now - 1.97),
        # prefill 请求：大 prompt，抢占 budget
        FakeRequest("r4-p1",   0, 1500,  0, now - 0.1, 1.0,  64, None, 0, 50.0, True, None),
        FakeRequest("r4-p2", 200, 1200,  0, now - 0.3, 0.8, 128, None, 1, 30.0, True, None),
    ]
    waiting = [
        FakeRequest("r4-w1", 0, 2048, 0, now - 0.08, 1.0, 128, None, 1, 30.0, None, None),
        FakeRequest("r4-w2", 0, 1500, 0, now - 0.05, 0.8, 100, None, 0, 50.0, None, None),
        FakeRequest("r4-w3", 0,  800, 0, now - 0.02, 0.5,  64, None, 1, 30.0, None, None),
    ]

    result = scheduler.sched_decision({"running": running, "waiting": waiting, "current_time": now})
    print(f"  调度结果: {result}")
    if result["decode_only"]:
        print("  ✅ 如预期触发了 decode_only！")
    else:
        print("  ℹ️  未触发 decode_only（仿真后 TPOT 仍在 SLO 范围内）")


# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 70)
    print("SLOScheduler.sched_decision 调用示例")
    print("=" * 70)

    # 创建调度器（token_budget = 2048）
    scheduler = create_scheduler(max_num_scheduled_tokens=2048)

    # 依次运行各场景
    scenario_1_decode_only(scheduler)
    scenario_2_mixed_prefill_decode(scheduler)
    scenario_3_running_with_waiting(scheduler)
    scenario_4_tpot_violation(scheduler)

    print("\n" + "=" * 70)
    print("所有场景运行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()
