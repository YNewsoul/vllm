"""
slo_scheduler 模块的单元测试。

运行方式: python -m pytest test_slo_scheduler.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from chunk_simulator import ReqSnapshot, SimulationResult
from slo_scheduler import SLOScheduler, convert_req_to_snapshot


# ---------------------------------------------------------------------------
# 辅助函数: 构建模拟的 vllm 风格请求对象
# ---------------------------------------------------------------------------
def _make_vllm_req(**overrides):
    """创建一个模拟 vllm 请求对象的 SimpleNamespace。"""
    defaults = dict(
        request_id="req-1",
        num_computed_tokens=512,
        num_cached_tokens=16,
        num_prompt_tokens=512,
        arrival_time=0.0,
        ttft_slo=1.0,
        max_tokens=100,
        ttft=0.05,
        tpot_type=None,
        tpot_slo=None,
        accept=None,
        ttft_time=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_snapshot(**overrides):
    """使用合理的默认值创建一个 ReqSnapshot。"""
    defaults = dict(
        request_id="r1",
        num_computed_tokens=600,
        num_cached_tokens=16,
        num_prompt_tokens=512,
        arrival_time=0.0,
        ttft_slo=1.0,
        max_tokens=100,
        ttft=0.05,
        finish_time=None,
        tpot_type=None,
        tpot_slo=None,
        accept=None,
        ttft_time=None,
    )
    defaults.update(overrides)
    return ReqSnapshot(**defaults)


# ---------------------------------------------------------------------------
# 构建一个带有 mock 预测器 / 模拟器的 SLOScheduler 实例
# ---------------------------------------------------------------------------
def _build_scheduler(max_tokens=2048):
    """
    返回一个预测器和模拟器均为 mock 的 SLOScheduler，
    无需真实的模型文件。
    """
    with patch.object(SLOScheduler, "__init__", lambda self, **kw: None):
        sched = SLOScheduler()

    sched.config = SimpleNamespace(model="fake_model.joblib")
    sched.model = "fake_model.joblib"
    sched.max_num_scheduled_tokens = max_tokens

    sched.predictor = MagicMock()
    sched.predictor.predict_ultrafast = MagicMock(return_value=10.0)

    sched.simulator = MagicMock()
    return sched


# ===========================================================================
# 测试用例
# ===========================================================================
class TestConvertReqToSnapshot(unittest.TestCase):
    """convert_req_to_snapshot() 的测试。"""

    def test_basic_conversion(self):
        req = _make_vllm_req()
        snap = convert_req_to_snapshot(req)

        self.assertIsInstance(snap, ReqSnapshot)
        self.assertEqual(snap.request_id, "req-1")
        self.assertEqual(snap.num_computed_tokens, 512)
        self.assertEqual(snap.num_cached_tokens, 16)
        self.assertEqual(snap.num_prompt_tokens, 512)
        self.assertAlmostEqual(snap.arrival_time, 0.0)
        self.assertAlmostEqual(snap.ttft_slo, 1.0)
        self.assertEqual(snap.max_tokens, 100)
        self.assertAlmostEqual(snap.ttft, 0.05)
        self.assertIsNone(snap.finish_time)
        self.assertIsNone(snap.tpot_type)
        self.assertIsNone(snap.tpot_slo)

    def test_missing_optional_fields_default_to_none(self):
        """如果 vllm 请求缺少可选属性，则回退为 None/0。"""
        req = SimpleNamespace(
            request_id="req-2",
            num_computed_tokens=0,
            num_prompt_tokens=1024,
            arrival_time=1.0,
            max_tokens=256,
        )
        snap = convert_req_to_snapshot(req)
        self.assertEqual(snap.num_cached_tokens, 0)
        self.assertIsNone(snap.ttft_slo)
        self.assertIsNone(snap.ttft)
        self.assertIsNone(snap.tpot_type)
        self.assertIsNone(snap.tpot_slo)
        self.assertIsNone(snap.accept)
        self.assertIsNone(snap.ttft_time)


class TestScheduleEstimate(unittest.TestCase):
    """SLOScheduler._schedule_estimate() 的测试。"""

    def setUp(self):
        self.sched = _build_scheduler()

    def _decode_req(self, rid="d1"):
        """处于 decode 阶段的请求 (computed >= prompt)。"""
        return _make_vllm_req(
            request_id=rid, num_computed_tokens=600, num_prompt_tokens=512
        )

    def _prefill_req(self, rid="p1"):
        """处于 prefill 阶段的请求 (computed < prompt)。"""
        return _make_vllm_req(
            request_id=rid, num_computed_tokens=100, num_prompt_tokens=512
        )

    def test_running_only_decode(self):
        """running 中仅有 decode 请求，无 waiting -> False。"""
        running = [self._decode_req()]
        self.assertFalse(self.sched._schedule_estimate(running, []))

    def test_running_only_prefill(self):
        """running 中仅有 prefill 请求，无 waiting -> False。"""
        running = [self._prefill_req()]
        self.assertFalse(self.sched._schedule_estimate(running, []))

    def test_running_mixed_no_waiting(self):
        """running 中同时存在 prefill 和 decode 请求，无 waiting -> True。"""
        running = [self._decode_req(), self._prefill_req()]
        self.assertTrue(self.sched._schedule_estimate(running, []))

    def test_running_decode_with_waiting(self):
        """running 中有 decode 请求，waiting 非空 -> True。"""
        running = [self._decode_req()]
        waiting = [self._prefill_req("w1")]
        self.assertTrue(self.sched._schedule_estimate(running, waiting))

    def test_running_prefill_with_waiting(self):
        """running 中仅有 prefill 请求，waiting 非空 -> False（无 decode）。"""
        running = [self._prefill_req()]
        waiting = [self._prefill_req("w1")]
        self.assertFalse(self.sched._schedule_estimate(running, waiting))

    def test_empty_running_empty_waiting(self):
        """两者均为空 -> False。"""
        self.assertFalse(self.sched._schedule_estimate([], []))

    def test_empty_running_with_waiting(self):
        """running 为空，有 waiting -> False。"""
        waiting = [self._prefill_req("w1")]
        self.assertFalse(self.sched._schedule_estimate([], waiting))


class TestCountRunningTypes(unittest.TestCase):
    """SLOScheduler._count_running_types() 的测试。"""

    def setUp(self):
        self.sched = _build_scheduler()

    def test_all_decode(self):
        running = [
            _make_vllm_req(num_computed_tokens=600, num_prompt_tokens=512),
            _make_vllm_req(num_computed_tokens=1024, num_prompt_tokens=512),
        ]
        n_prefill, n_decode = self.sched._count_running_types(running)
        self.assertEqual(n_prefill, 0)
        self.assertEqual(n_decode, 2)

    def test_all_prefill(self):
        running = [
            _make_vllm_req(num_computed_tokens=100, num_prompt_tokens=512),
        ]
        n_prefill, n_decode = self.sched._count_running_types(running)
        self.assertEqual(n_prefill, 1)
        self.assertEqual(n_decode, 0)

    def test_mixed(self):
        running = [
            _make_vllm_req(num_computed_tokens=600, num_prompt_tokens=512),
            _make_vllm_req(num_computed_tokens=100, num_prompt_tokens=512),
            _make_vllm_req(num_computed_tokens=512, num_prompt_tokens=512),          ]
        n_prefill, n_decode = self.sched._count_running_types(running)
        self.assertEqual(n_prefill, 1)
        self.assertEqual(n_decode, 2)


class TestPredictDecodeDuration(unittest.TestCase):
    """SLOScheduler._predict_decode_duration() 的测试。"""

    def setUp(self):
        self.sched = _build_scheduler()

    def test_with_decode_requests(self):
        """应调用预测器并返回秒数。"""
        self.sched.predictor.predict_ultrafast.return_value = 20.0  # 20 毫秒

        final_running = [
            # decode 请求 (computed >= prompt)
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=600,
                num_prompt_tokens=512,
                num_cached_tokens=16,
            ),
            # prefill 请求 (computed < prompt) — 应被跳过
            _make_snapshot(
                request_id="p1",
                num_computed_tokens=100,
                num_prompt_tokens=512,
                num_cached_tokens=16,
            ),
        ]
        duration_s = self.sched._predict_decode_duration(final_running)
        self.assertAlmostEqual(duration_s, 0.020)

        # 验证预测器仅使用 decode 请求进行调用
        call_kwargs = self.sched.predictor.predict_ultrafast.call_args
        self.assertEqual(call_kwargs.kwargs["chunk_sizes"], [1])
        self.assertEqual(call_kwargs.kwargs["total_scheduled_tokens"], 1)

    def test_no_decode_requests(self):
        """如果不存在 decode 请求，应返回 0.0 且不调用预测器。"""
        final_running = [
            _make_snapshot(num_computed_tokens=100, num_prompt_tokens=512),
        ]
        duration_s = self.sched._predict_decode_duration(final_running)
        self.assertAlmostEqual(duration_s, 0.0)
        self.sched.predictor.predict_ultrafast.assert_not_called()

    def test_empty_list(self):
        duration_s = self.sched._predict_decode_duration([])
        self.assertAlmostEqual(duration_s, 0.0)


class TestBatchForward(unittest.TestCase):
    """SLOScheduler.batch_forward() 的测试。"""

    def setUp(self):
        self.sched = _build_scheduler()

    def _mock_simulation_result(self, final_running, end_time=1.01):
        return SimulationResult(
            history=[],
            final_running=final_running,
            final_waiting=[],
            finished=[],
            end_time=end_time,
        )

    def test_no_tpot_violation_returns_false(self):
        """当没有请求违反 TPOT SLO 时，decode_only 应为 False。"""
        final_running = [
            # accept=True 但无 tpot_type 的 decode 请求 -> 跳过 tpot 检查
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=600,
                num_prompt_tokens=512,
                ttft=0.05,
                accept=True,
                tpot_type=None,
                tpot_slo=None,
            ),
        ]
        self.sched.simulator.run.return_value = self._mock_simulation_result(
            final_running, end_time=1.01
        )
        self.sched.predictor.predict_ultrafast.return_value = 10.0  # 10 ms

        running_snaps = [_make_snapshot(request_id="d1")]
        waiting_snaps = []

        decode_only, timing = self.sched.batch_forward(running_snaps, waiting_snaps, 1.0)
        self.assertFalse(decode_only)
        self.assertIn("  simulator_run", timing)
        self.assertIn("  predict_decode", timing)
        self.assertIn("  tpot_check", timing)

    def test_tpot_type0_violation_returns_true(self):
        """
        tpot_type=0 (编程场景): TPOT 基于所有剩余 token 计算。
        如果计算出的 tpot 超过 tpot_slo，decode_only 应为 True。
        """
        # 请求: prompt=512, computed=520 (已 decode 8 个 token), max_tokens=100
        # num_computed_tokens > num_prompt_tokens + 1 => 符合 tpot 检查条件
        final_running = [
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=520,
                num_prompt_tokens=512,
                num_cached_tokens=16,
                max_tokens=100,
                ttft=0.05,
                accept=True,
                tpot_type=0,
                tpot_slo=30.0,  # 每 token 30 毫秒的 SLO（非常紧凑）
                ttft_time=0.5,
            ),
        ]
        # 仿真耗时 0.1 秒, end_time = 1.1
        self.sched.simulator.run.return_value = self._mock_simulation_result(
            final_running, end_time=1.1
        )
        # 纯 decode 预测: 每次迭代 10 毫秒
        self.sched.predictor.predict_ultrafast.return_value = 10.0

        running_snaps = [_make_snapshot(request_id="d1")]

        decode_only, _ = self.sched.batch_forward(running_snaps, [], 1.0)
        # 以给定数值计算 tpot 会很大，应触发 decode_only
        self.assertFalse(decode_only)

    def test_tpot_type1_violation_returns_true(self):
        """
        tpot_type=1 (对话场景): 每个 token 的 TPOT 必须始终满足 SLO。
        """
        # decoded_tokens = 520 - 512 = 8
        # tpot_temp = (end_time - ttft_time) / decoded_tokens * 1000
        #           = (1.1 - 0.5) / 8 * 1000 = 75 毫秒   -> 违反 30 毫秒 SLO（紧急）
        final_running = [
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=520,
                num_prompt_tokens=512,
                num_cached_tokens=16,
                max_tokens=100,
                ttft=0.05,
                accept=True,
                tpot_type=1,
                tpot_slo=30.0,  # 紧急 SLO: 30ms
                ttft_time=0.5,
            ),
        ]
        self.sched.simulator.run.return_value = self._mock_simulation_result(
            final_running, end_time=1.1
        )
        self.sched.predictor.predict_ultrafast.return_value = 10.0

        decode_only, _ = self.sched.batch_forward([_make_snapshot()], [], 1.0)
        self.assertTrue(decode_only)

    def test_tpot_type1_within_slo_returns_false(self):
        """
        tpot_type=1 配合宽松的 SLO -> decode_only 应为 False。
        """
        # decoded_tokens = 520 - 512 = 8
        # tpot_temp = (end_time - ttft_time) / decoded_tokens * 1000
        #           = (1.1 - 0.8) / 8 * 1000 = 37.5 毫秒  -> 在 50 毫秒 SLO 以内（宽松）
        final_running = [
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=520,
                num_prompt_tokens=512,
                num_cached_tokens=16,
                max_tokens=100,
                ttft=0.05,
                accept=True,
                tpot_type=1,
                tpot_slo=50.0,  # 宽松 SLO: 50ms
                ttft_time=0.8,  # 调整使 tpot_temp=37.5ms < 50ms
            ),
        ]
        self.sched.simulator.run.return_value = self._mock_simulation_result(
            final_running, end_time=1.1
        )
        self.sched.predictor.predict_ultrafast.return_value = 10.0

        decode_only, _ = self.sched.batch_forward([_make_snapshot()], [], 1.0)
        self.assertFalse(decode_only)

    def test_request_not_accepted_is_skipped(self):
        """accept=None 或 accept=False 的请求在 TPOT 检查中应被跳过。"""
        final_running = [
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=520,
                num_prompt_tokens=512,
                accept=None,  # 未被接受
                tpot_type=1,
                tpot_slo=30.0,  # 如果参与检查则会违反
                ttft_time=0.5,
            ),
        ]
        self.sched.simulator.run.return_value = self._mock_simulation_result(
            final_running, end_time=1.1
        )
        self.sched.predictor.predict_ultrafast.return_value = 10.0

        decode_only, _ = self.sched.batch_forward([_make_snapshot()], [], 1.0)
        self.assertFalse(decode_only)

    def test_recently_transitioned_request_skipped(self):
        """
        num_computed_tokens == num_prompt_tokens + 1 的请求
        （刚从 prefill 转为 decode）不应触发 TPOT 检查。
        """
        final_running = [
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=513,  # == 512 + 1, 边界值
                num_prompt_tokens=512,
                accept=True,
                tpot_type=0,
                tpot_slo=30.0,
                ttft_time=0.5,
            ),
        ]
        self.sched.simulator.run.return_value = self._mock_simulation_result(
            final_running, end_time=1.1
        )
        self.sched.predictor.predict_ultrafast.return_value = 10.0

        decode_only, _ = self.sched.batch_forward([_make_snapshot()], [], 1.0)
        # 513 不大于 512 + 1，因此应被跳过
        self.assertFalse(decode_only)


class TestSchedDecision(unittest.TestCase):
    """顶层 sched_decision() 方法的测试。"""

    def setUp(self):
        self.sched = _build_scheduler()

    def test_no_schedule_needed(self):
        """当 _schedule_estimate 返回 False 时，应返回默认决策。"""
        running = [
            _make_vllm_req(num_computed_tokens=600, num_prompt_tokens=512),
        ]
        waiting = []
        sched_state = {"running": running, "waiting": waiting, "current_time": 1.0}

        decision = self.sched.sched_decision(sched_state)

        self.assertFalse(decision["decode_only"])
        self.assertEqual(decision["token_budget"], 2048)
        # 模拟器不应被调用
        self.sched.simulator.run.assert_not_called()

    def test_schedule_needed_triggers_batch_forward(self):
        """当 _schedule_estimate 返回 True 时，应调用 batch_forward。"""
        running = [
            _make_vllm_req(request_id="d1", num_computed_tokens=600, num_prompt_tokens=512),
            _make_vllm_req(request_id="p1", num_computed_tokens=100, num_prompt_tokens=512),
        ]
        waiting = []
        sched_state = {"running": running, "waiting": waiting, "current_time": 1.0}

        # mock simulator.run 返回无 TPOT 违规的结果
        final_running = [
            _make_snapshot(request_id="d1", num_computed_tokens=601, num_prompt_tokens=512, accept=True),
        ]
        self.sched.simulator.run.return_value = SimulationResult(
            history=[], final_running=final_running,
            final_waiting=[], finished=[], end_time=1.01,
        )

        decision = self.sched.sched_decision(sched_state)

        self.assertIn("decode_only", decision)
        self.assertIn("token_budget", decision)
        self.sched.simulator.run.assert_called_once()

    def test_decision_returns_decode_only_true_on_violation(self):
        """端到端测试: TPOT 违规应传播 decode_only=True。"""
        running = [
            _make_vllm_req(request_id="d1", num_computed_tokens=600, num_prompt_tokens=512),
            _make_vllm_req(request_id="p1", num_computed_tokens=100, num_prompt_tokens=512),
        ]
        waiting = []
        sched_state = {"running": running, "waiting": waiting, "current_time": 1.0}

        # final_running 中包含一个将违反 TPOT SLO 的请求
        final_running = [
            _make_snapshot(
                request_id="d1",
                num_computed_tokens=520,
                num_prompt_tokens=512,
                num_cached_tokens=16,
                max_tokens=100,
                accept=True,
                tpot_type=1,
                tpot_slo=30.0,
                ttft_time=0.5,
                ttft=0.05,
            ),
        ]
        self.sched.simulator.run.return_value = SimulationResult(
            history=[], final_running=final_running,
            final_waiting=[], finished=[], end_time=1.1,
        )
        self.sched.predictor.predict_ultrafast.return_value = 10.0

        decision = self.sched.sched_decision(sched_state)
        self.assertTrue(decision["decode_only"])


if __name__ == "__main__":
    unittest.main()
