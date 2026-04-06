from typing import List

try:
    from .batch_forwarder import BatchForwarder
    from .utils import ReqSnapshot
except ImportError:
    from batch_forwarder import BatchForwarder
    from utils import ReqSnapshot


def select_dp_batch_decision(
    batch_forwarder: BatchForwarder,
    decoding: List[ReqSnapshot],
    prefilling: List[ReqSnapshot],
    waiting: List[ReqSnapshot],
    token_budget: int,
    cur_max_budget: int,
    current_time: float,
    max_iter_time: float,
) -> dict[str, object] | None:
    """
    基于 sliding 策略做二级选择 + 背包动态规划，返回本轮建议的分配方案。

    返回格式：
    - token_budget: 建议本轮使用的总预算（含 decoding 的 1-token 预算）
    - assigned: 每个 request_id 对应的分配 token 数
    - selected_slack: 最终选中的 DP 方案 slack（秒）
    """

    # 预算必须满足：至少覆盖 decoding，且不超过外部传入 token_budget。
    cur_max_budget = max(len(decoding), min(token_budget, cur_max_budget))
    # 用该预算跑一次前向，拿到“当前分配长什么样”。
    _, cur_max_assigned = batch_forwarder.forward(
        decoding=decoding,
        prefilling=prefilling,
        waiting=waiting,
        token_budget=cur_max_budget,
    )

    # 汇总所有 prefill 请求的剩余 token 与 TTFT slack（ddl）。
    all_prefill = prefilling + waiting
    prefill_req_by_id: dict[str, ReqSnapshot] = {}
    prefill_remaining_by_id: dict[str, int] = {}
    prefill_slack_by_id: dict[str, float] = {}
    for req in all_prefill:
        # 剩余 token 数量
        remaining_tokens = req.num_prompt_tokens - req.num_computed_tokens
        # TTFT slack（ddl）
        slack = req.arrival_time + req.ttft_slo - current_time
        prefill_req_by_id[req.request_id] = req
        prefill_remaining_by_id[req.request_id] = remaining_tokens
        prefill_slack_by_id[req.request_id] = slack

    # 若当前分配中的 prefill（assigned > 1）存在 slack < max_iter_time，
    # 说明按当前时长执行会触发 TTFT 违约风险，需要进入二级选择。
    has_risk_prefill = False
    for req_id, assigned_tokens in cur_max_assigned.items():
        if assigned_tokens <= 1:
            continue
        if req_id not in prefill_slack_by_id:
            continue
        if prefill_slack_by_id[req_id] < max_iter_time:
            has_risk_prefill = True
            break

    if not has_risk_prefill:
        # 没有风险，返回 None
        return None

    # 候选集合：仅保留仍需 prefill 且 slack > 0 的请求。
    candidate_items: list[dict[str, float | int | str]] = []
    for req_id, req in prefill_req_by_id.items():
        slack = prefill_slack_by_id[req_id]
        if slack <= 0:
            continue
        candidate_items.append(
            {
                "request_id": req.request_id,
                "remaining": prefill_remaining_by_id[req_id],
                "slack": slack,
            }
        )
    if not candidate_items:
        return {
            "token_budget": cur_max_budget,
            "assigned": cur_max_assigned,
            "max_iter_time": max_iter_time,
        }

    # 按 ddl（slack）升序，越靠前越紧急。
    candidate_items.sort(key=lambda item: (float(item["slack"]), int(item["remaining"])))

    # decoding 请求固定每个分配 1 token。
    decode_assigned = {req.request_id: 1 for req in decoding}
    decode_tokens = len(decode_assigned)

    # 最优解比较维度：
    # 1) 满足请求数更多；2) 总价值更大；3) 占用预算更小。
    best_count = -1
    best_value = float("-inf")
    best_budget = cur_max_budget
    best_assigned = cur_max_assigned
    best_slack = max_iter_time

    # 以每个请求的 ddl 作为一个“锚点时间”：
    # 对应一个预算上限，再在该上限下做组合优化。
    for anchor_index, anchor_item in enumerate(candidate_items):
        anchor_slack = float(anchor_item["slack"])
        anchor_req_id = str(anchor_item["request_id"])
        anchor_remaining = int(anchor_item["remaining"])

        # 将锚点 ddl 转成本轮可用预算。
        anchor_budget, _ = batch_forwarder.time_to_token_budget(
            decoding=decoding,
            prefilling=prefilling,
            waiting=waiting,
            target_iter_ms=anchor_slack * 1000.0,
        )
        anchor_budget = max(decode_tokens, min(token_budget, anchor_budget))
        prefill_capacity = anchor_budget - decode_tokens
        # 锚点请求本身都塞不下，则该锚点无解。
        if anchor_remaining > prefill_capacity:
            continue

        # 只考虑 ddl 不小于锚点的请求（即“更宽松或同等紧急”的集合）。
        subset = sorted(
            candidate_items[anchor_index:],
            key=lambda item: int(item["remaining"]),
        )
        slack_sum = sum(float(item["slack"]) for item in subset)
        remaining_sum = sum(int(item["remaining"]) for item in subset)
        if slack_sum <= 0 or remaining_sum <= 0:
            continue

        # 价值函数：1 / (slack占比 + remaining占比)
        # 数值越大，表示“紧急且代价可控”。
        for item in subset:
            slack_ratio = float(item["slack"]) / slack_sum
            remaining_ratio = int(item["remaining"]) / remaining_sum
            denominator = slack_ratio + remaining_ratio
            item["value"] = 1.0 / denominator if denominator > 0 else 0.0

        # 锚点请求必须入选，先扣掉其容量。
        remain_capacity = prefill_capacity - anchor_remaining
        selected_ids = {anchor_req_id}
        selected_value = 0.0
        for item in subset:
            if str(item["request_id"]) == anchor_req_id:
                selected_value += float(item["value"])
                break

        others = [
            item for item in subset
            if str(item["request_id"]) != anchor_req_id
        ]

        # 在剩余容量上做 0/1 背包：
        # weight = remaining tokens, value = 上述价值函数。
        if remain_capacity > 0 and others:
            item_count = len(others)
            dp = [
                [0.0 for _ in range(remain_capacity + 1)]
                for _ in range(item_count + 1)
            ]
            take = [
                [False for _ in range(remain_capacity + 1)]
                for _ in range(item_count + 1)
            ]

            for index in range(1, item_count + 1):
                item = others[index - 1]
                item_weight = int(item["remaining"])
                item_value = float(item["value"])
                for cap in range(remain_capacity + 1):
                    dp[index][cap] = dp[index - 1][cap]
                    if item_weight <= cap:
                        candidate_value = dp[index - 1][cap - item_weight] + item_value
                        if candidate_value > dp[index][cap]:
                            dp[index][cap] = candidate_value
                            take[index][cap] = True

            cap = remain_capacity
            for index in range(item_count, 0, -1):
                if not take[index][cap]:
                    continue
                item = others[index - 1]
                req_id = str(item["request_id"])
                selected_ids.add(req_id)
                selected_value += float(item["value"])
                cap -= int(item["remaining"])

        selected_count = len(selected_ids)
        # 方案实际占用预算 = decoding 固定预算 + 选中 prefill 的总 remaining。
        used_prefill_tokens = sum(
            prefill_remaining_by_id[req_id]
            for req_id in selected_ids
        )
        used_budget = decode_tokens + used_prefill_tokens
        if used_budget > token_budget:
            continue

        if (
            selected_count > best_count
            or (selected_count == best_count and selected_value > best_value)
            or (
                selected_count == best_count
                and selected_value == best_value
                and used_budget < best_budget
            )
        ):
            candidate_assigned = dict(decode_assigned)
            for req_id in selected_ids:
                candidate_assigned[req_id] = prefill_remaining_by_id[req_id]
            best_count = selected_count
            best_value = selected_value
            best_budget = used_budget
            best_assigned = candidate_assigned
            best_total_ms = anchor_slack * 1000.0

    # 若所有锚点都无法产出可行方案，则返回 None 让上层兜底。
    if best_assigned is None:
        return None
    return {
        "token_budget": best_budget,
        "assigned": best_assigned,
        "max_iter_time": best_total_ms,
    }
