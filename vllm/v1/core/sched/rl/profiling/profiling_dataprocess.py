import json
from collections import Counter
from typing import Optional

def main(path, lower_ms: Optional[float] = None, upper_ms: Optional[float] = None):
    c_wait = Counter()
    c_run = Counter()
    total = 0
    run_ms_sum = 0.0
    run_ms_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            nw = obj.get("num_waiting")
            nr = obj.get("num_running")
            run_ms = obj.get("model_run_ms")
            if nw is not None:
                c_wait[nw] += 1
            if nr is not None:
                c_run[nr] += 1
            if (
                run_ms is not None
                and lower_ms is not None
                and upper_ms is not None
            ):
                try:
                    run_ms_val = float(run_ms)
                except (TypeError, ValueError):
                    pass
                else:
                    if lower_ms <= run_ms_val <= upper_ms:
                        run_ms_sum += run_ms_val
                        run_ms_count += 1
            total += 1

    def print_stats(name, counter):
        print(f"\n{name} (total lines={total}):")
        for k, v in sorted(counter.items()):
            pct = v / total * 100 if total else 0
            print(f"  {k}: {v} ({pct:.2f}%)")

    print_stats("num_waiting", c_wait)
    print_stats("num_running", c_run)
    if lower_ms is not None and upper_ms is not None:
        if run_ms_count:
            avg = run_ms_sum / run_ms_count
            print(
                f"model_run_ms between [{lower_ms}, {upper_ms}] (count={run_ms_count}): avg={avg:.2f}"
            )
        else:
            print(
                f"model_run_ms between [{lower_ms}, {upper_ms}] (count=0): no entries in range"
            )

if __name__ == "__main__":
    # 替换为你的文件路径
    main(
        "vllm/vllm/v1/core/sched/rl/profiling/2026-01-14/profiling_2026-01-14 00:12:02.jsonl",
        lower_ms=195,
        upper_ms=270,
    )
    main(
        "vllm/vllm/v1/core/sched/rl/profiling/2026-01-12/profiling_2026-01-12 00:42:48.jsonl",
        lower_ms=10,
        upper_ms=20,
    )
    main(
        "vllm/vllm/v1/core/sched/rl/profiling/2026-01-12/profiling_2026-01-12 23:01:36.jsonl",
        lower_ms=20,
        upper_ms=50,
    )
