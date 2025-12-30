import json
from collections import Counter

def main(path):
    c_wait = Counter()
    c_run = Counter()
    total = 0

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
            if nw is not None:
                c_wait[nw] += 1
            if nr is not None:
                c_run[nr] += 1
            total += 1

    def print_stats(name, counter):
        print(f"\n{name} (total lines={total}):")
        for k, v in sorted(counter.items()):
            pct = v / total * 100 if total else 0
            print(f"  {k}: {v} ({pct:.2f}%)")

    print_stats("num_waiting", c_wait)
    print_stats("num_running", c_run)

if __name__ == "__main__":
    # 替换为你的文件路径
    main("vllm/vllm/v1/core/sched/rl/profiling/2025-12-30/profiling_2025-12-30 23:40:59.jsonl")
