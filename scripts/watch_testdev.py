"""
scripts/watch_testdev.py —— test-dev 评测进度监控（一次性查询）

为什么需要：test-dev 由 **Windows 计划任务**运行（为脱离本环境约 5 分钟的后台作业上限），
因此**不会出现在 DSH 的 job_list 里** —— 必须有一个独立入口来查它。

用法：
    python scripts/watch_testdev.py            # 查询一次
    python scripts/watch_testdev.py --watch    # 每 60 秒刷新（Ctrl-C 退出）
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "runs" / "budget"
LOG = ROOT / "logs" / "testdev_run.log"


def ps(cmd: str) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"__ERR__ {e}"


def decode_log(p: Path) -> str:
    """日志是 GBK 双重编码（Tee-Object 在 GBK 控制台下写的），需修复。"""
    raw = p.read_bytes()
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    else:
        return ""
    # 尝试反转双重编码
    try:
        fixed = txt.encode("gbk", errors="ignore").decode("utf-8", errors="ignore")
        if len(fixed) > len(txt) * 0.5:
            return fixed
    except Exception:
        pass
    return txt


def report() -> None:
    print("=" * 96)
    print(f"test-dev 评测监控    {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 96)

    # ① 计划任务
    q = ps("schtasks /Query /TN DSH_testdev /FO LIST /V")
    st = re.search(r"Status:\s*(\S+)", q) or re.search(r"状态:\s*(\S+)", q)
    lr = re.search(r"Last Result:\s*(-?\d+)", q) or re.search(r"上次结果:\s*(-?\d+)", q)
    print(f"  计划任务状态 : {st.group(1) if st else '（查不到）'}")
    # ★ 计划任务的 "Last Result" 有两套码：0x41301/267009 = SCHED_S_TASK_RUNNING，
    #   0x800710E0/-2147020576 = 同样表示"任务正在运行"（调度器内部码）。
    #   我第一版把后者当成了异常 —— 判据应以 **Status=Running + 产物是否新增** 为准。
    code = lr.group(1) if lr else "?"
    meaning = {"267009": "运行中", "0": "已成功结束",
               "-2147020576": "运行中（调度器码）"}.get(code, "见 schtasks 文档")
    print(f"  上次结果码   : {code}  → {meaning}")

    # ② 进程
    procs = ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    running = []
    try:
        arr = json.loads(procs)
        arr = arr if isinstance(arr, list) else [arr]
        for p in arr:
            cl = str(p.get("CommandLine") or "")
            if "testdev" in cl or "budget_scan" in cl:
                tag = "编排" if "budget_scan_chunked" in cl else "子进程"
                head = re.search(r"--head (\w+)", cl)
                running.append((p.get("ProcessId"), tag,
                                f"head={head.group(1)}" if head else ""))
    except Exception:
        pass
    if running:
        for pid, tag, extra in running:
            print(f"  进程         : pid={pid}  [{tag}]  {extra}")
    else:
        print("  进程         : ❌ 无（已结束或被终止）")

    # ③ 分块产物
    segs = sorted(BUDGET.glob("r1_testdev_seg*_test_*.json"))
    by_head: dict[str, int] = {}
    for s in segs:
        m = re.search(r"_seg\d+_test_(\w+)\.json", s.name)
        if m:
            by_head[m.group(1)] = by_head.get(m.group(1), 0) + 1
    print(f"  分块产物     : {len(segs)} 个  {by_head}")
    exp_chunks = 11  # ceil(1610/150)
    for h in ("o2m", "o2o"):
        n = by_head.get(h, 0)
        bar = "█" * n + "░" * (exp_chunks - n)
        print(f"     {h}: [{bar}] {n}/{exp_chunks}")
    if segs:
        newest = max(segs, key=lambda p: p.stat().st_mtime)
        age = (datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime)).total_seconds()
        print(f"  最新产物     : {newest.name}  （{age:.0f} 秒前）"
              f"{'  ⚠️ 超过 5 分钟未更新，可能已停' if age > 300 else ''}")

    # ④ 合并产物
    for h in ("o2m", "o2o"):
        f = BUDGET / f"r1_testdev_test_{h}.json"
        print(f"  合并 {h}      : {'✅ ' + f.name if f.exists() else '（尚未合并）'}")

    # ⑤ 日志尾部
    if LOG.exists():
        lines = [l for l in decode_log(LOG).split("\n") if l.strip()]
        print("  日志尾部     :")
        for l in lines[-5:]:
            print(f"     {l.strip()[:104]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="test-dev 评测进度监控")
    ap.add_argument("--watch", action="store_true", help="每 60 秒刷新")
    a = ap.parse_args()
    if not a.watch:
        report()
        return 0
    try:
        while True:
            report()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n  （已退出）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
