"""
scripts/download_visdrone.py
============================
VisDrone-DET 的**可断点续传**下载器。

为什么不用 `urllib` 一把梭（本项目实测）：
    ① 直连 `github.com` 实测仅 **0.048 MB/s**，`gh-proxy.com` **0.101 MB/s**（快 2×）；
    ② 全程 1.8 GB，单次请求必然中途断；
    ③ 之前的简易下载被误清进程后留下 0 字节文件，**没有续传就只能从头再来**。
    → 必须：**探速选通道 → Range 续传 → 大小校验 → 失败重试**。

数据源（实测可达，2026-10-14）：
    https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{train,val,test-dev}.zip
    train 1478.1 MB / val 77.9 MB / test-dev 296.8 MB

许可提醒：VisDrone 为 **CC BY-NC-SA 3.0，仅学术用途**（见 AGENTS-PROJECT.md §3）。
         下载后**不得再分发**；论文中须按 CC BY-NC-SA 署名。

用法：
    python scripts/download_visdrone.py --parts val train test-dev
    python scripts/download_visdrone.py --parts val            # 只要小的
    python scripts/download_visdrone.py --speed-test           # 只测速不下载
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "visdrone"

GH = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
PROXY = "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/"

PARTS = {
    # (zip 名, 官方精确字节数)。★ 用**精确值**而非区间——区间上界设小了会导致
    # 明明下完却被判为"不完整"并无限重试（本机实测踩到：train 上界写成 1_549_000_000，
    # 实际 1_549_875_511，于是 .part 已达 1478.1MB 仍被判失败）。
    "train": ("VisDrone2019-DET-train.zip", 1_549_875_511),
    "val": ("VisDrone2019-DET-val.zip", 81_638_851),
    "test-dev": ("VisDrone2019-DET-test-dev.zip", 311_251_787),
}

CHANNELS = [("gh-proxy", PROXY), ("direct", GH)]

UA = {"User-Agent": "Mozilla/5.0 (yolo26-detection dataset fetcher)"}


def _head(url: str, timeout: int = 25) -> tuple[int, int]:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, int(r.headers.get("content-length") or 0)


def speed_test(label: str, url: str, budget_sec: float = 12.0) -> float:
    """在 budget 秒内尽量多读，返回 MB/s。失败返回 -1。"""
    try:
        req = urllib.request.Request(url, headers=UA)
        t0 = time.time()
        got = 0
        with urllib.request.urlopen(req, timeout=30) as r:
            while time.time() - t0 < budget_sec:
                blk = r.read(1 << 16)
                if not blk:
                    break
                got += len(blk)
        dt = time.time() - t0
        return got / dt / 1048576 if dt > 0 else -1
    except (urllib.error.URLError, OSError, TimeoutError):
        return -1


def pick_channel(zip_name: str) -> tuple[str, str, float]:
    """按实测速度挑最快的通道。"""
    best = ("", "", -1.0)
    for name, base in CHANNELS:
        sp = speed_test(f"{name}/{zip_name}", base + zip_name, budget_sec=10)
        print(f"    通道 {name:9s} 实测 {sp:6.3f} MB/s" if sp > 0 else f"    通道 {name:9s} 不可用")
        if sp > best[2]:
            best = (name, base + zip_name, sp)
    return best


def download(url: str, dest: Path, expect_bytes: int, *, chunk: int = 1 << 20) -> bool:
    """带 Range 续传的下载。返回是否完整。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        # 已完整？（大小落在允许区间）
        if have == expect_bytes:
            part.replace(dest)
            print(f"    ✅ 完成：{dest.name} = {have/1048576:.1f} MB")
            return True

        headers = dict(UA)
        mode = "wb"
        if have > 0:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"

        try:
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            last = t0
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("content-length") or 0) + have
                with part.open(mode) as f:
                    while True:
                        blk = r.read(chunk)
                        if not blk:
                            break
                        f.write(blk)
                        have += len(blk)
                        now = time.time()
                        if now - last >= 15:
                            spd = (have / max(now - t0, 1e-9)) / 1048576 if mode == "wb" else 0
                            pct = f"{have/total*100:.1f}%" if total else "?"
                            print(
                                f"      {have/1048576:7.1f} MB ({pct}) 用时 {now-t0:5.0f}s 本次 {spd:5.2f} MB/s",
                                flush=True,
                            )
                            last = now
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            got = part.stat().st_size if part.exists() else 0
            print(f"    ⚠️ 第 {attempt} 次中断（{type(e).__name__}），已存 {got/1048576:.1f} MB，10s 后续传…", flush=True)
            time.sleep(10)

    got = part.stat().st_size if part.exists() else 0
    print(f"    ❌ {dest.name} 重试耗尽，停在 {got/1048576:.1f} MB")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="VisDrone-DET 断点续传下载器")
    ap.add_argument("--parts", nargs="+", default=["val"], choices=list(PARTS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--speed-test", action="store_true", help="只测速不下载")
    a = ap.parse_args()

    report: dict = {"out": str(a.out), "parts": {}}
    ok_all = True

    for key in a.parts:
        zip_name, expect_bytes = PARTS[key]
        print(f"\n=== {key} ({zip_name}, 官方 {expect_bytes/1048576:.1f} MB) ===")
        try:
            st, cl = _head(GH + zip_name)
            print(f"  HEAD 直连 {st}，content-length = {cl/1048576:.1f} MB")
        except Exception as e:  # noqa: BLE001
            print(f"  HEAD 直连失败：{type(e).__name__}")

        if a.speed_test:
            name, url, sp = pick_channel(zip_name)
            report["parts"][key] = {"channel": name, "MBps": round(sp, 3)}
            continue

        name, url, sp = pick_channel(zip_name)
        if sp <= 0:
            print("  ❌ 所有通道不可用")
            report["parts"][key] = {"status": "no_channel"}
            ok_all = False
            continue
        eta = (expect_bytes / 1048576) / sp / 60 if sp > 0 else -1
        print(f"  → 选 {name}（{sp:.3f} MB/s），预计 {eta:.0f} 分钟")
        ok = download(url, a.out / zip_name, expect_bytes)
        report["parts"][key] = {"channel": name, "MBps": round(sp, 3), "ok": ok}
        ok_all &= ok

    rp = a.out / "download_report.json"
    a.out.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告 -> {rp}")
    print("许可提醒：VisDrone 为 CC BY-NC-SA 3.0（仅学术用途），**不得再分发**。")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
