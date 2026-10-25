"""
yolo26-detection / scripts/fetch_weights.py
==========================================
权重与资产的**镜像优先**下载器。

为什么必须存在（本机实测，2026-10-14）：
    `github.com` 在本机**完全不可达**（Node 与 Python 两侧均 10s+ 超时），
    而 ultralytics 的默认权重通道正是
        https://github.com/ultralytics/assets/releases/download/<tag>/<model>.pt
    → `YOLO("yolo26n.pt")` 会自动下载，但**在本机会失败**。
    实测可达的替代通道：
        https://hf-mirror.com/...            200 / 687ms   ← 首选
        https://gh-proxy.com/https://github.com/...  200 / 1010ms   ← 备用
        不可达：huggingface.co（直连）、ghfast.top、github.com

用法：
    python scripts/fetch_weights.py --list
    python scripts/fetch_weights.py yolo26n.pt yolo26s.pt
    python scripts/fetch_weights.py yolo26m.pt --out weights/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "weights"

# ultralytics 官方 assets 的 release tag（8.4.x 线）。若官方换 tag，改这里。
ASSETS_TAG = "v8.4.0"

# 通道按实测速度排序；每个通道一个 URL 模板。
#   {name} = 文件名（含扩展名）
CHANNELS: list[tuple[str, str]] = [
    ("hf-mirror", "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/{name}"),
    ("gh-proxy", f"https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/{ASSETS_TAG}/{{name}}"),
    ("github-direct", f"https://github.com/ultralytics/assets/releases/download/{ASSETS_TAG}/{{name}}"),
]

# 常用权重的已知大小（字节）——用于校验下载完整性。未知的写 None，只做非空校验。
KNOWN_SIZES = {
    "yolo26n.pt": 5544453,
    "yolo26s.pt": None,
    "yolo26m.pt": None,
    "yolo26l.pt": None,
    "yolo26x.pt": None,
    "yolo11n.pt": None,
    "yolov8n.pt": None,
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(1 << 20):
            h.update(blk)
    return h.hexdigest()


def download(name: str, out_dir: Path, *, force: bool = False, timeout: int = 180) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name

    if dest.exists() and not force:
        got = dest.stat().st_size
        exp = KNOWN_SIZES.get(name)
        if exp is None or got == exp:
            print(f"[skip] {dest} 已存在（{got:,} B）")
            return dest
        print(f"[warn] {dest} 大小 {got:,} B 与预期 {exp:,} B 不符，重新下载")

    for chan_name, tpl in CHANNELS:
        url = tpl.format(name=name)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            print(f"[try ] {chan_name}: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "yolo26-detection/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
                total = int(r.headers.get("content-length") or 0)
                got = 0
                while blk := r.read(1 << 20):
                    f.write(blk)
                    got += len(blk)
                    if total:
                        pct = got / total * 100
                        print(f"\r        {got:,}/{total:,} B ({pct:5.1f}%)", end="", flush=True)
            print()
            exp = KNOWN_SIZES.get(name)
            actual = tmp.stat().st_size
            if actual == 0:
                raise OSError("下载到 0 字节")
            if exp is not None and actual != exp:
                raise OSError(f"大小不符：得到 {actual:,} B，预期 {exp:,} B")
            tmp.replace(dest)
            print(f"[ok  ] {chan_name} -> {dest} ({actual:,} B, sha256={_sha256(dest)[:16]})")
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"\n[fail] {chan_name}: {type(e).__name__}: {e}")
            tmp.unlink(missing_ok=True)

    print(f"[ERR ] 所有通道均失败：{name}。请手工下载后放到 {out_dir}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="镜像优先的 YOLO 权重下载器")
    ap.add_argument("models", nargs="*", help="权重文件名，如 yolo26n.pt")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true", help="显示可用通道与已知权重")
    a = ap.parse_args()

    if a.list or not a.models:
        print("可用通道（按实测速度排序）：")
        for n, t in CHANNELS:
            print(f"  {n:15s} {t}")
        print(f"\n输出目录：{a.out}")
        print("已知权重：" + ", ".join(KNOWN_SIZES))
        return 0

    fails = []
    for m in a.models:
        if download(m, a.out, force=a.force) is None:
            fails.append(m)
    if fails:
        print(f"\n失败 {len(fails)} 个：{fails}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
