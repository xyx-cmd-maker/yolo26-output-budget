r"""
scripts/camera_demo.py —— 用训练好的模型**亲自看效果**（摄像头 / 单图 / 视频 / 文件夹）

═══════════════════════════════════════════════════════════════════════════
这是给你**自己看**用的，不是给论文用的
═══════════════════════════════════════════════════════════════════════════
论文的数字全部来自 `runs/budget/*.json` 里的离线评测（固定 conf/iou/max_det）。
本脚本的实时预览**只用于直观感受**，其输出**不得**写进任何对外文本。

═══════════════════════════════════════════════════════════════════════════
能不能实时跑？—— 先看实测
═══════════════════════════════════════════════════════════════════════════
· 模型 2.5 M 参数 / 6.1 GFLOPs @640，在 RTX 5060 Laptop 上单帧推理约 10–20 ms
· 所以 **`--imgsz 640` 下摄像头可以接近实时**；但本项目的教训是
  **不要与训练并发**（P21：并发曾导致 cv2 原生崩溃、R1 在 epoch 28 被杀）
  → 脚本启动前会检查有无训练在跑，除非 `--allow-concurrent`

═══════════════════════════════════════════════════════════════════════════
用法（都在项目根目录下执行）
═══════════════════════════════════════════════════════════════════════════
  $PY = 'G:\conda-envs\yolo26train\python.exe'

  ① 摄像头实时预览
     & $PY scripts\camera_demo.py --source 0
  ② 摄像头 + 换成 one-to-one 头（NMS-free，本文的第二个设置）
     & $PY scripts\camera_demo.py --source 0 --head o2o
  ③ 摄像头 + 保存视频
     & $PY scripts\camera_demo.py --source 0 --save runs/detect/webcam.mp4
  ④ 对单张图 / 整个文件夹 / 一段视频
     & $PY scripts\camera_demo.py --source path\to\img.jpg
     & $PY scripts\camera_demo.py --source path\to\folder --save-dir runs/detect/mydir
  ⑤ 只想看某个类（VisDrone 类别名见 --help）
     & $PY scripts\camera_demo.py --source 0 --classes car bus truck
  ⑥ 压低/提高灵敏度（看更多或更少的框）
     & $PY scripts\camera_demo.py --source 0 --conf 0.10      # 默认 0.25
  ⑦ 关掉窗口模式（无显示器/远程时用）
     & $PY scripts\camera_demo.py --source 0 --no-show --save out.mp4

按 `q` 或 `Esc` 退出摄像头窗口。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_W = ROOT / "runs" / "train" / "r1_yolo26n_visdrone_640__s42" / "weights" / "best.pt"


def guard_concurrent(allow: bool) -> bool:
    """启动前检查有无训练在跑（P21 的教训：并发曾导致 cv2 原生崩溃）。

    返回 True = 可以继续。
    """
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process python -ErrorAction SilentlyContinue).Count"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        n = int(out or 0)
    except Exception:
        return True          # 查不到就不拦，避免误伤
    # 允许 1 个（本脚本自身所在解释器可能是被计数的那一个）
    if n > 1 and not allow:
        print(f"⚠️ 检测到 {n} 个 python 进程 —— 可能有训练在跑。")
        print("   P21 的教训：**并发 python 曾导致 OpenCV 原生崩溃，把 R1 在 epoch 28 杀掉**。")
        print("   训练期间做可视化预览会与它抢 GPU/内存。")
        print("   · 若确认无碍，加 `--allow-concurrent` 强制继续")
        print("   · 或先等训练结束（`python scripts/check_progress.py` 看进度）")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="用训练好的模型亲自看效果（摄像头 / 图片 / 视频 / 文件夹）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="VisDrone 10 类：pedestrian, people, bicycle, car, van, truck, "
               "tricycle, awning-tricycle, bus, motor")
    ap.add_argument("--weights", type=Path, default=DEFAULT_W,
                    help="权重路径（默认 seed 42 的 best.pt）")
    ap.add_argument("--source", default="0",
                    help="'0'=默认摄像头；也可是 图片/文件夹/视频 路径，或 rtsp:// 流地址")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值（越低框越多）")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    ap.add_argument("--max-det", type=int, default=300, help="每图最多输出框数")
    ap.add_argument("--head", choices=["o2m", "o2o"], default="o2m",
                    help="用哪个头：o2m=一对多（需 NMS）；o2o=一对一（NMS-free）")
    ap.add_argument("--device", default="0", help="'0'=GPU；'cpu'=CPU")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="只显示这些类（给出类别**名字**，如 car bus）")
    ap.add_argument("--save", type=Path, default=None,
                    help="保存带框的输出（摄像头/视频 → .mp4；单图 → 图片）")
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="批量处理时把带框结果写到该目录")
    ap.add_argument("--no-show", action="store_true",
                    help="不开窗口（远程/无显示器时用）")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="跳过「有训练在跑」的检查（自负后果，见 P21）")
    a = ap.parse_args()

    if not a.weights.exists():
        print(f"❌ 找不到权重：{a.weights}")
        print("   可用权重：")
        for p in sorted((ROOT / "runs" / "train").glob("*/weights/best.pt")):
            print(f"     {p.relative_to(ROOT)}")
        return 1

    if not guard_concurrent(a.allow_concurrent):
        return 2

    from ultralytics import YOLO

    print(f"加载权重：{a.weights.relative_to(ROOT)}")
    model = YOLO(str(a.weights))

    # ── 头选择：本项目的方法学核心之一 ────────────────────────
    # o2o = one-to-one（NMS-free）；o2m = one-to-many（需 NMS）
    # Ultralytics 的 `nms` 训练参数控制推理时用哪个头：
    #   nms=False → 用 one-to-one（端到端）
    #   nms=True  → 用 one-to-many + NMS（默认）
    use_nms = (a.head == "o2m")
    print(f"检测头：{a.head}（{'one-to-many + NMS' if use_nms else 'one-to-one, NMS-free'}）")

    # ── 类别名 → id ──────────────────────────────────────────
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    cls_ids = None
    if a.classes:
        inv = {str(v).lower(): int(k) for k, v in names.items()}
        cls_ids = []
        for c in a.classes:
            if c.lower() in inv:
                cls_ids.append(inv[c.lower()])
            elif c.isdigit() and int(c) in inv:
                cls_ids.append(int(c))
            else:
                print(f"⚠️ 未知类别 '{c}'；可用：{', '.join(inv)}")
        if not cls_ids:
            return 1
        print(f"只显示类别：{cls_ids} → {[names[i] for i in cls_ids]}")

    # ── 源 ───────────────────────────────────────────────────
    src = int(a.source) if a.source.isdigit() else a.source
    if isinstance(src, int):
        print(f"打开摄像头 index={src} …（按 q 或 Esc 退出）")

    kwargs = dict(
        source=src,
        imgsz=a.imgsz,
        conf=a.conf,
        iou=a.iou,
        max_det=a.max_det,
        device=a.device,
        classes=cls_ids,
        stream=True,          # ★ 摄像头/视频必须流式，否则会一次性缓冲
        show=not a.no_show,
        verbose=False,
    )
    # 头选择：通过 `nms` 传参（o2o 时置 False）
    try:
        results = model.predict(nms=use_nms, **kwargs)
    except TypeError:
        # 该版本不接受 nms= 参数时退回默认（并明确告知，不静默）
        print("⚠️ 该 ultralytics 版本不接受 `nms=` 参数 → 退回默认头（one-to-many + NMS）")
        results = model.predict(**kwargs)

    n_frame = 0
    t0 = time.perf_counter()
    writer = None
    out_path = None
    if a.save:
        out_path = a.save if a.save.is_absolute() else ROOT / a.save
        out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for r in results:
            n_frame += 1
            n_box = 0 if r.boxes is None else len(r.boxes)
            if n_frame % 10 == 1 or isinstance(src, (str, Path)):
                fps = n_frame / max(time.perf_counter() - t0, 1e-6)
                print(f"  帧 {n_frame}: {n_box} 个框   ({fps:.1f} fps 累计平均)", end="\r")

            # 保存
            if a.save_dir:
                d = a.save_dir if a.save_dir.is_absolute() else ROOT / a.save_dir
                d.mkdir(parents=True, exist_ok=True)
                r.save(filename=str(d / f"{Path(str(r.path)).stem}_pred.jpg"))
            elif out_path and isinstance(src, int):
                # 摄像头 → 逐帧写视频
                frame = r.plot()
                import cv2
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                writer.write(frame)
    except KeyboardInterrupt:
        print("\n（被用户中断）")
    finally:
        if writer is not None:
            writer.release()
            print(f"\n✅ 视频已保存：{out_path}")
        if a.save_dir:
            d = a.save_dir if a.save_dir.is_absolute() else ROOT / a.save_dir
            print(f"\n✅ 带框结果已写入：{d}")

    dt = time.perf_counter() - t0
    print(f"\n共处理 {n_frame} 帧，用时 {dt:.1f} 秒"
          + (f"，平均 {n_frame / dt:.1f} fps" if dt > 0 else ""))
    print("\n⚠️ 提醒：本脚本的输出仅供你**直观感受**；论文数字来自离线评测"
          "（`runs/budget/*.json`，固定 conf=0.001 / iou=0.7 / max_det=300）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
