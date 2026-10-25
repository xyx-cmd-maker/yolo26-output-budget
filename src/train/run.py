"""
yolo26-detection / src/train/run.py
===================================
唯一的训练入口。所有实验都从这里发起，不允许直接 `yolo train` —— 原因是
只有走这里才会自动落盘：种子、环境指纹、代码哈希、结果记录。

用法：
    # 冒烟（只验证管线；产物档位=smoke，永远不会进论文）
    python -m src.train.run --exp e00_smoke --data data/fixtures/synth_det/fixture.yaml \\
        --model weights/yolo26n.pt --tier smoke --epochs 3 --imgsz 64

    # 真实实验，3 个种子串行
    python -m src.train.run --exp e02_yolo26n_baseline --data <dataset> --seeds 0 1 2 --tier paper

    # 从实验配置读参数
    python -m src.train.run --expcfg src/configs/experiments/e02_yolo26n_baseline.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# 允许 `python src/train/run.py` 直接跑
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.console import setup_console  # noqa: E402

setup_console()  # ★ 必须在任何 print 之前：否则 Windows GBK 控制台会因 emoji 崩掉

from src.utils.repro import dump_fingerprint, env_fingerprint, seed_everything  # noqa: E402
from src.utils.results_io import PAPER_TIER, TIERS, ResultRecord, record_path, write_record  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# train() 的显式关键字集合。extra_train 里出现这些键会与显式传参撞名，
# Python 抛的是难懂的 "got multiple values for keyword argument"，所以前置拦下来。
_EXPLICIT_TRAIN_KEYS = frozenset(
    {
        "data", "epochs", "imgsz", "batch", "seed", "device", "optimizer",
        "project", "name", "exist_ok", "plots", "val", "deterministic",
    }
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(1 << 20):
            h.update(blk)
    return h.hexdigest()


def load_exp_cfg(p: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
    cfg.pop("_comment", None)
    return cfg


def resolve_data(name_or_path: str) -> str:
    """把数据集短名解析成 yaml 路径；也接受直接给的路径或 ultralytics 内置名。"""
    if name_or_path.endswith((".yaml", ".yml")):
        p = Path(name_or_path)
        return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())
    cand = PROJECT_ROOT / "src" / "configs" / "datasets" / f"{name_or_path}.yaml"
    if cand.exists():
        return str(cand)
    # 交给 ultralytics 自己找内置数据集（coco8.yaml 等）
    return name_or_path


def dataset_id(data_yaml: str) -> str:
    """结果记录里的 dataset 标识（结果文件名的组成部分）。

    优先用 yaml 里的 `name:` 字段，否则退回文件名 stem。
    为什么需要：`.../fixture.yaml` 的 stem 是 "fixture"，与真实数据集短名可能相撞，
    一旦相撞两条不同来源的实验会被写进同一个记录名 —— 数字来源就不可信了。
    """
    p = Path(data_yaml)
    if p.exists():
        try:
            import yaml as _yaml

            cfg = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(cfg, dict) and cfg.get("name"):
                return str(cfg["name"])
        except Exception:
            pass
    return p.stem


def count_split_images(data_yaml: str, split: str = "val") -> int | None:
    """直接从数据集 yaml 数出某个 split 的图像数。

    为什么不从 `metrics` 取：实测 ultralytics 8.4.177 的 `Metrics` 对象
    **不暴露 dataloader / seen**，`getattr(metrics,'seen',0)` 恒为 0，
    于是 `n_images` 永远写不进去。数文件是唯一可靠的来源。
    """
    p = Path(data_yaml)
    if not p.exists():
        return None
    try:
        import yaml as _yaml

        cfg = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw = cfg.get(split)
        if not raw:
            return None
        root = cfg.get("path")
        base = Path(root) if root else p.parent
        if not base.is_absolute():
            base = (p.parent / base).resolve()
        sp = Path(raw)
        if not sp.is_absolute():
            sp = (base / raw).resolve()
        if not sp.exists():
            return None
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sum(1 for f in sp.rglob("*") if f.suffix.lower() in exts)
    except Exception:
        return None


def train_one_seed(
    *,
    exp_id: str,
    data: str,
    model: str,
    seed: int,
    tier: str,
    epochs: int,
    imgsz: int,
    batch: int | str,
    device: str,
    optimizer: str,
    extra_train: dict,
    extra_val: dict,
) -> ResultRecord:
    from ultralytics import YOLO

    # ★ 修 F1 的回归：曾经把 `deterministic` 同时作为显式关键字**和** extra_train 的键
    #   传给 train()，Python 会抛
    #       TypeError: Model.train() got multiple values for keyword argument 'deterministic'
    #   触发条件是 expcfg 的 train_args 里写了 deterministic —— 不写就看不出来，
    #   所以最初的冒烟测试没抓到。正确做法：读出来之后**必须从 extra_train 里剔除**。
    deterministic = bool(extra_train.pop("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    # ★★ 兜底：周期性 checkpoint 必须在**代码层**强制，不能只依赖 expcfg。
    #
    # 为什么（P31，2026-10-15 实测）：
    #   我在 23:04:10 把 `save_period: 10` 写进 r1 的 expcfg，而 **R1 进程在 ~23:02 已启动**
    #   → 它加载的是旧配置 → 从 checkpoint 读回的 `train_args.save_period` 是 **-1**（默认禁用）
    #   → **epoch10.pt / epoch20.pt 从未产生**，我此前"最坏损失降到 10 个 epoch"的说法**不成立**。
    #
    # 教训：**"改了配置文件" ≠ "运行中的进程用了新配置"**。
    # 因此把保护放在代码里，并对**所有**实验生效（expcfg 若显式给了值则不覆盖）。
    extra_train.setdefault("save_period", 10)

    run_name = f"{exp_id}__s{seed}"
    run_dir = PROJECT_ROOT / "runs" / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_fingerprint(run_dir / f"{run_name}__env.json")

    print(f"\n{'=' * 70}\n[{exp_id}] seed={seed} tier={tier} model={model}\n{'=' * 70}")

    yolo = YOLO(model)
    t0 = time.time()
    yolo.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=deterministic,
        device=device,
        optimizer=optimizer,
        project=str(run_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        **extra_train,
    )
    train_seconds = time.time() - t0

    # ---- 用 best.pt 重新在 val 上评测，拿到我们自己的指标（不信任 trainer 的打印）----
    best = run_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"训练未产出 best.pt：{best}")
    w_hash = _sha256(best)

    t1 = time.time()
    metrics = yolo.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        split="val",
        project=str(PROJECT_ROOT / "runs" / "val"),
        name=run_name,
        exist_ok=True,
        **extra_val,
    )
    eval_seconds = time.time() - t1

    md = metrics.results_dict  # dict，键形如 'metrics/mAP50(B)'
    clean = {
        k.replace("metrics/", "").replace("(B)", ""): float(v)
        for k, v in md.items()
        if isinstance(v, (int, float))
    }

    # ★ 修 B8：把评测协议与诊断信息显式落盘。
    #   实测（8.4.177）：`Metrics` **没有** `seen` 属性，所以原先的
    #   `getattr(metrics,'seen',0)` 恒为 0 → n_images 一直是 None。
    #   正确来源是验证器的 dataloader；同时 `metrics.box.maps` 给按类 AP。
    n_images = count_split_images(data, "val")

    per_class_ap: dict[str, float] = {}
    try:
        box = metrics.box
        names = getattr(metrics, "names", {}) or {}
        for idx, ap in zip(box.ap_class_index, box.maps):
            nm = names.get(int(idx), str(int(idx))) if isinstance(names, dict) else str(int(idx))
            per_class_ap[str(nm)] = float(ap)
    except Exception:
        per_class_ap = {}

    n_boxes = None
    try:
        n_boxes = int(sum(metrics.nt_per_class))
    except Exception:
        n_boxes = None

    # GFLOPs 与峰值显存（AGENTS §0.4 要求报资源）
    try:
        from ultralytics.utils.torch_utils import get_flops

        flops = float(get_flops(yolo.model, imgsz=imgsz))
    except Exception:
        flops = None
    peak_mem_gb = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_mem_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except ImportError:
        pass

    speed = getattr(metrics, "speed", {}) or {}

    # 评测协议参数若没显式给，落盘里就是 None —— 明确提醒，别让读者以为"没设阈值"。
    if not any(k in extra_val for k in ("conf", "iou", "max_det")):
        print(
            "[注意] val_args 未显式指定 conf/iou/max_det。记录里这三个字段将为 null，"
            "别人无法复现你用的是哪个 NMS 阈值。正式实验请写进 expcfg 的 val_args。"
        )

    return ResultRecord(
        exp_id=exp_id,
        dataset=dataset_id(data),
        tier=tier,
        seed=seed,
        metrics=clean,
        n_images=n_images,
        n_instances=n_boxes,
        imgsz=imgsz,
        conf=extra_val.get("conf"),
        iou=extra_val.get("iou"),
        max_det=extra_val.get("max_det"),
        epochs=epochs,
        batch=batch if isinstance(batch, int) else None,
        optimizer=optimizer,
        split="val",
        weights=str(best.relative_to(PROJECT_ROOT)),
        weights_sha256=w_hash,
        eval_seconds=round(eval_seconds, 3),
        fingerprint=env_fingerprint(),
        extra={
            "train_seconds": round(train_seconds, 1),
            "speed_ms_per_image": speed,
            "model_cfg": model,
            "n_params": sum(p.numel() for p in yolo.model.parameters()),
            "gflops": flops,
            "peak_gpu_mem_gb": peak_mem_gb,
            "deterministic": deterministic,
            # 按类 AP：长尾/小类分析必须靠它，不能只留总 mAP
            "per_class_ap": per_class_ap,
            "per_class_ap_note": "键为类别名，值为该类 AP（与主指标同口径的聚合值由 ultralytics 给出）",
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="yolo26-detection 统一训练入口")
    ap.add_argument("--exp", help="实验 id，例如 e02_yolo26n_baseline")
    ap.add_argument("--expcfg", type=Path, help="实验配置 yaml；给了就覆盖 --exp 的其他参数")
    ap.add_argument("--data", default="coco8", help="数据集短名 / yaml 路径")
    # ★★ `--model` 默认值改为 None（2026-10-16 修，P46）。
    #    原默认 `"yolo26n.pt"` 与 `cfg.get("model", a.model)` 组合，使 **CLI 永远被忽略** ——
    #    因为 `cfg.get` 只在键**缺失**时才回退，而 expcfg 里 `model:` 恒存在。
    #    实测：`--model runs\train\_warmstart_s44_ep38.pt` 被静默丢弃，
    #    args.yaml 记成 `model=weights/yolo26n.pt`，即**从 COCO 原始权重从头训**，
    #    100 epoch 的算力会被浪费在与预期不同的起点上。
    #    **与 P34（`--seeds` 被 expcfg 覆盖）同族，是同一处反了三次的优先级。**
    ap.add_argument("--model", default=None,
                    help="模型权重或 yaml；★ 显式给定时优先于 expcfg 的 `model:`")
    ap.add_argument("--tier", default="smoke", choices=TIERS, help="产物档位；只有 paper 能进论文")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="★ 显式给定时**优先于** expcfg 的 `seeds:`（默认 None = 回退到 expcfg）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的同种子记录")
    # ★★ `--warmstart`（2026-10-16 加，P46）：以**纯权重**检查点为起点开新 run。
    #    用于"被中断的训练"继续 —— 见 L341 附近的长注释。
    ap.add_argument("--warmstart", type=Path, default=None,
                    help="以该纯权重 .pt 为初始权重（非 resume；epoch 从 0 重数）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的内容")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="★ 绕过训练前置自检（P21：并发会导致 cv2 原生崩溃、"
                         "训练成果全丢）。仅在明确知道后果时使用。")
    a = ap.parse_args()

    cfg: dict = {}
    if a.expcfg:
        cfg = load_exp_cfg(a.expcfg)
        print(f"已读实验配置 {a.expcfg}：\n{json.dumps(cfg, indent=2, ensure_ascii=False)}")

    exp_id = cfg.get("exp_id") or a.exp
    if not exp_id:
        ap.error("必须给 --exp 或 --expcfg（含 exp_id）")

    data = resolve_data(cfg.get("data", a.data))
    # ★★ 与 `seeds` 同一判据（P34/P46）：**显式命令行参数优先于 expcfg**。
    #    `a.model` 默认已改为 None，故能区分"用户没给"与"用户给了"。
    #    原写法 `cfg.get("model", a.model)` 使 CLI **永远被忽略**（expcfg 里 `model:` 恒存在）→
    #    实测 `--model runs\train\_warmstart_s44_ep38.pt` 被静默丢弃，
    #    args.yaml 记成 `model=weights/yolo26n.pt`（从 COCO 原始权重从头训），
    #    100 epoch 的算力会浪费在与预期不同的起点上。
    model = a.model if a.model is not None else cfg.get("model", "yolo26n.pt")
    # ★★ `--warmstart`：以指定检查点（**纯权重**）为起点开新 run（2026-10-16，P46）
    #
    # 为什么需要：`--allow-concurrent` **只关掉进程检查**，输出目录那道护栏仍生效
    # （`preflight.py` L151：目录已存在且非空即拒绝）；而 Ultralytics 续训要求
    # `exist_ok` 指向**已有**目录 —— 两者冲突，于是被中断的训练无法续训。
    # 处置：旧目录移开保留证据 → `rebuild_weights.py` 把 `last.pt` 剥成纯权重
    # （去掉 trainer 状态，避免触发 resume 特例）→ 新 run 以它为起点。
    # 代价：epoch 从 0 重数（总训练量 = 已训 + 100）。
    warmstart = Path(a.warmstart) if getattr(a, "warmstart", None) else None
    if warmstart is not None:
        if not warmstart.exists():
            ap.error(f"--warmstart 指向的检查点不存在：{warmstart}")
        model = str(warmstart)
        print(f"[warmstart] 初始权重 = {warmstart}（纯权重，非 resume）")
    tier = cfg.get("tier", a.tier)
    # ★★ 显式命令行参数**优先于** expcfg（2026-10-15 修，P35）。
    #
    # 原写法 `cfg.get("seeds", a.seeds)` 是**反的**：只要 expcfg 里有 `seeds:`
    # （本项目 r1 写的是 `seeds: [42]`），命令行 `--seeds 43` 就被**静默忽略**
    # → 程序去重跑 seed 42 → **若没有 preflight 拦截，R1 的 100-epoch 成果会被直接覆盖**。
    # 实测经过：`--seeds 43` 被 preflight 以
    #   "输出目录已存在且非空：runs\train\r1_yolo26n_visdrone_640__s42"
    # 拒绝 —— 报的是 **s42**，暴露出它根本没在跑 43。
    #
    # 判据：`--seeds` 默认值由 `[0]` 改为 `None`，从而能区分"用户没给"与"用户给了 0"。
    seeds = a.seeds if a.seeds is not None else cfg.get("seeds", [0])
    epochs = int(cfg.get("epochs", a.epochs))
    imgsz = int(cfg.get("imgsz", a.imgsz))
    batch = cfg.get("batch", a.batch)
    device = cfg.get("device", a.device)
    optimizer = cfg.get("optimizer", a.optimizer)
    extra_train = dict(cfg.get("train_args", {}))
    extra_val = dict(cfg.get("val_args", {}))
    ds_id = dataset_id(data)

    # 顶层 `deterministic` 是 run.py 的显式传参，不放进 train_args。
    # ★ 注意顺序：先校验 train_args（此时不含 deterministic），再把它注进去，
    #   否则自己的注入会被自己的校验拦下。
    top_deterministic = bool(cfg["deterministic"]) if "deterministic" in cfg else None

    # ★★ 训练前置自检（P21 的机器强制，内置而非外部脚本）
    #   实测代价：R1 曾因并发任务在 epoch 28 崩溃（cv2.pyd 访问违规），
    #   27 个 epoch / 约 1.9 小时全丢且无 checkpoint。
    #   dry-run 不做自检（它不真的训练）。
    if not a.dry_run:
        from src.utils.preflight import check_before_train, report_system

        run_dirs = [
            str(PROJECT_ROOT / "runs" / "train" / f"{exp_id}__s{s}") for s in seeds
        ]
        pf = check_before_train(
            out_dir=run_dirs[0] if run_dirs else None,
            allow_concurrent=bool(getattr(a, "allow_concurrent", False)),
        )
        print(pf.report())
        if not pf.ok:
            return 2
        print("  · 系统快照：" + report_system())

    # ★ 前置校验：拦截 train_args 与显式传参撞名
    clash = sorted(set(extra_train) & _EXPLICIT_TRAIN_KEYS)
    if clash:
        print("\n[中止] train_args 里出现了会被显式传参覆盖的键：")
        for c in clash:
            print(f"   - {c}")
        print("   这些键请写到实验配置的**顶层**（与 epochs/imgsz/batch 同级），")
        print("   不要放进 train_args。否则 ultralytics 会抛 'got multiple values for keyword argument'。")
        return 2

    if top_deterministic is not None:
        extra_train["deterministic"] = top_deterministic

    plan = {
        "exp_id": exp_id,
        "dataset_id": ds_id,
        "data": data,
        "model": model,
        "tier": tier,
        "seeds": seeds,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "optimizer": optimizer,
        "deterministic": bool(extra_train.get("deterministic", True)),
        "train_args": extra_train,
        "val_args": extra_val,
    }
    print("执行计划：\n" + json.dumps(plan, indent=2, ensure_ascii=False, default=str))

    if tier != PAPER_TIER:
        print(f"\n[注意] tier={tier}：产物档位非 paper，按 AGENTS.md §2 不得进入任何对外文本。")
    if a.dry_run:
        print("\n--dry-run：未执行。")
        return 0

    # ★ 修 B13：记录已存在的检查必须**前置**。原写法要等训练+评测全跑完才在
    #   write_record 抛 FileExistsError，整轮算力白费。
    if not a.overwrite:
        clash = [record_path(exp_id, ds_id, tier, int(s)) for s in seeds if record_path(exp_id, ds_id, tier, int(s)).exists()]
        if clash:
            print("\n[中止] 以下记录已存在，训练前即停止（避免白跑）：")
            for c in clash:
                print("   -", c)
            print("   确认要重跑请加 --overwrite。")
            return 1

    failed: list[str] = []
    for s in seeds:
        try:
            rec = train_one_seed(
                exp_id=exp_id,
                data=data,
                model=model,
                seed=int(s),
                tier=tier,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                optimizer=optimizer,
                extra_train=dict(extra_train),
                extra_val=dict(extra_val),
            )
            p = write_record(rec, overwrite=a.overwrite)
            print(f"\n[完成] seed={s} 记录已落盘：{p}")
            print(f"   主指标 {rec.primary_metric} = {rec.primary_value:.4f}")
        except Exception as e:  # noqa: BLE001 —— 一个种子失败不应静默跳过，要记账
            failed.append(f"seed={s}: {type(e).__name__}: {e}")
            print(f"\n[失败] seed={s}：{type(e).__name__}: {e}")

    if failed:
        print("\n以下种子失败（必须处理，不得选择性报告）：")
        for f in failed:
            print("  -", f)
        return 1

    print(f"\n全部 {len(seeds)} 个种子完成。聚合：")
    print(f"  python -m src.utils.results_io {exp_id} {ds_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
