"""
yolo26-detection / src/utils/results_io.py
==========================================
结果目录的唯一写入通道 + 档位闸门。

为什么需要这个文件：
    AGENTS.md §2 规定「自检/冒烟/调试/降级档位的产物不得进入任何对外文本」。
    口头规定会被违反；这里把它变成**代码拒绝**：
        任何 tier != "paper" 的记录，aggregate() 直接拒绝聚合，
        emit_markdown() 直接拒绝出表。
    想要绕过，必须显式调用 force=True —— 让绕过的动作留下痕迹。

结果文件命名（唯一来源）：
    runs/records/<exp_id>__<dataset>__<tier>__s<seed>.json
    exp_id 例如 e02_yolo26n_baseline
聚合产物：
    runs/aggregated/<exp_id>__<dataset>__<tier>.json
    results/tables/<exp_id>__<dataset>.md
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "TIERS",
    "PAPER_TIER",
    "ResultRecord",
    "record_path",
    "write_record",
    "load_records",
    "aggregate",
    "emit_markdown",
    "TierViolation",
]

# 档位由低到高。只有 paper 档能进对外文本。
#   pilot = 真实数据但规模/设计不足以支撑论文主张（例如本项目 125 帧的视频抽帧集）
TIERS = ("debug", "smoke", "dev", "pilot", "paper")
PAPER_TIER = "paper"

# 数据集 → 主指标名。含糊的 "mAP" 不算数，必须写清是哪个 mAP。
PRIMARY_METRIC = {
    "coco": "mAP50-95",
    "coco8": "mAP50-95",
    "coco128": "mAP50-95",
    "visdrone": "mAP50",
    "tt100k": "mAP50",
    "bdd100k": "mAP50",
    "exdark": "mAP50",
    "dota": "mAP50",
    "default": "mAP50-95",
}


class TierViolation(RuntimeError):
    """试图把非 paper 档结果写进对外文本时抛出。"""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class ResultRecord:
    """一次 (exp_id, dataset, tier, seed) 评测的完整记录。

    metrics 里必须至少包含主指标；额外指标随意。所有数字都必须来自这次运行
    实际落盘的验证输出，不接受手工填写。
    """

    exp_id: str
    dataset: str
    tier: str
    seed: int
    metrics: dict[str, float]
    # --- 可追溯性字段：缺任何一个，这条记录就不能进论文 ---
    n_images: int | None = None
    n_instances: int | None = None
    imgsz: int | None = None
    conf: float | None = None
    iou: float | None = None
    max_det: int | None = None
    epochs: int | None = None
    batch: int | None = None
    optimizer: str | None = None
    split: str = "val"
    weights: str | None = None
    weights_sha256: str | None = None
    eval_seconds: float | None = None
    notes: str = ""
    fingerprint: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"未知档位 {self.tier!r}；合法值：{TIERS}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed 必须是非负整数，得到 {self.seed!r}")
        if "mAP" not in "".join(self.metrics):
            raise ValueError("metrics 里必须含至少一个以 mAP 开头的指标，否则无法与其他结果对齐")

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC.get(self.dataset, PRIMARY_METRIC["default"])

    @property
    def primary_value(self) -> float:
        key = self.primary_metric
        if key not in self.metrics:
            raise KeyError(f"记录 {self.exp_id}/{self.dataset} 缺少主指标 {key}；实际有 {list(self.metrics)}")
        return float(self.metrics[key])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_path(exp_id: str, dataset: str, tier: str, seed: int, root: Path | None = None) -> Path:
    """结果文件的唯一合法路径构造器。不要手拼路径。"""
    if tier not in TIERS:
        raise ValueError(f"未知档位 {tier!r}")
    return (root or _project_root()) / "runs" / "records" / f"{exp_id}__{dataset}__{tier}__s{seed}.json"


def write_record(rec: ResultRecord, root: Path | None = None, overwrite: bool = False) -> Path:
    """写入一条结果记录。默认拒绝覆盖——覆盖等于丢掉一个种子。"""
    p = record_path(rec.exp_id, rec.dataset, rec.tier, rec.seed, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} 已存在。要覆盖请显式 overwrite=True，并确认这不是在丢一个种子。")
    p.write_text(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_records(
    root: Path | None = None, exp_id: str | None = None, dataset: str | None = None, tier: str | None = None
) -> list[ResultRecord]:
    d = (root or _project_root()) / "runs" / "records"
    if not d.exists():
        return []
    out: list[ResultRecord] = []
    for p in sorted(d.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[results_io] 跳过无法解析的文件：{p}")
            continue
        if exp_id and obj.get("exp_id") != exp_id:
            continue
        if dataset and obj.get("dataset") != dataset:
            continue
        if tier and obj.get("tier") != tier:
            continue
        out.append(ResultRecord(**obj))
    return out


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)  # stdev = 样本标准差 (n-1)


def aggregate(
    exp_id: str,
    dataset: str,
    root: Path | None = None,
    *,
    force: bool = False,
    min_seeds: int = 3,
    tier: str = PAPER_TIER,
) -> dict[str, Any]:
    """把同 (exp_id, dataset, tier) 的多条记录聚合成 mean/std。

    闸门：
      * 默认**只聚合 paper 档**，且**只取 paper 档的记录**。
        ★ 修 B6：原实现会把同 exp_id 下所有档位的记录都抓进来，导致
        「dev 跑过一次同名实验」→ 这个 exp_id 的聚合被永久锁死。
        现在 dev 与 paper 各聚各的，互不污染。
      * 种子数 < min_seeds 时给出 blocked 标记，**不**默认放行。
      * force=True 能出结果，但产物里会带 forced=True 与原因，方便事后审计。
    """
    root = root or _project_root()
    all_recs = load_records(root, exp_id=exp_id, dataset=dataset)
    if not all_recs:
        raise FileNotFoundError(f"没有找到 {exp_id} / {dataset} 的任何记录")

    recs = load_records(root, exp_id=exp_id, dataset=dataset, tier=tier)
    present_tiers = sorted({r.tier for r in all_recs})
    if not recs:
        raise TierViolation(
            f"{exp_id}/{dataset} 只有档位 {present_tiers} 的记录，没有 {tier} 档。"
            f"按 AGENTS.md §2，非 paper 档不得聚合进对外文本。"
        )
    if tier != PAPER_TIER and not force:
        raise TierViolation(
            f"{exp_id}/{dataset} 请求聚合档位 {tier}（非 paper）。确认可放行请 force=True。"
        )

    by_metric: dict[str, list[float]] = {}
    for r in recs:
        for k, v in r.metrics.items():
            by_metric.setdefault(k, []).append(float(v))

    n_seeds = len({r.seed for r in recs})
    summary: dict[str, Any] = {
        "exp_id": exp_id,
        "dataset": dataset,
        "tier": tier,
        "tiers_present_for_this_exp": present_tiers,
        "n_records": len(recs),
        "n_seeds": n_seeds,
        "seeds": sorted({r.seed for r in recs}),
        "primary_metric": recs[0].primary_metric,
        "forced": bool(force and tier != PAPER_TIER),
        "blocked_reasons": [],
        "metrics": {},
        "provenance": {
            "weights_sha256": sorted({r.weights_sha256 for r in recs if r.weights_sha256}),
            "code_hashes": sorted({str((r.fingerprint or {}).get("code_hash")) for r in recs}),
            "gpu": sorted({str((r.fingerprint or {}).get("gpu")) for r in recs}),
            "imgsz": sorted({r.imgsz for r in recs if r.imgsz is not None}),
            "conf": sorted({r.conf for r in recs if r.conf is not None}),
            "iou": sorted({r.iou for r in recs if r.iou is not None}),
        },
    }
    if tier != PAPER_TIER:
        summary["blocked_reasons"].append(f"档位为 {tier}，非 paper")
    if n_seeds < min_seeds:
        summary["blocked_reasons"].append(f"种子数 {n_seeds} < 要求 {min_seeds}，方差不可估")

    for k, xs in sorted(by_metric.items()):
        m, s = _mean_std(xs)
        summary["metrics"][k] = {
            "mean": round(m, 5),
            "std": round(s, 5),
            "n": len(xs),
            "values": [round(x, 5) for x in xs],
        }

    out = root / "runs" / "aggregated" / f"{exp_id}__{dataset}__{tier}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["_written_to"] = str(out)
    return summary


def emit_markdown(summary: dict[str, Any], root: Path | None = None) -> Path:
    """把聚合结果出成 Markdown 表。非 paper 档直接拒绝（除非 summary 里 forced）。"""
    if summary.get("blocked_reasons") and not summary.get("forced"):
        raise TierViolation(
            f"该聚合结果带未消解的阻断原因，拒绝出表：{summary['blocked_reasons']}"
        )
    root = root or _project_root()
    tier = summary.get("tier", PAPER_TIER)
    lines = [
        f"# {summary['exp_id']} / {summary['dataset']}",
        "",
        f"- 档位：`{tier}`  种子数：{summary['n_seeds']}  记录数：{summary['n_records']}",
        f"- 主指标：`{summary['primary_metric']}`（均值 ± 样本标准差）",
    ]
    if summary.get("forced"):
        lines.append(f"- ⚠️ **本表经过强制放行**，阻断原因：{summary['blocked_reasons']}")
    prov = summary.get("provenance") or {}
    if prov:
        lines.append(f"- 可追溯：code_hash={prov.get('code_hashes')} gpu={prov.get('gpu')} imgsz={prov.get('imgsz')}")
    lines += ["", "| 指标 | mean | std | n | 各种子值 |", "|---|---|---|---|---|"]
    for k, v in summary["metrics"].items():
        star = " **←主**" if k == summary["primary_metric"] else ""
        lines.append(
            f"| `{k}`{star} | {v['mean']:.4f} | {v['std']:.4f} | {v['n']} | {', '.join(f'{x:.4f}' for x in v['values'])} |"
        )
    # ★ 修 B11：文件名必须带档位，否则 --force 出的非 paper 表会覆盖同名正式表
    out = root / "results" / "tables" / f"{summary['exp_id']}__{summary['dataset']}__{tier}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法：python -m src.utils.results_io <exp_id> <dataset> [--force] [--tier paper]")
        raise SystemExit(2)
    tier = PAPER_TIER
    if "--tier" in sys.argv:
        tier = sys.argv[sys.argv.index("--tier") + 1]
    s = aggregate(sys.argv[1], sys.argv[2], force="--force" in sys.argv, tier=tier)
    print(json.dumps(s, indent=2, ensure_ascii=False))
    if not s.get("blocked_reasons") or s.get("forced"):
        print("表格 ->", emit_markdown(s))
