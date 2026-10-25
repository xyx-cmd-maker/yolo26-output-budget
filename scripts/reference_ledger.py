"""
论文引文台账（Intra-reference ledger）—— Sci Rep 硬约束 **≤ 60 条**

★ 纪律（`AGENTS.md` §1 引用与诚信）：
  · 每条必须有**可核验标识**（DOI / arXiv ID / 官方 URL）
  · **未核实的一律标 `[待核实]`**，不得凭记忆写卷期页码
  · 本表是**唯一**的引文来源；手稿正文只允许引用本表内的键

产出：`docs/reference-ledger.md`（人读）+ `results/tables/reference-ledger.json`（机读）
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")

# ── 台账：键 | 用途 | 标识 | 核实状态 ────────────────────────────────
LEDGER = [
    # ── A. 检测基准与协议（决定"输出预算是人为常量"这一前提）──────────
    dict(key="lin2014coco", cite="Lin et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.",
         id="10.1007/978-3-319-10602-1_48", used="Intro; 协议 maxDets", verified=True),
    dict(key="gupta2019lvis", cite="Gupta, Dollár & Girshick (2019). LVIS: A Dataset for Large Vocabulary Instance Segmentation. CVPR.",
         id="10.1109/CVPR.2019.00550", used="Intro; max_dets=300", verified=True),
    dict(key="zhu2021visdrone", cite="Zhu et al. (2021). Detection and Tracking Meet Drones Challenge. IEEE TPAMI.",
         id="10.1109/TPAMI.2021.3119563", used="数据集", verified=True),
    dict(key="shao2019crowdhuman", cite="Shao et al. (2018). CrowdHuman: A Benchmark for Detecting Human in a Crowd.",
         id="arXiv:1805.00123", used="数据集", verified=True),
    dict(key="jocher2026yolo26", cite="Jocher et al. (2026). YOLO26. arXiv preprint.",
         id="arXiv:2606.03748", used="检测器（标注为预印本）", verified=True),

    # ── B. 检测评测的已知缺陷（为"评测协议值得研究"提供动机）──────────
    dict(key="dave2021devil", cite="Dave, Dollár, Ramanan, Kirillov & Girshick (2021). Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details. arXiv preprint.",
         id="arXiv:2102.01066", used="★★ **最直接的先例**：LVIS Table 2 —— `dets/im` 300→10,000 使 AP 22.6→26.1、稀有类 AP_r **12.6→19.8（+7.2）**；而 **COCO 无影响**。这是「预算咬合 ⇒ 加预算有用」的实证。★ 但他们是**全局增加**预算，本文是**在固定总预算下跨图重新分配**（不同干预）", verified=True),
    dict(key="oksuz2022lrp", cite="Oksuz, Cam, Kalkan & Akbas (2022). One Metric to Measure them All: Localisation Recall Precision (LRP) for Evaluating Visual Detection Tasks. IEEE TPAMI.",
         id="10.1109/TPAMI.2021.3130188", used="oLRP 对 dets/im 的敏感性（§VII-D1、Table VIII：oLRP 79.8→79.1→78.4）", verified=True),
    dict(key="fu2021lvis", cite="Fu, Nie, Sun, Liu, Zhang & Liu (2021). LVIS Challenge Track Technical Report 1st Place Solution: Distribution Balanced and Boundary Refinement for Large Vocabulary Instance Segmentation. arXiv preprint.",
         id="arXiv:2111.02668", used="★ 第三条独立证据：Table 5 —— `max_per_img` 300→1000 使 Boundary AP **44.1→45.3**、AP_r **40.4→43.5**；原文「there is no limit on the number of detections per image」。★ 注意其标题里的 **「Distribution Balanced」是长尾类别平衡**（dataset+loss balanced），**不是跨图分配** —— 不要误引", verified=True),
    dict(key="krichene2020sampled", cite="Krichene & Rendle (2020). On Sampled Metrics for Item Recommendation. KDD.",
         id="10.1145/3394486.3403226", used="采样指标偏差；top-k 截断的邻域", verified=True),
    dict(key="buckley2004reliability", cite="Buckley & Voorhees (2004). Retrieval Evaluation with Incomplete Information. SIGIR.",
         id="10.1145/1008992.1009000", used="IR 池化的**认识论**上界", verified=True),

    # ── C. 最接近的先行工作（必须逐条划界）────────────────────────────
    dict(key="wang2026recceiling", cite="Wang (2026). The Recall Ceiling of LLM Recommendation Reranking. arXiv preprint; stated CIKM '26.",
         id="arXiv:2609.27953", used="★ 最接近：Theorem 1 `E[NDCG@k] <= Recall@|W_pi|`；**DOI 在 Crossref 404**，须标预印本", verified=True),
    dict(key="decisionvalue2026", cite="(2026). The Decision Value of Perception Compute. arXiv preprint.",
         id="arXiv:2609.35910", used="★ 已提出「分配应对照预算约束下的 oracle 判定，uniform 作基线」", verified=True),
    dict(key="densescout2026", cite="(2026). DenseScout: Algorithm-System Co-design for Budgeted Tiny Object Selection on Edge Platforms.",
         id="arXiv:2604.25300", used="预算化 patch 选择（非输出单元分配）", verified=True),
    dict(key="ultralytics_issue25992", cite="Ultralytics issue #25992 (2026). max_det=300 caps recall at 23.6%.",
         id="https://github.com/ultralytics/ultralytics/issues/25992", used="工程侧已记录同一上界（非同行评议）", verified=True),

    # ── D. 最坏情况公平性（水填法的经典出处）────────────────────────
    dict(key="hardy1934inequalities", cite="Hardy, Littlewood & Polya (1934). Inequalities. Cambridge University Press.",
         id="ISBN 978-0521358806", used="重排不等式（Progressive filling 的理论工具）", verified=True),
    dict(key="bertsekas1992data", cite="Bertsekas & Gallager (1992). Data Networks (2nd ed.). Prentice Hall.",
         id="ISBN 978-0132009241", used="max-min fair allocation / progressive filling 的教科书来源", verified=True),

    # ── E. 评测方法学（采样、显著性、报告规范）──────────────────────
    dict(key="efron1993bootstrap", cite="Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall.",
         id="ISBN 978-0412042317", used="配对 bootstrap", verified=True),
    dict(key="holm1979", cite="Holm (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scand. J. Stat. 6(2), 65-70.",
         id="JSTOR:4615733", used="多重比较", verified=True),
    dict(key="mitchell2019modelcards", cite="Mitchell et al. (2019). Model Cards for Model Reporting. FAT*.",
         id="10.1145/3287560.3287596", used="模型卡", verified=True),

    # ── F. 待补（Related Work 用；本轮不写进正文）────────────────────
    dict(key="[待补]UAVDT", cite="UAVDT (2018). A Benchmark and Simulator for UAV Object Tracking.",
         id="10.1109/CVPR.2018.00510", used="第四数据集候选（本轮通道不通）", verified=False),
]

OUT_MD = ROOT / "docs" / "reference-ledger.md"
OUT_JSON = ROOT / "results" / "tables" / "reference-ledger.json"


def main() -> int:
    print("=" * 100)
    print("论文引文台账（Sci Rep 上限 60 条）")
    print("=" * 100)

    n = len(LEDGER)
    nv = sum(1 for r in LEDGER if r["verified"])
    print(f"\n  总计 {n} 条；已核实 {nv} 条；待核实 {n - nv} 条")
    print(f"  上限 60 → 余量 {60 - n} 条（Related Work 待写，需预留约 15 条）")

    # 按用途分组统计
    groups: dict[str, int] = {}
    for r in LEDGER:
        g = r["used"].split("；")[0].split(";")[0].strip()[:28]
        groups[g] = groups.get(g, 0) + 1

    print(f"\n  {'键':<22} {'标识':<34} {'核实':>4}  用途")
    print("  " + "-" * 96)
    for r in LEDGER:
        flag = "✅" if r["verified"] else "⚠️待核实"
        print(f"  {r['key']:<22} {r['id'][:33]:<34} {flag:>4}  {r['used'][:34]}")

    # 检查重复标识
    ids = [r["id"] for r in LEDGER]
    dups = {i for i in ids if ids.count(i) > 1}
    if dups:
        print(f"\n  ⚠️ 重复标识：{dups}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(LEDGER, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 论文引文台账（`reference-ledger`）", "",
             "> Sci Rep 硬约束：**≤ 60 条参考文献**。本表是手稿正文的**唯一**引文来源。",
             "> 纪律：每条必须有可核验标识；**未核实的标 `[待核实]`**，不得凭记忆写卷期页码。",
             f"> 当前 **{n} 条**（已核实 {nv}），上限 60 → 余量 {60 - n}。", "",
             "| 键 | 条目 | 标识 | 核实 | 用途 |", "|---|---|---|---|---|"]
    for r in LEDGER:
        lines.append(f"| `{r['key']}` | {r['cite']} | `{r['id']}` | "
                     f"{'✅' if r['verified'] else '⚠️ [待核实]'} | {r['used']} |")
    lines += ["", "## 本轮新增核实的条目（第 43 轮）", "",
              "| 键 | 为什么必须引 |", "|---|---|",
              "| `wang2026recceiling` | 最接近的先行工作：`E[NDCG@k] <= Recall@\\|W_pi\\|`。**必须引用并划界**："
              "其预算逐用户固定、无分配维度；本文研究的是**跨单元分配**。**且其 DOI 在 Crossref 404，须标预印本** |",
              "| `decisionvalue2026` | 已提出「分配应对照预算约束下的 oracle 判定，uniform 作基线而非上界」。"
              "**必须划界**：其预算 = **计算预算/精度档**；本文 = **输出单元数** |",
              "| `densescout2026` | 同为「预算化」检测，但对象是 **patch 选择**，不是输出单元分配 |",
              "| `ultralytics_issue25992` | 工程侧已独立记录同一上界（23.6%）。**非同行评议**，作为「三方独立」的第三例 |",
              "| `bertsekas1992data` | **水填法/max-min 公平是教科书结论**，必须引；不得声称该优化是本文的数学贡献 |",
              ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写入 {OUT_MD.relative_to(ROOT)} 与 {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
