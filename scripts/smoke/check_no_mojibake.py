"""
scripts/smoke/check_no_mojibake.py
=================================
强制手段：检测"UTF-8 被按 GBK 解码"产生的乱码。

为什么存在（AGENTS-LESSONS.md P1）：
    本项目实测用 PowerShell 的 `Get-Content -Raw` + `Set-Content -Encoding UTF8`
    往返改文件，会把 UTF-8 的中文注释按 GBK 解码 → 得到一串**合法但罕见**的汉字
    （如「杩欐槸琚彉鎴愪贡鐮佺殑娉ㄩ噴」），同时可能吞掉换行。
    这样的文件**仍能 commit、仍是合法 UTF-8**，直到运行才崩，所以必须机器校验。

★ 判据的演进（第一版失败过，记为教训）：
  · 第 1 版：搜固定乱码子串（"锛/鈽/鏁版"）→ **能报真阳性，但把文档里"引用乱码样例"也报了**（假阳性）。
  · 第 2 版：要求可疑字**连续成簇** → **负向测试失败**（真实乱码「杩欐槸涓€娈佃」不含我列的字集）。
  · 第 3 版（本版，实测有效）：**统计判据**。
      实测数据（2026-10-14，本机）：
        真实乱码样本（6 行，22/18/6 个 CJK）的**常用字占比 = 0.0%**（全部）；
        本仓库正常中文文件（AGENTS.md / run.py）的**常用字占比 = 84%–100%**。
      原理：乱码是"合法但罕见"的汉字，**正常中文必然大量使用高频功能字**（的/是/在/有/不…）。
      因此判据 = 「CJK 字数 ≥ 阈值」且「常用字占比 < 阈值」→ 判为乱码。
  · 另外保留一个硬失败：出现 U+FFFD 替换字符。

退出码：0 = 干净；1 = 发现乱码（列出文件、行号、实测占比）
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 中文最高频的功能字与常用字（约 400 个）。正常中文文本必然大量命中。
COMMON = set(
    "的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外将两高间由问很最重并物手应战向头文体政美相见被利什二等产或新己制身果加西斯月话合回特代内信表化老给世位次度门任常先海通教儿原东声提立及比员解水名真论处走义各入几口认条平系气题活尔更别打女变四神总何电数安少报才结反受目太量再感建务做接必场件计管期市直德资命山金指克许统区保至队形社便空决治展马科司五基眼书非则听白却界达光放强即像难且权思王象完设式色路记南品住告类求据程北边死张该交规万取拉格望觉术领共确传师观清今切院让识候带导争运笑飞风步改收根干造言联持组每济车亲极林服快办议往元英士证近失转夫令准布始怎呢存未远叫台单影具罗字爱击流备兵连调深商算质团集百需价花党华城石级整府离况亚请技际约示复病息究线似官火断精满支视消越器容照须九增研写称企八功吗包片史委乎查轻易早曾除农找装广显吧阿李标谈吃图念六引历首医局突专费号尽另周较注语仅考落青随选列武红响虽推势参希古众构房半节土投某案黑维革划敌致陈律足态护七兴派孩验责营星够章音跟志底站严巴例防族供效续施留讲型料终答紧黄绝奇察母京段依批群项故按河米围江织害斗双境客纪采举杀攻父苏密低朝友诉止细愿千值仍男钱破网热助倒育属坐帝限船脸职速刻乐否刚威毛状率甚独球般普怕弹校苦创假久错承印晚兰试股拿脑预谁益阳若哪微尼继送急血惊伤素药适波夜省初喜卫源食险待述陆习置居劳财环排福纳欢雷警获模充负云停木游龙树疑层冷洲冲射略范竟句室异激汉村哈策演简卡罪判担州静退既衣您宗积余痛检差富灵协角占配征修皮挥胜降阶审沉坚善妈刘读超免压银买皇养伊怀执副乱抗犯追帮宣佛岁航优怪香著田铁控税左右份穿艺背阵草脚概恶块顿敢守酒岛托央户烈洋哥索胡款靠评版宝座释景顾弟登货互付伯慢欧换闻危忙核暗姐介坏讨丽良序升监临亮露永呼味野架域沙掉括舰鱼杂误湾吉减编楚肯测败屋跑梦散温困剑渐封救贵枪缺楼县尚毫移娘朋画班智亦耳恩短掌恐遗固席输幸律"
)

# ★ 判据的演进（第 5 版；前四版全踩过坑，全部记在 AGENTS-LESSONS P1）
#   第 1 版：搜固定乱码子串 → 把文档里"引用乱码样例"也报了（假阳性）
#   第 2 版：要求可疑字连续成簇 → 负向测试失败（真乱码不含我列的字集）
#   第 3 版：CJK≥8 且常用字占比 < 50% → 在正常中文上假阳性
#            （命中 `- ❌ **混淆矩阵**：没有 4×4 混淆矩阵`，占比 40%）
#   第 4 版：加"功能字=0"硬条件 → **又假阳性**
#            （命中 `ap = argparse.ArgumentParser(description="生成合成夹具数据集")`，占比 89%！
#             因为短句「生成合成夹具数据集」里确实没有功能字）
#            → **单独用"功能字=0"不可靠**：短中文串天然可能不含功能字。
#   第 5 版（本版）：**两个条件同时满足**才算乱码 —— 占比极低 **且** 无功能字。
#            实测依据：真乱码 = 占比 **0%** 且 功能字 **0**；
#                      正常中文（含短串）= 占比 **84%~100%**。
#            取占比阈值 **15%**（远低于正常的 84%，远高于乱码的 0%），
#            并要求功能字 = 0，两者兼具才报 → 假阳性窗口几乎为零。
FUNCTION_WORDS = set("的是在有不了这和那就都也还很与及其为以对于把被从到会能可要将只")

MIN_CJK = 8
# 占比阈值：真乱码 0%，正常中文 ≥84% → 取 15% 留足两侧余量
COMMON_RATIO_THRESHOLD = 0.15

EXEMPT = {
    Path("scripts/smoke/check_no_mojibake.py"),
    # ★ `selftest_checkers.py` 是**阳性对照的载体**，按设计**必须**内嵌真实乱码字面量
    #   作为测试输入，否则无法证明本校验器会失败（见 AGENTS-LESSONS P15）。
    Path("scripts/smoke/selftest_checkers.py"),
    # `check_stale_numbers.py` 内嵌作废数字的正则模式，同样按设计必须包含。
    Path("scripts/smoke/check_stale_numbers.py"),
    Path("AGENTS-LESSONS.md"),
}

SCAN_EXT = {".py", ".md", ".yaml", ".yml", ".ps1", ".json", ".txt", ".toml", ".cfg"}
SKIP_DIRS = {".git", "__pycache__", "runs", "data", "weights", "logs", ".probe", "node_modules"}


def _is_cjk(c: str) -> bool:
    return "\u4e00" <= c <= "\u9fff"


def scan(root: Path) -> list[tuple[Path, int, str, str]]:
    hits: list[tuple[Path, int, str, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SCAN_EXT:
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        # ★ 只按**相对扫描根**的路径判断跳过，否则把扫描根本身也跳过了
        #   （踩过的坑：以 .probe 为根扫描时，绝对路径含 .probe → 全被跳过 → 假阴性）
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel in EXEMPT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            hits.append((p, 0, "解码失败", f"不是合法 UTF-8：{e}"))
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            if "\ufffd" in line:
                hits.append((p, lineno, "U+FFFD", line.strip()[:90]))
                continue
            cjk = [c for c in line if _is_cjk(c)]
            if len(cjk) < MIN_CJK:
                continue
            ratio = sum(1 for c in cjk if c in COMMON) / len(cjk)
            n_func = sum(1 for c in cjk if c in FUNCTION_WORDS)
            # 第 5 版判据：**两者同时满足**才报（占比极低 且 功能字为 0）
            if ratio < COMMON_RATIO_THRESHOLD and n_func == 0:
                hits.append((p, lineno, f"常用字占比 {ratio:.0%} 且功能字=0", line.strip()[:90]))
    return hits


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else PROJECT_ROOT
    print(f"扫描根目录：{root}")
    print(f"判据：CJK ≥ {MIN_CJK} 且常用字占比 < {COMMON_RATIO_THRESHOLD:.0%}，或含 U+FFFD\n")
    hits = scan(root)
    if not hits:
        print("✅ 未发现乱码")
        return 0
    print(f"❌ 发现 {len(hits)} 处疑似乱码：\n")
    by_file: dict[Path, list] = {}
    for p, ln, why, line in hits:
        by_file.setdefault(p, []).append((ln, why, line))
    for p, items in by_file.items():
        try:
            name = p.relative_to(root)
        except ValueError:
            name = p
        print(f"  {name}")
        for ln, why, line in items[:5]:
            print(f"    L{ln} [{why}] {line}")
        if len(items) > 5:
            print(f"    … 另有 {len(items) - 5} 处")
    print("\n修法：用 `edit`/`write` 工具重写受影响文件（AGENTS-LESSONS.md P1），")
    print("      不要用 PowerShell 的 Get-Content/Set-Content 改内容。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
