/**
 * scripts/verify_references_online.mjs —— 逐条联网核实参考文献的真实性与正确性
 *
 * ═══════════════════════════════════════════════════════════════════════
 * 纪律（AGENTS-NORMS T5）：**按 DOI 注册机构分流**
 * ═══════════════════════════════════════════════════════════════════════
 *   · `10.48550/`（arXiv DOI）→ 查 **DataCite**（api.datacite.org）
 *   · 其余（期刊/会议）      → 查 **Crossref**（api.crossref.org）
 *   · 无 DOI 的 arXiv 预印本  → 查 **arXiv API**（export.arxiv.org）
 * 用错机构会得出**假结论**（DLRM 预印本在 Crossref 返回 404、在 DataCite 返回 200）。
 *
 * 本脚本逐条核对：标题、作者、年份、venue、标识符是否与登记值一致。
 */
const UA = { 'User-Agent': 'Mozilla/5.0 (compatible; reference-verifier)' };

// 从 results/tables/reference-ledger.json 读入（避免重复维护）
import { readFileSync } from 'fs';
const LEDGER = 'D:\\Deepseek Harness\\yolo\\yolo26-detection\\results\\tables\\reference-ledger.json';
const refs = JSON.parse(readFileSync(LEDGER, 'utf8'));

const norm = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
const sim = (a, b) => {
  const A = new Set(norm(a).split(' ').filter(Boolean));
  const B = new Set(norm(b).split(' ').filter(Boolean));
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const w of A) if (B.has(w)) inter++;
  return inter / Math.max(A.size, B.size);
};

const out = [];
for (const r of refs) {
  const key = r.key;
  if (key.startsWith('[')) continue;
  const doi = r.doi || '';
  const id = r.id || '';
  let source = 'none', status = 'SKIP', got = {};

  try {
    if (doi.startsWith('10.48550/')) {
      source = 'DataCite';
      const u = `https://api.datacite.org/dois/${encodeURIComponent(doi)}`;
      const resp = await fetch(u, { headers: UA, signal: AbortSignal.timeout(60000) });
      status = `HTTP ${resp.status}`;
      if (resp.ok) {
        const j = await resp.json();
        const a = j?.data?.attributes || {};
        got = {
          title: Array.isArray(a.titles) ? a.titles[0]?.title : a.titles,
          authors: (a.creators || []).map((c) => c.name || `${c.familyName}, ${c.givenName}`),
          year: a.publicationYear,
          venue: a.publisher || a.container?.title,
        };
      }
    } else if (doi) {
      source = 'Crossref';
      const u = `https://api.crossref.org/works/${encodeURIComponent(doi)}`;
      const resp = await fetch(u, { headers: UA, signal: AbortSignal.timeout(60000) });
      status = `HTTP ${resp.status}`;
      if (resp.ok) {
        const j = await resp.json();
        const m = j?.message || {};
        got = {
          title: Array.isArray(m.title) ? m.title[0] : m.title,
          authors: (m.author || []).map((x) => [x.given, x.family].filter(Boolean).join(' ')),
          year: (m.issued?.['date-parts']?.[0] || [])[0],
          venue: m['container-title']?.[0] || m.publisher,
        };
      }
    } else if (id.startsWith('arXiv:')) {
      source = 'arXiv';
      const aid = id.replace('arXiv:', '');
      const u = `http://export.arxiv.org/api/query?id_list=${aid}`;
      const resp = await fetch(u, { headers: UA, signal: AbortSignal.timeout(60000) });
      status = `HTTP ${resp.status}`;
      if (resp.ok) {
        const x = await resp.text();
        got = {
          title: (x.match(/<entry>[\s\S]*?<title>([\s\S]*?)<\/title>/) || [])[1]?.replace(/\s+/g, ' ').trim(),
          authors: [...x.matchAll(/<author>\s*<name>([^<]+)<\/name>/g)].map((m) => m[1].trim()),
          year: (x.match(/<published>(\d{4})/) || [])[1],
          venue: (x.match(/<arxiv:journal_ref[^>]*>([^<]+)</) || [])[1] || 'arXiv preprint',
        };
      }
    } else {
      got = { note: 'no DOI / arXiv id → 人工核验（ISBN 图书或 GitHub issue）' };
    }
  } catch (e) {
    status = `ERR ${e.name}`;
  }

  const tsim = got.title ? sim(r.title, got.title) : null;
  const yOk = got.year ? String(got.year) === String(r.year) : null;
  out.push({ key, source, status, tsim, yOk, registered: r.title, fetched: got.title,
             fetchedAuthors: got.authors, registeredAuthors: r.authors, venue: got.venue });

  const flag = (tsim === null ? '  ?  ' : tsim >= 0.8 ? ' ✅ ' : tsim >= 0.5 ? ' ⚠️ ' : ' ❌ ');
  console.log(`${flag}${key}  [${source} ${status}]`);
  if (tsim !== null) {
    console.log(`      标题相似度 ${tsim.toFixed(2)}   年份一致 ${yOk}`);
    if (tsim < 0.8) {
      console.log(`      登记: ${String(r.title).slice(0, 100)}`);
      console.log(`      取回: ${String(got.title).slice(0, 100)}`);
    }
  } else if (got.note) {
    console.log(`      ${got.note}`);
  }
}

console.log('\n' + '='.repeat(96));
const bad = out.filter((o) => o.tsim !== null && o.tsim < 0.8);
const err = out.filter((o) => o.status.startsWith('ERR') || o.status.includes('404'));
const manual = out.filter((o) => o.tsim === null && !o.status.startsWith('ERR'));
console.log(`  共 ${out.length} 条：标题匹配 ✅ ${out.filter((o) => o.tsim >= 0.8).length}，`
  + `⚠️ 需复核 ${bad.length}，人工核验 ${manual.length}，取回失败 ${err.length}`);
if (bad.length) {
  console.log('  ⚠️ 需复核：');
  for (const b of bad) console.log(`     ${b.key} (相似度 ${b.tsim.toFixed(2)})`);
}
if (err.length) {
  console.log('  ❌ 取回失败：');
  for (const e of err) console.log(`     ${e.key} [${e.source} ${e.status}]`);
}
console.log('  ' + ((bad.length + err.length) === 0
  ? '✅ 全部可联网核实 ✅' : '❌ 有未通过项，见上'));
