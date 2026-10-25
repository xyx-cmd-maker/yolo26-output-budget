/**
 * 撤稿/更正检查（citation-and-related-work §2 的 T5 层）
 *
 * 按 DOI 注册机构分流：
 *   · 10.48550/ 开头 = arXiv → DataCite
 *   · 其余（期刊/会议）      → Crossref
 * 读 Crossref 的 `update-to` 字段（撤稿/更正会在这里留下记录）。
 */
import { readFileSync } from 'fs';
const UA = { 'User-Agent': 'Mozilla/5.0 (compatible; retraction-checker)' };
const LEDGER = 'D:\\Deepseek Harness\\yolo\\yolo26-detection\\results\\tables\\reference-ledger.json';
const refs = JSON.parse(readFileSync(LEDGER, 'utf8'));

const rows = [];
console.log('='.repeat(96));
console.log('撤稿 / 更正检查');
console.log('='.repeat(96));
for (const r of refs) {
  const key = r.key;
  if (key.startsWith('[')) continue;
  const doi = r.doi || '';
  if (!doi) {
    rows.push({ key, status: '无 DOI，跳过（ISBN/GitHub/arXiv-id 类）', verdict: 'n/a' });
    continue;
  }
  const isArxiv = doi.startsWith('10.48550/');
  const url = isArxiv
    ? `https://api.datacite.org/dois/${encodeURIComponent(doi)}`
    : `https://api.crossref.org/works/${encodeURIComponent(doi)}`;
  const agency = isArxiv ? 'DataCite' : 'Crossref';
  try {
    const resp = await fetch(url, { headers: UA, signal: AbortSignal.timeout(60000) });
    if (!resp.ok) {
      rows.push({ key, doi, agency, status: `HTTP ${resp.status}`, verdict: '❓ 未注册/不可解析' });
      continue;
    }
    const j = await resp.json();
    const m = isArxiv ? (j.data?.attributes || {}) : (j.message || {});
    const upd = m['update-to'] || m.updates || null;
    const isRetracted = Array.isArray(upd)
      ? upd.some((u) => /retract/i.test(u.type || ''))
      : /retract/i.test(String(upd?.type || ''));
    const verdict = isRetracted ? '❌ 已撤稿' : (upd ? '⚠️ 有更正/更新' : '✅ 正常（无撤稿记录）');
    rows.push({ key, doi, agency, status: `HTTP 200${upd ? ' + update-to' : ''}`, verdict,
                upd: upd ? JSON.stringify(upd).slice(0, 120) : '' });
    console.log(`  ${verdict}  ${key}`);
    console.log(`      ${agency}  ${doi}`);
    if (upd) console.log(`      update-to: ${JSON.stringify(upd).slice(0, 200)}`);
  } catch (e) {
    rows.push({ key, doi, agency, status: `ERR ${e.name}`, verdict: '❓ 检查失败' });
  }
}

console.log('\n' + '='.repeat(96));
const norm = rows.filter((r) => r.verdict.startsWith('✅')).length;
const bad = rows.filter((r) => r.verdict.startsWith('❌') || r.verdict.startsWith('⚠️'));
const na = rows.filter((r) => r.verdict === 'n/a').length;
const unk = rows.filter((r) => r.verdict.startsWith('❓')).length;
console.log(`  正常 ${norm} ｜ 有更新/撤稿 ${bad.length} ｜ 无 DOI 类 ${na} ｜ 未决 ${unk}`);
if (bad.length) {
  console.log('  需处理：');
  for (const b of bad) console.log(`     ${b.key}: ${b.verdict}`);
}
console.log('  ' + (bad.length === 0 ? '✅ 无撤稿/更正记录' : '❌ 见上'));
