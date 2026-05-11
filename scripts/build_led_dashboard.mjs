import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ART = resolve(ROOT, "artifacts");
const OUT = resolve(ROOT, "outputs");

function parseCsv(text) {
  const rows = [];
  let field = "";
  let row = [];
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  const headers = rows.shift();
  return rows.filter((r) => r.length === headers.length).map((r) => Object.fromEntries(headers.map((h, i) => [h, r[i]])));
}

function num(row, key, fallback = 0) {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : fallback;
}

function countBy(rows, key) {
  return rows.reduce((acc, row) => {
    acc[row[key]] = (acc[row[key]] || 0) + 1;
    return acc;
  }, {});
}

function pct(part, whole) {
  return whole ? Math.round((part / whole) * 100) : 0;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function bar(name, count, total) {
  const width = pct(count, total);
  return `<div class="bar-row"><strong>${escapeHtml(name)}</strong><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><span>${count} (${width}%)</span></div>`;
}

function topFixture(row) {
  return `<div class="fixture-row"><strong>${escapeHtml(row.fixture)}</strong><div><strong>${escapeHtml(row.reason)}</strong><div class="reason-text">${escapeHtml(row.summary)}</div><div class="top-evidence"><span>LED ${Math.round(row.led * 100)}</span><span>Driver ${Math.round(row.driver * 100)}</span><span>${escapeHtml(row.confidence)} confidence</span><span>C${escapeHtml(row.circuit)}, ${row.windows} windows</span></div></div><strong>${Math.round(row.triage * 100)}</strong></div>`;
}

function signed(value, unit) {
  const rounded = formatNumber(value, 1);
  return `${value > 0 ? "+" : ""}${rounded}${unit}`;
}

function formatNumber(value, digits = 1) {
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: digits });
}

function buildDiagnosis({ status, reason, triage, led, driver, confidence, windows, spanMonths, staleFrac, boostLevel, envelope, levelDrift, envelopeDrift }) {
  const scoreText = `triage ${Math.round(triage * 100)}, LED ${Math.round(led * 100)}, driver ${Math.round(driver * 100)}`;
  const evidenceText = `${confidence.toLowerCase()} confidence from ${windows} batch windows over ${spanMonths.toFixed(2)} months; stale fraction ${Math.round(staleFrac * 100)}%.`;
  const measurements = `Boost level ${formatNumber(boostLevel)} mV, envelope ${formatNumber(envelope)} mV, level drift ${signed(levelDrift, " mV/month")}, envelope drift ${signed(envelopeDrift, " mV/month")}.`;

  if (status === "Working normally") {
    return `No repair signal: both LED-side and driver-side stress are low (${scoreText}). ${measurements} ${evidenceText}`;
  }
  if (status === "Monitor") {
    return `Not urgent yet: one signal is present, but the combined score stays below the watchlist threshold (${scoreText}). ${measurements} ${evidenceText}`;
  }
  if (reason === "LED and driver stress") {
    return `Likely electrical degradation: both the LED-side voltage signal and driver envelope signal are elevated (${scoreText}). ${measurements} ${evidenceText}`;
  }
  if (reason === "LED voltage drift") {
    return `Likely LED-side stress: the LED score is higher than the driver score, meaning boost-voltage level or level drift is the main concern (${scoreText}). ${measurements} ${evidenceText}`;
  }
  return `Likely driver-side instability: the driver score is higher than the LED score, meaning envelope magnitude or envelope drift is the main concern (${scoreText}). ${measurements} ${evidenceText}`;
}

function buildSummary({ status, reason, boostLevel, envelope, levelDrift, envelopeDrift, spanMonths, staleFrac }) {
  const trend = [
    `boost ${formatNumber(boostLevel)} mV`,
    `envelope ${formatNumber(envelope)} mV`,
    `level drift ${signed(levelDrift, " mV/month")}`,
    `envelope drift ${signed(envelopeDrift, " mV/month")}`,
  ].join("; ");
  const evidence = `${spanMonths.toFixed(2)} months of batch data; ${Math.round(staleFrac * 100)}% stale events.`;

  if (status === "Working normally") {
    return `No strong repair signal. Both LED-side voltage and driver envelope stress are low. Evidence: ${trend}. ${evidence}`;
  }
  if (status === "Monitor") {
    return `Some stress is visible, but it is not high enough for the watchlist. Evidence: ${trend}. ${evidence}`;
  }
  if (reason === "LED and driver stress") {
    return `Repair first: both the LED voltage signal and driver envelope signal are high. Evidence: ${trend}. ${evidence}`;
  }
  if (reason === "LED voltage drift") {
    return `Repair candidate: the LED-side voltage signal is the main concern. Evidence: ${trend}. ${evidence}`;
  }
  return `Repair candidate: the driver envelope signal is the main concern. Evidence: ${trend}. ${evidence}`;
}

async function loadRows() {
  const featureRows = parseCsv(await readFile(resolve(ART, "phase2_features.csv"), "utf8"));
  const features = new Map(featureRows.map((row) => [row.location_id, row]));
  const scoreRows = parseCsv(await readFile(resolve(ART, "phase3_scores.csv"), "utf8"));

  return scoreRows.map((row) => {
    const feat = features.get(row.fixture_id) || {};
    const triage = num(row, "S_triage");
    const led = num(row, "S_LED");
    const driver = num(row, "S_driver");

    let status = "Monitor";
    if (triage >= 0.85) status = "Repair priority";
    else if (triage >= 0.65) status = "Watchlist";
    else if (triage <= 0.35 && led <= 0.45 && driver <= 0.45) status = "Working normally";

    let reason = led > driver ? "LED voltage drift" : "Driver envelope instability";
    if (status === "Working normally") reason = "Low electrical stress";
    else if (Math.abs(led - driver) <= 0.12 && triage >= 0.65) reason = "LED and driver stress";

    const confidence = row.confidence_tier || "Low";
    const spanMonths = Number(num(row, "span_months").toFixed(2));
    const boostLevel = Number(num(feat, "F1_level_mV").toFixed(1));
    const envelope = Number(num(feat, "F2a_mag_mV").toFixed(1));
    const levelDrift = Number(num(feat, "F3_slope_mV_per_month").toFixed(2));
    const envelopeDrift = Number(num(feat, "F4_envelope_slope_mV_per_month").toFixed(2));
    const staleFrac = Number(num(feat, "stale_frac").toFixed(3));
    const windows = Math.round(num(row, "n_windows"));
    const record = {
      fixture: row.fixture_id,
      circuit: row.circuit,
      status,
      reason,
      confidence,
      data_flag: confidence === "Low" ? "Needs more evidence" : "Usable evidence",
      triage: Number(triage.toFixed(4)),
      led: Number(led.toFixed(4)),
      driver: Number(driver.toFixed(4)),
      costress: Number(num(row, "S_costress").toFixed(4)),
      windows,
      span_months: spanMonths,
      boost_level_mv: boostLevel,
      envelope_mv: envelope,
      level_drift_mv_month: levelDrift,
      envelope_drift_mv_month: envelopeDrift,
      stale_frac: staleFrac,
    };
    return {
      ...record,
      diagnosis: buildDiagnosis({
        status,
        reason,
        triage,
        led,
        driver,
        confidence,
        windows,
        spanMonths,
        staleFrac,
        boostLevel,
        envelope,
        levelDrift,
        envelopeDrift,
      }),
      summary: buildSummary({
        status,
        reason,
        spanMonths,
        staleFrac,
        boostLevel,
        envelope,
        levelDrift,
        envelopeDrift,
      }),
    };
  });
}

function render(rows) {
  const total = rows.length;
  const byStatus = countBy(rows, "status");
  const byConf = countBy(rows, "confidence");
  const flagged = rows.filter((row) => row.status === "Repair priority" || row.status === "Watchlist");
  const byReason = countBy(flagged, "reason");
  const repair = byStatus["Repair priority"] || 0;
  const watch = byStatus.Watchlist || 0;
  const monitor = byStatus.Monitor || 0;
  const healthy = byStatus["Working normally"] || 0;
  const top10 = [...rows].sort((a, b) => b.triage - a.triage).slice(0, 10);

  const reasonBars = Object.entries(byReason)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => bar(name, count, flagged.length))
    .join("");
  const confidenceBars = Object.entries(byConf)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => bar(`${name} confidence`, count, total))
    .join("");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LED Maintenance Triage Dashboard</title>
  <style>
    :root{color-scheme:light;--ink:#1b2430;--muted:#5a6675;--line:#d9e0e7;--panel:#fff;--page:#f5f7fa;--repair:#b83232;--watch:#b56b12;--monitor:#256f7f;--healthy:#237047;--accent:#315f9c;--soft-red:#fff0ed;--soft-amber:#fff5df;--soft-blue:#eaf5f8;--soft-green:#eaf7ef}
    *{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font-family:Arial,Helvetica,sans-serif;line-height:1.45}header{background:#132235;color:white;padding:26px 32px 22px;border-bottom:5px solid #d7a03a}h1,h2,h3,p{margin:0}h1{font-size:clamp(28px,4vw,44px);font-weight:800;letter-spacing:0}header p{max-width:1040px;color:#d8e1ec;margin-top:8px;font-size:16px}main{max-width:1400px;margin:0 auto;padding:24px 24px 42px}.meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}.pill{border:1px solid rgba(255,255,255,.26);color:#eef5ff;border-radius:999px;padding:7px 11px;font-size:13px;white-space:nowrap}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px;box-shadow:0 1px 2px rgba(20,30,45,.06)}.metric{min-height:150px}.label{color:var(--muted);font-size:13px;font-weight:700;text-transform:uppercase}.value{font-size:44px;font-weight:800;margin-top:6px;line-height:1}.caption{color:var(--muted);margin-top:10px;font-size:14px}.repair{border-top:5px solid var(--repair);background:var(--soft-red)}.watch{border-top:5px solid var(--watch);background:var(--soft-amber)}.monitor{border-top:5px solid var(--monitor);background:var(--soft-blue)}.healthy{border-top:5px solid var(--healthy);background:var(--soft-green)}.section{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(340px,.8fr);gap:16px;margin-top:16px}h2{font-size:20px;margin-bottom:14px}.bars{display:grid;gap:12px}.bar-row{display:grid;grid-template-columns:190px 1fr 64px;align-items:center;gap:12px;font-size:14px}.bar-track{height:14px;background:#edf1f5;border-radius:999px;overflow:hidden;border:1px solid #dde4eb}.bar-fill{height:100%;background:var(--accent);border-radius:999px}.reason .bar-fill{background:#8f5634}.confidence .bar-fill{background:#5d7899}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:middle}th{color:#334155;font-size:12px;text-transform:uppercase;background:#f4f7fa;position:sticky;top:0;z-index:1}.table-wrap{max-height:620px;overflow:auto;border:1px solid var(--line);border-radius:8px}.status{display:inline-flex;align-items:center;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:700;white-space:nowrap}.status.repair-priority{color:var(--repair);background:#f8d9d3}.status.watchlist{color:var(--watch);background:#f8e4ba}.status.monitoring{color:var(--monitor);background:#d8edf2}.status.working-normally{color:var(--healthy);background:#d6efdf}.score{display:grid;grid-template-columns:52px 90px;align-items:center;gap:8px}.spark{height:8px;background:#e3e8ee;border-radius:999px;overflow:hidden}.spark span{display:block;height:100%;background:#315f9c}.controls{display:grid;grid-template-columns:1.4fr repeat(3,minmax(150px,.5fr));gap:10px;margin-bottom:12px}input,select{width:100%;border:1px solid var(--line);border-radius:6px;padding:10px 11px;background:white;color:var(--ink);font:inherit}.small{color:var(--muted);font-size:13px}.top-list{display:grid;gap:11px}.fixture-row{display:grid;grid-template-columns:98px 1fr 58px;gap:12px;align-items:start;padding:12px 0;border-bottom:1px solid var(--line)}.fixture-row:last-child{border-bottom:0}.reason-text{color:var(--muted);font-size:13px;line-height:1.45;margin-top:4px}.top-evidence{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.top-evidence span{display:inline-flex;padding:3px 7px;border:1px solid #d7dde5;border-radius:999px;background:#f7f9fb;color:#536173;font-size:12px;white-space:nowrap}.diagnosis{min-width:360px;max-width:620px}.diagnosis strong{display:block;margin-bottom:4px}.diagnosis-text{color:#334155;font-size:13px;line-height:1.4}.data-chip{display:inline-flex;margin-top:7px;margin-right:6px;padding:3px 7px;border:1px solid #d7dde5;border-radius:999px;background:#f7f9fb;color:#536173;font-size:12px;white-space:nowrap}.note{margin-top:16px;padding:14px 16px;background:#fff;border:1px solid var(--line);border-left:5px solid #315f9c;border-radius:8px;color:#394657}.split{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}@media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.section,.split{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}}@media(max-width:650px){header{padding:22px 18px}main{padding:16px}.grid,.controls{grid-template-columns:1fr}.bar-row{grid-template-columns:1fr;gap:6px}.fixture-row{grid-template-columns:1fr 48px}.diagnosis{min-width:280px}th:nth-child(4),td:nth-child(4),th:nth-child(8),td:nth-child(8){display:none}}
  </style>
</head>
<body>
  <header>
    <h1>LED Maintenance Triage</h1>
    <p>YYC fixture health dashboard based on boost-voltage level, boost-voltage envelope, level drift, envelope drift, and confidence metadata. Counts are triage recommendations, not confirmed failure labels.</p>
    <div class="meta"><span class="pill">${total} fixtures scored</span><span class="pill">Within-circuit normalization</span><span class="pill">Generated May 11, 2026</span><span class="pill">Source: artifacts/phase3_scores.csv</span></div>
  </header>
  <main>
    <section class="grid" aria-label="Fleet summary">
      <article class="card metric repair"><div class="label">Should be fixed first</div><div class="value">${repair}</div><p class="caption">${pct(repair, total)}% of fixtures are repair priority because electrical stress is in the highest risk band.</p></article>
      <article class="card metric watch"><div class="label">Watchlist</div><div class="value">${watch}</div><p class="caption">${pct(watch, total)}% show elevated signs and should be reviewed before routine maintenance.</p></article>
      <article class="card metric monitor"><div class="label">Monitor</div><div class="value">${monitor}</div><p class="caption">${pct(monitor, total)}% are neither clean nor urgent based on available evidence.</p></article>
      <article class="card metric healthy"><div class="label">Working normally</div><div class="value">${healthy}</div><p class="caption">${pct(healthy, total)}% have low LED and driver stress scores.</p></article>
    </section>
    <section class="section">
      <article class="card"><h2>Why fixtures are flagged</h2><div class="bars reason">${reasonBars}</div><p class="caption">LED voltage drift points to elevated forward-voltage support under the CCR assumption. Driver envelope instability points to wider or worsening boost-voltage spread.</p></article>
      <article class="card"><h2>Evidence quality</h2><div class="bars confidence">${confidenceBars}</div><p class="caption">${byConf.Low || 0} fixtures have low confidence, usually because the export is batch-like and sparse. They remain visible, but the UI labels the evidence level.</p></article>
    </section>
    <section class="split">
      <article class="card"><h2>Top repair candidates</h2><div class="top-list">${top10.map(topFixture).join("")}</div></article>
      <article class="card"><h2>How to read this</h2><p class="small">Repair priority means the fixture sits in the high end of the circuit-normalized triage score. Working normally means both LED-side and driver-side stress are low. Low-confidence rows should be inspected with extra caution because missing operational telemetry can hide or exaggerate risk.</p><div class="note">Missing telemetry that would make this production-grade: input current, true LED junction temperature, alarm/failsafe streams, RF health, runtime counters, and maintenance/failure labels.</div></article>
    </section>
    <section class="card" style="margin-top:16px"><h2>Fixture table</h2><div class="controls"><input id="search" type="search" placeholder="Search fixture, reason, diagnosis, or status"><select id="statusFilter" aria-label="Status filter"><option value="all">All statuses</option><option>Repair priority</option><option>Watchlist</option><option>Monitor</option><option>Working normally</option></select><select id="circuitFilter" aria-label="Circuit filter"><option value="all">All circuits</option><option value="1">Circuit 1</option><option value="2">Circuit 2</option></select><select id="confidenceFilter" aria-label="Confidence filter"><option value="all">All confidence</option><option>Medium</option><option>Low</option></select></div><div class="table-wrap"><table><thead><tr><th>Fixture</th><th>Status</th><th>Why this lamp is flagged</th><th>Circuit</th><th>Triage</th><th>LED</th><th>Driver</th><th>Evidence</th><th>Boost mV</th><th>Envelope mV</th></tr></thead><tbody id="fixtureRows"></tbody></table></div></section>
  </main>
  <script>
    const fixtures=${JSON.stringify(rows)};
    const tbody=document.getElementById('fixtureRows'),search=document.getElementById('search'),statusFilter=document.getElementById('statusFilter'),circuitFilter=document.getElementById('circuitFilter'),confidenceFilter=document.getElementById('confidenceFilter');
    function cls(status){return status.toLowerCase().replace(/ /g,'-').replace('monitor','monitoring')}
    function score(value){const p=Math.round(value*100);return '<div class="score"><span>'+p+'</span><div class="spark"><span style="width:'+p+'%"></span></div></div>'}
    function fmt(value){return value.toLocaleString('en-US',{maximumFractionDigits:1})}
    function signed(value,unit){return (value>0?'+':'')+fmt(value)+unit}
    function renderTable(){const q=search.value.trim().toLowerCase(),status=statusFilter.value,circuit=circuitFilter.value,confidence=confidenceFilter.value;const visible=fixtures.filter(row=>{const haystack=(row.fixture+' '+row.status+' '+row.reason+' '+row.diagnosis+' '+row.confidence).toLowerCase();return(status==='all'||row.status===status)&&(circuit==='all'||row.circuit===circuit)&&(confidence==='all'||row.confidence===confidence)&&(!q||haystack.includes(q))});tbody.innerHTML=visible.map(row=>'<tr><td><strong>'+row.fixture+'</strong></td><td><span class="status '+cls(row.status)+'">'+row.status+'</span></td><td class="diagnosis"><strong>'+row.reason+'</strong><div class="diagnosis-text">'+row.diagnosis+'</div><span class="data-chip">Level drift '+signed(row.level_drift_mv_month,' mV/mo')+'</span><span class="data-chip">Envelope drift '+signed(row.envelope_drift_mv_month,' mV/mo')+'</span><span class="data-chip">Stale '+Math.round(row.stale_frac*100)+'%</span></td><td>C'+row.circuit+'</td><td>'+score(row.triage)+'</td><td>'+score(row.led)+'</td><td>'+score(row.driver)+'</td><td>'+row.confidence+'<div class="small">'+row.windows+' windows</div></td><td>'+fmt(row.boost_level_mv)+'</td><td>'+fmt(row.envelope_mv)+'</td></tr>').join('')}
    [search,statusFilter,circuitFilter,confidenceFilter].forEach(el=>el.addEventListener('input',renderTable));renderTable();
  </script>
</body>
</html>`;
}

const rows = await loadRows();
await mkdir(OUT, { recursive: true });
const outPath = resolve(OUT, "led_dashboard.html");
await writeFile(outPath, render(rows), "utf8");
console.log(`Wrote: ${outPath}`);
