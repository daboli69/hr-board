#!/usr/bin/env node
/**
 * View overlap audit — renders EVERY reachable screen and compares what each one actually
 * shows, so duplicated and (worse) *conflicting* surfaces become visible.
 *
 * WHY THIS EXISTS. The app grew one view at a time, and the same underlying question now gets
 * answered in several places by different code paths. "Which hitters are best for a 1+ hit
 * prop" is asked by the Hits tab, Top Picks, Qualified Bets, the SGP pool and the Ladder — and
 * because each computes its probability slightly differently, they return DIFFERENT PLAYERS.
 * That is worse than redundancy: the app contradicts itself and there is no way to tell which
 * answer to trust.
 *
 * The audit renders each view against real board.json, extracts the player IDs and prop types
 * it surfaces, then reports three categories:
 *
 *   DUPLICATE  — two views show substantially the same players for the same prop.
 *                Candidates to merge or drop.
 *   CONFLICT   — two views claim to rank the same thing but disagree on WHO.
 *                These must be reconciled; one of them is wrong.
 *   DISTINCT   — genuinely different questions. Keep both.
 *
 * Run:  node tests/audit_views.js
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.dirname(__dirname);
const HTML = path.join(ROOT, "docs", "index.html");

// Every reachable screen. Sub-views are listed explicitly because a tab inside a view is a
// separate destination to a person tapping around, even though the router calls it one view.
// `ranks` = the prop this screen puts in priority order. null means it is not a ranked list
// of players for a prop (a calendar, a help page, a matchup explorer), so it is never compared.
//
// `scope` = which population the screen ranks.
//   "all"      — the whole slate, so a disagreement with another "all" screen is a real
//                contradiction: two functions ranking the same people differently.
//   "filtered" — a deliberate subset (temperature-sensitive hitters, short-starter spots,
//                jackpot-eligible longshots). Showing different players is the WHOLE POINT, so
//                these are never flagged against a full-board view.
const VIEWS = [
  { id: "today",      label: "Today",                 fn: "renderToday",   ranks: "HR", scope: "filtered" },
  { id: "board",      label: "Board",                 fn: "render",        ranks: "HR", scope: "all" },
  { id: "toppicks",   label: "Top Picks",             fn: "renderTopPicks", ranks: "MIXED" },
  { id: "qualified",  label: "Qualified Bets",        fn: "renderQualified", ranks: "MIXED" },
  { id: "props",      label: "Props · HR",            fn: "renderProps", sub: { propsView: "ev" }, ranks: "HR", scope: "all" },
  { id: "props",      label: "Props · Hits",          fn: "renderProps", sub: { propsView: "hit" }, ranks: "Hit", scope: "all" },
  { id: "props",      label: "Props · HRR",           fn: "renderProps", sub: { propsView: "hrr" }, ranks: "HRR", scope: "all" },
  { id: "props",      label: "Props · Ks",            fn: "renderProps", sub: { propsView: "k" }, ranks: "Ks", scope: "all" },
  { id: "props",      label: "Props · Game Lines",    fn: "renderProps", sub: { propsView: "ml" }, ranks: "ML" },
  { id: "props",      label: "Props · Parlay",        fn: "renderProps", sub: { propsView: "parlay" }, ranks: null },
  { id: "converge",   label: "Convergence · HR",      fn: "renderConverge", sub: { cvProp: "hr" }, ranks: "HR", scope: "all" },
  { id: "converge",   label: "Convergence · Hits",    fn: "renderConverge", sub: { cvProp: "hit" }, ranks: "Hit", scope: "all" },
  { id: "converge",   label: "Convergence · HRR",     fn: "renderConverge", sub: { cvProp: "hrr" }, ranks: "HRR", scope: "all" },
  { id: "converge",   label: "Convergence · Ks",      fn: "renderConverge", sub: { cvProp: "ks" }, ranks: "Ks", scope: "all" },
  { id: "converge",   label: "Convergence · ML",      fn: "renderConverge", sub: { cvProp: "ml" }, ranks: "ML" },
  { id: "edges",      label: "Edges · Arms",          fn: "renderEdges", sub: { edgesView: "arms" }, ranks: "ARM", scope: "all" },
  { id: "edges",      label: "Edges · Quick Target",  fn: "renderEdges", sub: { edgesView: "qt" }, ranks: "ARM", scope: "filtered" },
  { id: "edges",      label: "Edges · Bullpens",      fn: "renderEdges", sub: { edgesView: "pens" }, ranks: "PEN" },
  { id: "edges",      label: "Edges · Microclimate",  fn: "renderEdges", sub: { edgesView: "micro" }, ranks: "HR", scope: "filtered" },
  { id: "edges",      label: "Edges · Late-HR",       fn: "renderEdges", sub: { edgesView: "late" }, ranks: "HR", scope: "filtered" },
  { id: "pitchmix",   label: "Team vs Pitch Mix",     fn: "renderPitchMix", ranks: null },
  { id: "sgp",        label: "Parlay Builder",        fn: "renderSGP",     ranks: "MIXED" },
  { id: "jackpot",    label: "Long Ball Jackpot",     fn: "renderJackpot", ranks: "HR", scope: "filtered" },
  { id: "ladder",     label: "Ladder Challenge",      fn: "renderLadder",  ranks: "MIXED" },
  { id: "calendar",   label: "HR Calendar",           fn: "renderCalendar", ranks: null },
  { id: "tracker",    label: "Tracker / Trends",      fn: "renderTracker", ranks: null },
  { id: "weather",    label: "Weather & Parks",       fn: "renderWeather", ranks: null },
  { id: "faq",        label: "Guide & FAQ",           fn: "renderFaq",     ranks: null },
];

function extractScripts(html) {
  const out = [];
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g;
  let m;
  while ((m = re.exec(html))) out.push(m[1]);
  return out.join("\n;\n");
}

function mkEl() {
  let ih = "";
  return {
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    style: { setProperty() {} }, setAttribute() {}, getAttribute: () => null,
    addEventListener() {}, appendChild() {}, insertAdjacentHTML(p, s) { ih += s; },
    querySelector: () => null, querySelectorAll: () => [],
    get innerHTML() { return ih; }, set innerHTML(v) { ih = v; },
    value: "", dataset: {}, offsetHeight: 50, textContent: "",
  };
}

function buildSandbox(board, backtest, history) {
  const mainEl = mkEl();
  const store = {};
  return {
    ctx: {
      __B__: board, __T__: backtest, __H__: history, __main__: mainEl,
      console, Date, Math, JSON, Intl, Set, Map,
      document: {
        getElementById: (id) => (id === "main" || id === "liveBody") ? mainEl : mkEl(),
        createElement: () => mkEl(), querySelector: () => mkEl(),
        querySelectorAll: () => [], addEventListener() {}, body: mkEl(), documentElement: mkEl(),
      },
      window: {
        addEventListener() {}, matchMedia: () => ({ matches: false, addEventListener() {} }),
        scrollTo() {}, location: { hash: "" }, innerWidth: 390, scrollY: 0,
      },
      navigator: { userAgent: "audit" },
      fetch: () => Promise.resolve({ ok: false }),
      localStorage: {
        getItem: (k) => store[k] || null, setItem: (k, v) => (store[k] = v),
        removeItem: (k) => delete store[k],
      },
      __ERR__: null,
      alert: () => {}, setTimeout: () => 0, setInterval: () => 0,
      clearInterval() {}, clearTimeout() {}, requestAnimationFrame: (f) => { f && f(); return 0; },
    },
    mainEl,
  };
}

/** What does this rendered screen actually put in front of a person? */
function fingerprint(html, board) {
  const names = new Set();
  const order = [];
  const byName = {};
  (board.players || []).forEach((p) => { if (p.name) byName[p.name] = p.id; });
  (board.pitcher_props || []).forEach((a) => { if (a.name) byName[a.name] = "P" + a.id; });
  // Record the order names appear in the rendered HTML — that IS the ranking a person sees.
  const hits = [];
  Object.keys(byName).forEach((n) => {
    let idx = -1;
    for (const pat of [">" + n + "<", n + "<", ">" + n]) {
      const i = html.indexOf(pat);
      if (i >= 0 && (idx < 0 || i < idx)) idx = i;
    }
    if (idx >= 0) { names.add(byName[n]); hits.push([idx, byName[n]]); }
  });
  hits.sort((a, b) => a[0] - b[0]).forEach(([, id]) => order.push(id));
  const props = new Set();
  [["HR", /Home Run|\bHR\b/], ["Hit", /1\+ Hits?|\bHits?\b/], ["HRR", /H\+R\+RBI|HRR/],
   ["Ks", /\bKs\b|Strikeout/], ["ML", /Moneyline|win prob|Game Lines/]].forEach(([k, re]) => {
    if (re.test(html)) props.add(k);
  });
  return { entities: names, order, props, size: html.length };
}

/** Spearman-style rank agreement over the entities both screens show. */
function rankAgreement(orderA, orderB) {
  const posB = new Map(orderB.map((id, i) => [id, i]));
  const pairs = [];
  orderA.forEach((id, i) => { if (posB.has(id)) pairs.push([i, posB.get(id)]); });
  if (pairs.length < 4) return null;
  const n = pairs.length;
  const mx = (n - 1) / 2;
  let num = 0, dx = 0, dy = 0;
  pairs.forEach(([a, b]) => {
    const ra = a - mx, rb = b - mx;   // ranks are already 0..n-1 within each list
    num += ra * rb; dx += ra * ra; dy += rb * rb;
  });
  return dx && dy ? num / Math.sqrt(dx * dy) : null;
}

function jaccard(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  a.forEach((x) => { if (b.has(x)) inter++; });
  return inter / (a.size + b.size - inter);
}

function main() {
  const html = fs.readFileSync(HTML, "utf8");
  const js = extractScripts(html);
  const board = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "board.json"), "utf8"));
  const backtest = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "backtest.json"), "utf8"));
  const history = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "history.json"), "utf8"));

  const results = [];
  for (const v of VIEWS) {
    const { ctx, mainEl } = buildSandbox(board, backtest, history);
    vm.createContext(ctx);
    let out = null, err = null;
    // BOARD/HISTORY are declared with let/const inside the bundle, so assigning them on the
    // context object after the fact does nothing — the binding already exists in module scope.
    // The assignment has to run as part of the SAME script, appended as a tail.
    const subAssigns = Object.entries(v.sub || {})
      .map(([k, val]) => `${k} = ${JSON.stringify(val)};`).join(" ");
    const tail = `
      ;(function(){
        BOARD = __B__; BACKTEST = __T__; HISTORY = __H__;
        view = ${JSON.stringify(v.id)}; ${subAssigns}
        try { ${v.fn} && ${v.fn}(); } catch(e) { __ERR__ = e.message; }
      })();`;
    try {
      vm.runInContext(js + tail, ctx, { timeout: 25000 });
      err = ctx.__ERR__ || null;
      out = mainEl.innerHTML || "";
    } catch (e) {
      err = e.message;
    }
    results.push({ ...v, err, fp: out ? fingerprint(out, board) : null,
                   bytes: out ? out.length : 0 });
  }

  // ---- report ----
  console.log("VIEW AUDIT — every reachable screen, rendered against the live board\n");
  console.log("1. DOES EVERY SCREEN RENDER?");
  let broken = 0, empty = 0;
  for (const r of results) {
    if (r.err) { console.log(`   FAIL  ${r.label.padEnd(26)} ${r.err}`); broken++; }
    else if (r.bytes < 400) { console.log(`   EMPTY ${r.label.padEnd(26)} ${r.bytes} bytes`); empty++; }
  }
  if (!broken && !empty) console.log("   all screens render with content");
  console.log(`   ${results.length} screens · ${broken} broken · ${empty} empty\n`);

  const live = results.filter((r) => r.fp && r.fp.entities.size >= 3);

  console.log("2. OVERLAP — which screens show the same players?\n");
  const pairs = [];
  for (let i = 0; i < live.length; i++) {
    for (let j = i + 1; j < live.length; j++) {
      const a = live[i], b = live[j];
      // Only compare screens that BOTH claim to rank something, and only when the thing is
      // comparable. MIXED lists (Top Picks, Qualified, parlay pools) legitimately span props,
      // so they are compared against single-prop screens but never flagged as conflicts.
      if (!a.ranks || !b.ranks) continue;
      const comparable = a.ranks === b.ranks;
      const mixed = a.ranks === "MIXED" || b.ranks === "MIXED";
      if (!comparable && !mixed) continue;
      // Only two full-slate rankings of the same prop can contradict each other.
      const filtered = a.scope === "filtered" || b.scope === "filtered";
      const sim = jaccard(a.fp.entities, b.fp.entities);
      const rho = rankAgreement(a.fp.order, b.fp.order);
      pairs.push({ a, b, sim, rho, sharedProps: [comparable ? a.ranks : "MIXED"], mixed, filtered });
    }
  }
  pairs.sort((x, y) => y.sim - x.sim);

  const DUP = [], CONFLICT = [], DISTINCT = [];
  for (const p of pairs) {
    if (p.mixed) { DISTINCT.push(p); continue; }        // a mixed list is meant to differ
    if (p.filtered) { DISTINCT.push(p); continue; }     // a subset is meant to differ
    // Two screens are only truly duplicated when they show the same people AND put them in
    // roughly the same order. Same membership with a different order is not redundancy — it is
    // two different rankings, which is the more interesting case.
    if (p.sim >= 0.60 && p.rho != null && p.rho >= 0.80) DUP.push(p);
    else if (p.sim >= 0.60) CONFLICT.push(p);          // same people, different order
    else if (p.sim < 0.35) CONFLICT.push(p);           // different people entirely
    else DISTINCT.push(p);
  }

  const show = (list, title, note) => {
    console.log(`   ${title}  (${list.length})`);
    if (note) console.log(`   ${note}`);
    list.slice(0, 12).forEach((p) => {
      const rhoTxt = p.rho == null ? "order n/a"
        : `order ${p.rho >= 0 ? "+" : ""}${p.rho.toFixed(2)}`;
      console.log(`     ${(100 * p.sim).toFixed(0).padStart(3)}% same players, ${rhoTxt}  ` +
        `${p.a.label} <-> ${p.b.label}   [${p.sharedProps.join(",") || "-"}]`);
    });
    if (list.length > 12) console.log(`     ... and ${list.length - 12} more`);
    console.log("");
  };

  show(DUP, "DUPLICATE — merge or drop one",
    "same prop, mostly the same players: two screens doing one job");
  show(CONFLICT, "CONFLICT — reconcile, one of these is wrong",
    "same prop, DIFFERENT players: the app contradicts itself here");
  console.log(`   DISTINCT — keep both  (${DISTINCT.length} pairs, different questions)\n`);

  console.log("3. PROP COVERAGE — how many screens answer each question?\n");
  const byProp = {};
  live.forEach((r) => { if (r.ranks) (byProp[r.ranks] = byProp[r.ranks] || []).push(r.label); });
  Object.entries(byProp).sort((a, b) => b[1].length - a[1].length).forEach(([prop, screens]) => {
    const flag = screens.length >= 4 ? "  <-- scattered across many screens" : "";
    console.log(`   ${prop.padEnd(5)} ${String(screens.length).padStart(2)} screens${flag}`);
    console.log(`         ${screens.join(", ")}`);
  });

  console.log("\nHOW TO READ THIS");
  console.log("  CONFLICT is the category that matters. Two screens ranking the same prop with");
  console.log("  different players means they use different probability functions, and only one");
  console.log("  can be right. Pick the calibrated one and have the other call it.");
  console.log("  DUPLICATE is cheaper to leave alone but costs payload and attention.");

  process.exit(broken ? 1 : 0);
}

main();
