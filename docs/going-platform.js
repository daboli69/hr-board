/* GOING integration: compare measured model ranks with exact-game market ranks. */
(() => {
 const platform=location.pathname.startsWith('/yard/'), originalRender=renderView, originalLoadOdds=loadOddsEager;
 const html=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const decimal=a=>typeof a==='number'&&Math.abs(a)>=100?(a>0?1+a/100:1+100/-a):null;
 const median=xs=>{const a=xs.slice().sort((x,y)=>x-y);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2:null;};
 const aliases={AZ:'Arizona Diamondbacks',ARI:'Arizona Diamondbacks',ATL:'Atlanta Braves',BAL:'Baltimore Orioles',BOS:'Boston Red Sox',CHC:'Chicago Cubs',CWS:'Chicago White Sox',CIN:'Cincinnati Reds',CLE:'Cleveland Guardians',COL:'Colorado Rockies',DET:'Detroit Tigers',HOU:'Houston Astros',KC:'Kansas City Royals',LAA:'Los Angeles Angels',LAD:'Los Angeles Dodgers',MIA:'Miami Marlins',MIL:'Milwaukee Brewers',MIN:'Minnesota Twins',NYM:'New York Mets',NYY:'New York Yankees',OAK:'Athletics',ATH:'Athletics',PHI:'Philadelphia Phillies',PIT:'Pittsburgh Pirates',SD:'San Diego Padres',SF:'San Francisco Giants',SEA:'Seattle Mariners',STL:'St. Louis Cardinals',TB:'Tampa Bay Rays',TEX:'Texas Rangers',TOR:'Toronto Blue Jays',WSH:'Washington Nationals',WAS:'Washington Nationals'};
 const matchesTeam=(name,code)=>normName(name)===normName(aliases[code]||code)||(['OAK','ATH'].includes(code)&&normName(name).endsWith('athletics'));
 let quoteIndex=null,indexedShared=null,confirmedOnly=false;
 let shared=null,loadNote='Saved bookmaker snapshot',query='',onlySaved=false;
 let watch={};try{watch=JSON.parse(localStorage.getItem('going.yard.fantasy.watch')||'{}');}catch{}
 const save=id=>{watch[id]=!watch[id];try{localStorage.setItem('going.yard.fantasy.watch',JSON.stringify(watch));}catch{loadNote='Watchlist could not be saved; browser storage is unavailable.';}renderView();};
 function currentQuotes(p,market='player_home_runs'){
  if(!shared?.props)return [];
  const g=(BOARD?.games||[]).find(g=>g.game_pk===p.game_pk);if(!g||Date.parse(g.time)<=Date.now())return [];
  if(indexedShared!==shared){quoteIndex=new Map();for(const r of shared.props){const key=normName(r.player);if(!quoteIndex.has(key))quoteIndex.set(key,[]);quoteIndex.get(key).push(r);}indexedShared=shared;}
  return (quoteIndex.get(normName(p.name))||[]).filter(r=>r.market_key===market&&(market!=='player_home_runs'||r.line===.5)&&normName(r.player)===normName(p.name)&&matchesTeam(r.home_team,g.home)&&matchesTeam(r.away_team,g.away)&&Math.abs(Date.parse(r.commence_time)-Date.parse(g.time))<=30*60000&&Date.parse(r.commence_time)>Date.now()&&!r.is_dfs_flat_payout&&!r.dfs_normalized&&!['underdog','prizepicks','betr','pick6','sleeper'].includes(r.bookmaker)&&Date.now()-Date.parse(r.last_update)<=24*3600000&&Date.parse(r.last_update)<=Date.now());
 }
 function marketComparison(){
  const result=[];
  for(const p of BOARD?.players||[]){
   if(!Number.isFinite(p.heat)||!['confirmed','projected'].includes(p.lineup_status)||(confirmedOnly&&p.lineup_status!=='confirmed'))continue;
   const books=new Map();for(const r of currentQuotes(p)){const d=decimal(r.over_price);if(d)books.set(r.bookmaker,1/d);}
   // No same-day join by name alone; doubleheaders need the matching event above.
   if(books.size<2)continue;
   result.push({p,books:books.size,market:median([...books.values()]),score:p.heat});
  }
  const model=result.slice().sort((a,b)=>b.score-a.score),market=result.slice().sort((a,b)=>b.market-a.market);
  for(const r of result){r.modelRank=1+model.filter(x=>x.score>r.score).length;r.marketRank=1+market.filter(x=>x.market>r.market).length;r.delta=r.marketRank-r.modelRank;}
  return result.filter(r=>r.delta>=5).sort((a,b)=>b.delta-a.delta);
 }
 function legacyFromShared(raw){
  if(!BOARD?.slate_date)return null;
  const prices={},props={hits:{},hrr:{},pk:{},tb:{}},map={player_hits:'hits',player_hits_runs_rbis:'hrr',player_strikeouts:'pk',player_total_bases:'tb'};
  const arms=(BOARD.arms||[]).map(p=>{const games=(BOARD.games||[]).filter(g=>[g.home,g.away].includes(p.team)&&[g.home,g.away].includes(p.opp));return {...p,game_pk:games.length===1?games[0].game_pk:null};});
  const active=[...(BOARD.players||[]),...arms].filter(p=>p.game_pk!=null&&Date.parse(p.time)>Date.now());
  for(const p of active){
   const peers=active.filter(q=>normName(q.name)===normName(p.name));if(peers.length!==1)continue;
   const rows=currentQuotes(p),key=normName(p.name),e={name:p.name,line:.5,books:{},last_update:null};
   for(const r of rows){if(!['draftkings','fanatics','fanduel'].includes(r.bookmaker)||!decimal(r.over_price))continue;e.books[r.bookmaker]=r.over_price;e.last_update=r.last_update;e.home_team=r.home_team;e.away_team=r.away_team;}
   const primary=Object.entries(e.books).filter(([b])=>b!=='fanduel'),pool=primary.length?primary:Object.entries(e.books);pool.sort((a,b)=>decimal(b[1])-decimal(a[1]));if(pool.length){e.best_book=pool[0][0];e.best=pool[0][1];e.fallback=!primary.length;prices[key]=e;}
   for(const [market,bucket] of Object.entries(map)){
    const grouped=new Map();for(const r of currentQuotes(p,market)){if(!Number.isFinite(r.line)||!['draftkings','fanatics','fanduel'].includes(r.bookmaker))continue;if(!grouped.has(r.line))grouped.set(r.line,[]);grouped.get(r.line).push(r);}
    const entries=[];for(const [line,quotes] of grouped){const out={name:p.name,line,books:{}};for(const q of quotes){const over=decimal(q.over_price),under=decimal(q.under_price);if(!over&&!under)continue;out.books[q.bookmaker]={line,over,under};for(const side of ['over','under']){const value=decimal(q[side+'_price']);if(value&&(!out[side]||value>out[side])){out[side]=value;out[side+'_american']=q[side+'_price'];out[side+'_book']=q.bookmaker;}}out.home_team=q.home_team;out.away_team=q.away_team;out.last_update=q.last_update;}if(Object.keys(out.books).length)entries.push(out);}
    entries.sort((a,b)=>Object.keys(b.books).length-Object.keys(a.books).length||a.line-b.line);if(entries.length)props[bucket][key]={...entries[0],alt_lines:Object.fromEntries(entries.map(e=>[String(e.line),e]))};
   }
  }
  // Only merge a market that the shared endpoint actually loaded. A failed
  // category keeps its dated snapshot; an empty successful category is empty.
  return {...ODDS,updated:ODDS?.updated||raw.generated_at,shared_updated:raw.generated_at,slate_date:BOARD.slate_date,props:Object.fromEntries(Object.entries(map).map(([market,bucket])=>[bucket,raw.coverage?.markets?.[market]?.status==='loaded'?props[bucket]:(ODDS?.props?.[bucket]||{})])),prices:raw.coverage?.markets?.player_home_runs?.status==='unavailable'?(ODDS?.prices||{}):prices,count:Object.keys(prices).length};
 }
 loadOddsEager=async function(){
  await originalLoadOdds();if(!platform){loadNote='Open Model vs market in the shared GOING app for verified live game-matched prices.';return;}
  try{const r=await fetch('/api/odds?sport=mlb',{cache:'no-store'});if(!r.ok)throw Error();const data=await r.json();if(data.provider!=='parlay'||!Array.isArray(data.props))throw Error();shared=data;loadNote=`Shared live prices checked ${new Date(data.generated_at).toLocaleTimeString()}${data.coverage?.possibly_truncated?'; provider limit reached, coverage may be partial':''}${Object.values(data.coverage?.markets||{}).some(m=>m.status!=='loaded')?'; some market categories are unavailable':''}`;const merged=legacyFromShared(data);if(merged)ODDS=merged;}
  catch{loadNote='Shared live odds unavailable; saved baseball prices remain in the original views. Market-rank comparison needs a verified live feed.';}
  if(['market-disagreement','fantasy-research'].includes(view))renderView();
 };
 function card(p,body){return `<article class="going-card"><b>${html(p.name)}</b><span>${html(p.team)} vs ${html(p.opp_team)} · lineup ${html(p.lineup_spot??'not set')} · ${html(p.lineup_status||'not confirmed')}</span>${body}<button data-going-save="${p.id}" aria-pressed="${!!watch[p.id]}">${watch[p.id]?'★ Saved':'☆ Save to fantasy watchlist'}</button></article>`;}
 function renderMarket(){
  const rows=marketComparison().filter(r=>!query||`${r.p.name} ${r.p.team}`.toLowerCase().includes(query));
  document.getElementById('main').innerHTML=`<section class="going-section"><h2>Model above the market</h2>${!platform?'<p><a href="https://going-long.vercel.app/yard/?mode=market">Open the live comparison in GOING</a></p>':''}<p>The contact model ranks these hitters at least five places higher than the bookmaker comparison. Both rankings use the same eligible players. Projected lineups are labeled and can change. This is a difference in ranking—not a measured win chance or a claim the books are wrong.</p><p>Market order uses one current home-run price per book for the same game and 0.5 line, with at least two books. The bookmaker’s extra charge is still in those prices. ${html(loadNote)}.</p><button id="goingConfirmed" aria-pressed="${confirmedOnly}">${confirmedOnly?'Confirmed lineups only':'Include projected lineups'}</button><input id="goingQuery" placeholder="Search player or team" aria-label="Search market disagreements" value="${html(query)}"><div class="going-grid">${rows.slice(0,60).map(r=>card(r.p,`<p>Model rank <strong>#${r.modelRank}</strong> · market rank <strong>#${r.marketRank}</strong><br>${r.delta} places higher · ${r.books} books compared</p><p>Recorded contact score ${r.score.toFixed(1)}. The existing model has not been changed.</p>`)).join('')||'<p>No eligible disagreements yet. Eligible lineups and fresh matching prices are required; missing evidence is not filled in.</p>'}</div></section>`;
  document.getElementById('goingConfirmed').onclick=()=>{confirmedOnly=!confirmedOnly;renderMarket();};
  document.getElementById('goingQuery').oninput=e=>{query=e.target.value.toLowerCase();const at=e.target.selectionStart;renderMarket();const input=document.getElementById('goingQuery');input.focus();input.setSelectionRange(at,at);};
 }
 function renderFantasy(){
  const rows=(BOARD?.players||[]).filter(p=>(!onlySaved||watch[p.id])&&(!query||`${p.name} ${p.team}`.toLowerCase().includes(query))).sort((a,b)=>(b.heat||0)-(a.heat||0));
  document.getElementById('main').innerHTML=`<section class="going-section"><h2>Going Yard Fantasy</h2><p>Daily hitter research and a persistent watchlist. These are contact-quality rankings, not projected fantasy points. League import, scoring settings and roster management are not connected.</p><div class="going-actions"><input id="goingQuery" placeholder="Search player or team" aria-label="Search fantasy research" value="${html(query)}"><button id="goingSaved" aria-pressed="${onlySaved}">${onlySaved?'Show all hitters':'Saved players'}</button></div><div class="going-grid">${rows.slice(0,80).map(p=>card(p,`<p>Contact model score ${(p.heat||0).toFixed(1)} · ${html(p.trend||'No trend')}<br>Opponent: ${html(p.opp_pitcher?.name||'Starter not confirmed')}</p><p>Recent home runs: ${typeof p.hr_recent==='number'?p.hr_recent:'See player details'} · ${html(p.time||'Start time unavailable')}</p><button data-going-player="${p.id}">Open complete player research</button>`)).join('')||'<p>No players match. Saved players return here when they appear on a published slate.</p>'}</div></section>`;
  document.getElementById('goingSaved').onclick=()=>{onlySaved=!onlySaved;renderFantasy();};document.getElementById('goingQuery').oninput=e=>{query=e.target.value.toLowerCase();const at=e.target.selectionStart;renderFantasy();const input=document.getElementById('goingQuery');input.focus();input.setSelectionRange(at,at);};
 }
 renderView=function(){if(!BOARD)return originalRender();if(view==='market-disagreement')renderMarket();else if(view==='fantasy-research')renderFantasy();else originalRender();};
 document.getElementById('main').addEventListener('click',e=>{const saveButton=e.target.closest('[data-going-save]'),player=e.target.closest('[data-going-player]');if(saveButton)save(Number(saveButton.dataset.goingSave));if(player)scanToPlayer(Number(player.dataset.goingPlayer));});
 const nav=document.createElement('nav');nav.className='going-platform-nav';nav.setAttribute('aria-label','GOING sports and modes');nav.innerHTML=`<a href="${platform?'/':'https://going-long.vercel.app/'}" class="going-logo">GOING<span> / YARD</span></a><div><button data-going-view="board">Betting</button><button data-going-view="fantasy-research">Fantasy</button><button data-going-view="market-disagreement">Model vs market</button></div>`;document.body.prepend(nav);nav.onclick=e=>{const b=e.target.closest('[data-going-view]');if(b)goView(b.dataset.goingView);};
 const style=document.createElement('style');style.textContent='.going-platform-nav{padding:12px 18px;background:#111914;color:#eef2e9;border-bottom:1px solid #3d4b40;display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}.going-platform-nav a{color:inherit;text-decoration:none}.going-logo{font:900 26px Arial;letter-spacing:-2px}.going-logo span{font-size:12px;letter-spacing:1px;color:#e98347}.going-platform-nav button,.going-section button{min-height:44px;background:#263129;color:#f4f5ed;border:1px solid #6b7b68;border-radius:8px;padding:8px 12px;cursor:pointer}.going-platform-nav div,.going-actions{display:flex;gap:8px;flex-wrap:wrap}.going-section{max-width:1050px;margin:15px auto;color:var(--ink);padding:12px}.going-section p{line-height:1.65;color:var(--muted)}.going-section input{min-height:44px;font-size:16px;max-width:100%;padding:10px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px}.going-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px;margin-top:18px}.going-card{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:16px;min-width:0}.going-card>span{display:block;margin-top:6px;font-size:12px;color:var(--muted)}.going-card button{margin-top:8px}.going-section button[aria-pressed=true]{border-color:#ec844b;color:#ffb483}.going-card b{font-size:18px}';document.head.append(style);
 const mode=new URLSearchParams(location.search).get('mode');if(mode){const destination=mode==='fantasy'?'fantasy-research':mode==='market'?'market-disagreement':'board';view=destination;savePrefs();}
 window.goingMarketComparison=marketComparison;
})();
