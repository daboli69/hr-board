/* Existing Yard signals, not a new probability model. Browser + scheduled ETL. */
(function(root){'use strict';
const teams={ARI:'Arizona Diamondbacks',ATL:'Atlanta Braves',BAL:'Baltimore Orioles',BOS:'Boston Red Sox',CHC:'Chicago Cubs',CWS:'Chicago White Sox',CIN:'Cincinnati Reds',CLE:'Cleveland Guardians',COL:'Colorado Rockies',DET:'Detroit Tigers',HOU:'Houston Astros',KC:'Kansas City Royals',LAA:'Los Angeles Angels',LAD:'Los Angeles Dodgers',MIA:'Miami Marlins',MIL:'Milwaukee Brewers',MIN:'Minnesota Twins',NYM:'New York Mets',NYY:'New York Yankees',ATH:'Athletics',OAK:'Athletics',PHI:'Philadelphia Phillies',PIT:'Pittsburgh Pirates',SD:'San Diego Padres',SEA:'Seattle Mariners',SF:'San Francisco Giants',STL:'St. Louis Cardinals',TB:'Tampa Bay Rays',TEX:'Texas Rangers',TOR:'Toronto Blue Jays',WSH:'Washington Nationals'};
const norm=x=>String(x||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/\b(jr|sr|ii|iii)\b/g,'').replace(/[^a-z0-9]/g,''),finite=Number.isFinite;
const eastern=x=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York'}).format(new Date(x));
function stats(p,market='hr'){if(market==='hrr')return ['exp_hits','exp_runs','exp_rbi','xpa'].map((key,i)=>({label:['Expected hits','Expected runs','Expected RBIs','Expected plate appearances'][i],value:p.hrr_proj?.[key],max:i===3?6:3,note:'Existing Yard HRR projection'}));if(market==='hits')return [{label:'Hit research strength',value:p.hit_heat,max:100,note:'Not a probability'},{label:'Recent batting average',value:p.windows?.L14d?.ba,max:1,note:'Published 14-day window'},{label:'Recent expected average',value:p.windows?.L14d?.xba,max:1,note:'Expected, not an actual outcome'},{label:'Recent contact rate',value:p.windows?.L14d?.contact_pct,max:100,note:'Percent of swings with contact'}];return [{label:'Contact strength',value:p.heat,max:100,note:'Research score, not probability'},{label:'Pitch mix fit',value:p.mix_punish?.score,max:100,note:'Existing mix-punish score'},{label:'Recent barrel rate',value:p.metrics?.barrel_pct?.recent,max:100,display:finite(p.metrics?.barrel_pct?.recent)?p.metrics.barrel_pct.recent+'%':null,note:'Published recent window'},{label:'Starter vulnerability',value:p.opp_pitcher?.hr_score,max:100,note:p.opp_pitcher?.name||'Opposing starter'}];}
function build(board,odds,now=Date.now()){
 if(!board||!odds||!finite(Date.parse(board.generated_at))||now-Date.parse(board.generated_at)>30*3600000)return [];
 const rows=[],games=board.games||[];
 for(const p of board.players||[]){const game=games.find(g=>String(g.game_pk)===String(p.game_pk)),kickoff=Date.parse(game?.time);if(!finite(kickoff)||kickoff<=now||['out','bench','inactive'].includes(String(p.lineup_status).toLowerCase()))continue;
  for(const [market,score,label,pool]of [['hr',p.heat,'Home runs',odds.prices],['hits',p.hit_heat,'Hits',odds.props?.hits],['hrr',p.hrr_heat,'Hits + runs + RBIs',odds.props?.hrr]]){
   if(!finite(score))continue;
   const matches=Object.values(pool||{}).filter(q=>norm(q.name)===norm(p.name)&&norm(q.home_team)===norm(teams[game.home]||game.home)&&norm(q.away_team)===norm(teams[game.away]||game.away));if(matches.length!==1)continue;
   const q=matches[0],stamp=Date.parse(q.last_update),line=q.line;
   // Legacy quotes lack game IDs. Only a unique matchup on the same slate day is safe.
   if(!finite(stamp)||stamp>now+300000||now-stamp>24*3600000||!finite(line)||odds.slate_date!==eastern(game.time)||games.filter(g=>g.home===game.home&&g.away===game.away).length!==1)continue;
   for(const [book,quote]of Object.entries(q.books||{})){const dec=market==='hr'?(finite(quote)&&Math.abs(quote)>=100?(quote>0?1+quote/100:1+100/-quote):null):quote?.line===line?quote.over:null;if(!finite(dec)||dec<=1)continue;
    const american=market==='hr'?quote:Math.round(dec>=2?(dec-1)*100:-100/(dec-1));
    rows.push({id:`mlb|${game.game_pk}|${p.id}|${market}|${line}|${book}`,sport:'mlb',qualification:(board.top_plays||[]).some(r=>r.id===p.id)?'Existing Yard top play':'Research watchlist',event:String(game.game_pk),profileId:String(p.id),player:p.name,market,marketLabel:label,label:`${p.name} · Over ${line} ${label.toLowerCase()}`,line,side:'Over',home:game.home,away:game.away,kickoff:game.time,book,odds:american,dec,updatedAt:q.last_update,strength:score,projectionLabel:market==='hrr'&&finite(p.hrr_proj?.hrr)?`${p.hrr_proj.hrr.toFixed(2)} expected H+R+RBI`:`${score.toFixed(1)} research strength / 100`,confidence:p.reliability?.label||'Sample not rated',why:market==='hr'?(p.why||'Published power evidence is shown below.'):market==='hrr'?`The existing production score is ${score.toFixed(1)}. Expected hits, runs and RBIs are shown separately below; lineup spot is ${p.lineup_spot??'unconfirmed'}.`:`The existing contact score is ${score.toFixed(1)}. Recent batting average, expected average and contact measurements are shown below where recorded.`,risk:[...(p.score_breakdown?.flags||[]),p.lineup_status==='confirmed'?'Lineup confirmed; playing time can still change.':'Lineup not confirmed.',q.fallback?'Saved fallback quote: verify at the book.':'A research score is not a calibrated win probability.'].join(' '),parlay:'Compare with football at the same book. No estimated joint win chance; same-game correlation requires a book quote.',chartStats:stats(p,market),evidence:[['Market signal',`${market} strength ${score}; not a probability`],['Recent sample',JSON.stringify(p.sample||{})],['Lineup',`${p.lineup_status||'Unknown'} · spot ${p.lineup_spot??'unknown'}`],['Badges',(p.badges||[]).map(b=>b.t).join(', ')||'None'],['Quote',`${q.fallback?'FALLBACK':'Published'} · ${q.last_update}`],['Board cutoff',board.generated_at]],source:`Yard ${board.model_version||'published model'} · existing current-season signals`,researchUrl:`/yard/?player=${p.id}`});
   }
  }
 }return rows.sort((a,b)=>b.strength-a.strength||a.player.localeCompare(b.player));
}
function fromModel(payload,now=Date.now()){
 if(!payload||payload.status==='STALE'||payload.status==='FAILED')return [];
 return (payload.opportunities||[]).filter(r=>r.freshness==='FRESH'&&Date.parse(r.event.start)>now&&now-Date.parse(r.price.quoted_at)<=3*3600000).map(r=>({
  id:r.id,sport:'mlb',event:String(r.game_pk),profileId:r.entity.id==='game'?null:r.entity.id,player:r.entity.name,
  market:r.market,marketLabel:({hr:'Home runs',hits:'Hits',hrr:'Hits + runs + RBIs',pk:'Pitcher strikeouts',moneyline:'Moneyline',spread:'Run line',total:'Game total'})[r.market],
  label:r.bet_description,line:r.market_line,side:r.side,home:r.event.home,away:r.event.away,kickoff:r.event.start,
  book:r.price.book,odds:r.price.american,dec:r.price.decimal,updatedAt:r.price.quoted_at,
  strength:r.devigged_edge==null?null:100*r.devigged_edge,qualification:'Model research · not a proven edge',
  projectionLabel:(finite(r.going_projection)?r.going_projection.toFixed(2)+' projected · ':'')+(100*r.going_probability).toFixed(1)+'% model',
  confidence:r.confidence_level+' sample confidence',why:r.why.join(' · '),risk:r.main_risk,
  parlay:r.parlay_fit.rule,chartStats:[{label:'Model win probability',value:r.going_probability*100,max:100},
   {label:'Devigged market probability',value:r.devigged_market_probability==null?null:r.devigged_market_probability*100,max:100},
   {label:'Push probability',value:r.push_probability*100,max:100}],
  evidence:[['Sample sizes',JSON.stringify(r.sample_sizes)],['Underlying model inputs',JSON.stringify(r.features)],
   ['Minor split input',JSON.stringify(r.split_overlap)],['Validation',r.validation_status],['Quote',r.price.quoted_at]],
  source:r.model_version,researchUrl:r.entity.id==='game'?null:'/yard/?player='+r.entity.id
 }));
}
root.GoingYardOpportunities={build,stats,fromModel};if(typeof module!=='undefined')module.exports=root.GoingYardOpportunities;
})(globalThis);
