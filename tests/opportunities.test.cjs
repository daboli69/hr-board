const {test}=require('node:test');
const assert=require('node:assert/strict');
const {fromModel}=require('../docs/shared/yard-opportunities.js');
test('new static contract preserves every family, book, side and freshness',()=>{
 const now=Date.parse('2026-09-22T12:00:00Z');
 const row={id:'one',game_pk:1,event:{start:'2026-09-22T18:00:00Z',home:'A',away:'B'},entity:{id:'42',name:'Player'},
 market:'pk',market_line:5.5,side:'under',bet_description:'Player Under 5.5 Ks',price:{book:'fanatics',decimal:2.1,american:110,quoted_at:'2026-09-22T11:30:00Z'},
 going_projection:4.7,going_probability:.63,push_probability:0,devigged_market_probability:.48,confidence_level:'moderate',why:['Actual input','Actual workload'],
 main_risk:'Removal risk',parlay_fit:{rule:'Distinct games'},features:{},sample_sizes:{pitcher_bf:400},split_overlap:{relative_change:.001},freshness:'FRESH'};
 const payload={status:'FRESH',opportunities:[row]};
 const card=fromModel(payload,now)[0];
 assert.equal(card.side,'under');assert.equal(card.dec,2.1);assert.match(card.projectionLabel,/63.0%/);
 assert.equal(fromModel({...payload,status:'STALE'},now).length,0);
 assert.equal(fromModel(payload,now+4*3600000).length,0);
 assert.equal(fromModel({...payload,opportunities:[{...row,freshness:'STALE'}]},now).length,0);
});
