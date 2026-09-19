const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');
const html=fs.readFileSync('docs/index.html','utf8');
const code=html.split('// SPLIT_OVERLAPS_START')[1].split('// SPLIT_OVERLAPS_END')[0];
const context={localStorage:{getItem:()=>null,setItem:()=>{}},console};vm.createContext(context);
vm.runInContext('// SPLIT_OVERLAPS_START'+code,context);
const rows=[{id:1,game_pk:1,overlap_strength:100,batter_stats:{pa:50,iso:.2},pitcher_stats:{bf:30,hr9:2}},
{id:2,game_pk:1,overlap_strength:80,batter_stats:{pa:100,iso:.3},pitcher_stats:{bf:50,hr9:1}},
{id:3,game_pk:1,overlap_strength:200,batter_stats:{pa:49,iso:.4},pitcher_stats:{bf:100,hr9:3}}];
test('filter before sort, both sides toggle, strength restores, no source mutation',()=>{
const visible=(key,dir)=>context.splitVisible(rows,{pa:50,bf:30},key,dir).map(r=>r.id).join(',');
assert.equal(visible('overlap_strength',-1),'1,2');assert.equal(visible('batter_stats.iso',-1),'2,1');assert.equal(visible('batter_stats.iso',1),'1,2');assert.equal(visible('pitcher_stats.hr9',-1),'1,2');assert.equal(rows.length,3);
});
test('thresholds are arbitrary and nulls sort last in both directions',()=>{
assert.equal(context.splitVisible(rows,{pa:0,bf:0}).length,3);
const missing={...rows[0],id:4,batter_stats:{pa:100,iso:null}};
for(const dir of [-1,1])assert.equal(context.splitVisible([...rows,missing],{pa:50,bf:30},'batter_stats.iso',dir).at(-1).id,4);
});
test('entire page scripts parse',()=>{for(const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g))new vm.Script(match[1]);});
