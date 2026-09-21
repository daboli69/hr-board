/* Shared GOING brand and cross-sport navigation. No betting state lives here. */
(()=>{
 const script=document.currentScript,workspace=script?.dataset.workspace||'home',base=script?.dataset.base||'';
 if(document.getElementById('going-shell'))return;
 document.body.dataset.goingWorkspace=workspace;
 const assetRoot=new URL('.',script.src);
 for(const name of ['design-tokens.css','consumer-ui.css']){const link=document.createElement('link');link.rel='stylesheet';link.href=new URL(name,assetRoot).href;document.head.append(link);}
 const names={home:'',long:'LONG',yard:'YARD',results:'Results',roster:'Players'};
 const shell=document.createElement('div');shell.id='going-shell';shell.setAttribute('role','banner');
 const links=[['home','Home','/'],['long','Football','/long/?mode=betting'],['yard','Baseball','/yard/?mode=betting'],['roster','Players','/players/'],['results','Results','/results/']];
 shell.innerHTML=`<div class="going-shell-inner"><a class="going-wordmark" href="${base}/" aria-label="GOING home"><span>GOING</span>${names[workspace]?`<span class="going-wordmark-sport">${names[workspace]}</span>`:''}<i aria-hidden="true"></i></a><nav class="going-global-nav" aria-label="GOING navigation">${links.map(([key,label,url])=>`<a href="${base}${url}"${workspace===key?' aria-current="page"':''}>${label}</a>`).join('')}</nav></div>`;
 document.body.prepend(shell);
 // Existing sticky controls need the actual shared-header height, including zoom.
 const measure=()=>document.documentElement.style.setProperty('--going-header-height',`${shell.getBoundingClientRect().height}px`);
 measure();if(typeof ResizeObserver!=='undefined')new ResizeObserver(measure).observe(shell);
 // Keep dynamically rendered controls in title case, preserving sport acronyms.
 const acronyms=new Set(['NFL','NCAA','MLB','DFS','TD','HR','RBI','RB','WR','TE','QB','DST','FBS','CSV','EV','IP','PA','BF','GOING']);
 const casing=node=>{for(const el of node.querySelectorAll?.('button,summary,[role=button]')||[]){const walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);let text;while(text=walker.nextNode()){if(text.parentElement.closest('svg,script,style'))continue;const value=text.nodeValue;if(/[A-Z]{3}/.test(value)&&value===value.toUpperCase())text.nodeValue=value.replace(/[A-Z]+/g,w=>acronyms.has(w)?w:w[0]+w.slice(1).toLowerCase());}}};
 casing(document);let scheduled=false;new MutationObserver(()=>{if(scheduled)return;scheduled=true;requestAnimationFrame(()=>{scheduled=false;casing(document);});}).observe(document.body,{childList:true,subtree:true});
})();
