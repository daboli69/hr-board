/* Shared GOING brand and cross-sport navigation. No betting state lives here. */
(()=>{
 const script=document.currentScript,workspace=script?.dataset.workspace||'home',base=script?.dataset.base||'';
 if(document.getElementById('going-shell'))return;
 document.body.dataset.goingWorkspace=workspace;
 const names={home:'',long:'LONG',yard:'YARD',results:'RESULTS'};
 const shell=document.createElement('div');shell.id='going-shell';shell.setAttribute('role','banner');
 const links=[['home','Home','/'],['long','Football','/long/?mode=betting'],['yard','Baseball','/yard/?mode=betting'],['results','Results','/validation/']];
 shell.innerHTML=`<div class="going-shell-inner"><a class="going-wordmark" href="${base}/" aria-label="GOING home"><span>GOING</span>${names[workspace]?`<span class="going-wordmark-sport">${names[workspace]}</span>`:''}<i aria-hidden="true"></i></a><nav class="going-global-nav" aria-label="GOING navigation">${links.map(([key,label,url])=>`<a href="${base}${url}"${workspace===key?' aria-current="page"':''}>${label}</a>`).join('')}</nav></div>`;
 document.body.prepend(shell);
 // Existing sticky controls need the actual shared-header height, including zoom.
 const measure=()=>document.documentElement.style.setProperty('--going-header-height',`${shell.getBoundingClientRect().height}px`);
 measure();if(typeof ResizeObserver!=='undefined')new ResizeObserver(measure).observe(shell);
})();
