import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const url=f=>api.apiURL(`/view?${new URLSearchParams({filename:f.filename,subfolder:f.subfolder||"",type:f.type||"temp"})}`);
function player(host,selectable,node){
  host.style.cssText="width:100%;height:100%;box-sizing:border-box;background:#101925;color:#dce8ff;padding:10px;border-radius:10px;font:12px system-ui;overflow:auto";
  host.innerHTML=`<b>${selectable?"Select performer → Picture":"SAM 3.1 · tracked masks"}</b><div class="pictures" style="display:flex;gap:6px;overflow:auto;margin:8px 0"></div><div class="stage" style="position:relative"><video controls playsinline preload="auto" style="width:100%;display:block;background:#000"></video><canvas style="position:absolute;top:0;left:0;width:100%;height:calc(100% - 36px);pointer-events:none"></canvas></div><div class="tools" style="display:flex;gap:6px;margin:7px 0"><button class="back">−frame</button><button class="next">+frame</button><button class="clear">Clear selected Picture</button><button class="all">Reset all marks</button></div><div class="status">Queue once to load the video.</div>`;
  const video=host.querySelector("video"),canvas=host.querySelector("canvas"),ctx=canvas.getContext("2d"),stage=host.querySelector(".stage"),status=host.querySelector(".status");
  let data=null,picture=1,raf=0,removed=false;
  function widget(){return node.widgets?.find(w=>w.name==="selection_json");}
  function doc(){try{return JSON.parse(widget()?.value||"{}");}catch{return {};}}
  function save(d){const w=widget();if(!w)return;w.value=JSON.stringify(d);w.callback?.(w.value);node.setDirtyCanvas(true,true);}
  function blank(){return {source:data.video.source,bindings:[]};}
  function frame(){return Math.max(0,Math.min((data?.video.frames||1)-1,Math.floor(video.currentTime*24+1e-4)));}
  function draw(){
    if(removed)return;
    raf=requestAnimationFrame(draw);
    if(!selectable||!data||document.hidden)return;
    const w=video.clientWidth,h=video.videoWidth? w*video.videoHeight/video.videoWidth:200;
    canvas.style.height=`${h}px`;canvas.width=Math.max(1,Math.round(w));canvas.height=Math.max(1,Math.round(h));ctx.clearRect(0,0,w,h);
    ctx.fillStyle="#07131fbb";ctx.fillRect(5,5,190,24);ctx.fillStyle="#fff";ctx.font="12px system-ui";ctx.textAlign="left";ctx.textBaseline="middle";ctx.fillText(`Frame ${frame()} · Picture ${picture}`,11,17);
    for(const e of doc().bindings||[]){
      if(e.frame!==frame())continue;
      for(const [points,negative] of [[e.positive,false],[e.negative||[],true]])for(const p of points){
        const x=p.x*w,y=p.y*h;ctx.fillStyle=negative?"#ff637c":"#00d9ae";ctx.strokeStyle="#07131f";ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,13,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle="#07131f";ctx.font="bold 13px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(negative?"−":String(e.picture),x,y);
      }
    }
  }
  host.querySelector(".back").onclick=()=>{video.pause();video.currentTime=Math.max(0,(frame()-1)/24);};
  host.querySelector(".next").onclick=()=>{video.pause();video.currentTime=Math.min((data?.video.frames-1)/24,(frame()+1)/24);};
  host.querySelector(".clear").onclick=()=>{if(!data)return;const d=doc();d.bindings=(d.bindings||[]).filter(e=>e.picture!==picture);save(d);};
  host.querySelector(".all").onclick=()=>{if(data){save(blank());status.textContent="Marks cleared. Pause, choose Picture X, click the matching performer.";}};
  video.addEventListener("click",e=>{
    if(!selectable||!data||!video.videoWidth)return;
    const r=video.getBoundingClientRect();const actualH=r.width*video.videoHeight/video.videoWidth;
    if(e.clientY-r.top>actualH-36)return;
    video.pause();let d=doc();
    if(d.bindings?.length&&d.source!==data.video.source){status.textContent="Video changed. Reset all marks before selecting again.";return;}
    if(!d.source)d=blank();
    const pos={x:Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),y:Math.min(1,Math.max(0,(e.clientY-r.top)/actualH))};
    let binding=d.bindings.find(b=>b.picture===picture);const at=frame();
    if(!binding||binding.frame!==at){d.bindings=d.bindings.filter(b=>b.picture!==picture);binding={picture,frame:at,positive:[],negative:[]};d.bindings.push(binding);}
    if(e.shiftKey)binding.negative.push(pos);
    else if(e.ctrlKey||e.metaKey)binding.positive.push(pos);
    else binding.positive=[pos];
    save(d);status.textContent=`Picture ${picture} · anchor frame ${at} · ${binding.positive.length} positive / ${binding.negative.length} negative points. Turn setup_only off and queue when ready.`;
  });
  if(!selectable){host.querySelector(".tools").style.display="none";canvas.style.display="none";}
  raf=requestAnimationFrame(draw);
  return {
    disabled(message){data=null;video.pause();stage.style.opacity=".35";status.textContent=message;},
    update(next){data=next;stage.style.opacity="1";const current=url(next.video);if(video.dataset.file!==next.video.filename){video.src=current;video.dataset.file=next.video.filename;}
      if(selectable){const refs=host.querySelector(".pictures");refs.replaceChildren();for(const f of next.pictures){const button=document.createElement("button");button.style.cssText="background:#17273b;color:#dce8ff;border:1px solid #436184;border-radius:5px;padding:4px;cursor:pointer";const img=document.createElement("img");img.src=url(f);img.style.cssText="width:70px;height:65px;object-fit:contain;display:block";const text=document.createElement("div");text.textContent=`Picture ${f.picture}`;button.append(img,text);button.onclick=()=>{picture=f.picture;for(const b of refs.children)b.style.outline="none";button.style.outline="2px solid #42e4c1";status.textContent=`Picture ${picture} selected. Pause the video and click its source performer.`;};refs.append(button);}}
      status.textContent=next.message||next.stats?.map(s=>`Picture ${s.picture}: ${s.empty_frames} empty frames`).join(" · ")||"Tracked masks. Colors are preview only.";
    },destroy(){removed=true;cancelAnimationFrame(raf);video.pause();video.removeAttribute("src");video.load();}
  };
}
app.registerExtension({name:"BFS.H3.CastBindingSelector",async beforeRegisterNodeDef(type,data){
  const selectable=data.name==="BFSH3SelectCastBindings";
  if(!selectable&&data.name!=="BFSH3SAM31CastMasks")return;
  const created=type.prototype.onNodeCreated;
  type.prototype.onNodeCreated=function(){const r=created?.apply(this,arguments);const host=document.createElement("div");this._bfsCast=player(host,selectable,this);this.addDOMWidget("bfs_cast_player","div",host,{serialize:false,hideOnZoom:false,getMinHeight:()=>440});this.setSize([Math.max(this.size[0],520),Math.max(this.size[1],710)]);return r;};
  const executed=type.prototype.onExecuted;type.prototype.onExecuted=function(message){executed?.apply(this,arguments);const d=message?.[selectable?"bfs_cast_selector":"bfs_cast_preview"]?.[0];if(d)this._bfsCast?.update(d);const disabled=message?.bfs_cast_disabled?.[0];if(disabled)this._bfsCast?.disabled(disabled);};
  const removed=type.prototype.onRemoved;type.prototype.onRemoved=function(){this._bfsCast?.destroy();return removed?.apply(this,arguments);};
}});
