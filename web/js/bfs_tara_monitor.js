import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// This visualizes the exact bias policy, NOT measured attention probabilities.
export function taraPenalty(dt, w, f, strength, cap) {
  return Math.min(cap, strength * (Math.max(0, Math.abs(dt) - w) / f) ** 2);
}
const panels = new Map();
api.addEventListener("bfs-tara-monitor", ({ detail }) => {
  const panel = panels.get(String(detail.node));
  if (panel) panel.update(detail);
});
app.registerExtension({
  name: "BFS.TARA.LivePolicyMonitor",
  async beforeRegisterNodeDef(type, data) {
    if (data.name !== "BFSMiniMaxH3NativeTemporal") return;
    const created = type.prototype.onNodeCreated;
    type.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const node = this;
      const host = document.createElement("div");
      host.style.cssText = "background:#0b1020;color:#dce6ff;border:1px solid #293854;border-radius:12px;padding:12px;box-sizing:border-box;font:12px system-ui;width:100%;height:100%;overflow:hidden";
      host.innerHTML = `<div style="display:flex;justify-content:space-between"><b>TARA · Temporal field</b><span style="color:#65dfd5">POLICY VIEW</span></div><canvas style="width:100%;height:180px;display:block;margin-top:8px"></canvas><div class="ranges" style="line-height:1.6"></div><div class="status" style="margin-top:8px;color:#8de7e0">Ready · adjust parameters above</div><div style="font-size:10px;color:#97a5c1;margin-top:7px">Animated scan is illustrative. Colors show applied logit penalties, not measured attention or predicted video quality. Widget changes apply to the next queued run.</div>`;
      const canvas = host.querySelector("canvas"), ctx = canvas.getContext("2d");
      let state = {}, raf = 0, removed = false, last = 0, visible = true;
      const value = (name, fallback) => Number(node.widgets?.find(w => w.name === name)?.value ?? fallback);
      function draw(now) {
        if (removed || !visible) return;
        raf = requestAnimationFrame(draw);
        if (now - last < 65 || !host.isConnected || document.hidden) return;
        last = now;
        const W = Math.max(260, host.clientWidth - 24), H = 180;
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(W*dpr) || canvas.height !== H*dpr) { canvas.width=Math.round(W*dpr); canvas.height=H*dpr; }
        ctx.setTransform(dpr,0,0,dpr,0,0);
        ctx.clearRect(0,0,W,H);
        const w=value("window_seconds",.2), f=value("falloff_seconds",.25), s=value("strength",2), c=value("max_penalty",6);
        const active=s>0 && c>0, reach=active ? w+f*Math.sqrt(c/s) : Infinity;
        // Zoomed local temporal neighborhood, independent of the video duration.
        const span=Math.max(1, Math.min(20, Number.isFinite(reach) ? reach*1.35 : w+f*3));
        const x0=28, y0=12, width=W-40, height=116;
        for(let y=0;y<height;y+=3) for(let x=0;x<width;x+=3) {
          const dt=((x/width)-(y/height))*span*2;
          const penalty=taraPenalty(dt,w,f,s,c), ratio=c>0 ? penalty/c : 0;
          ctx.fillStyle=`rgb(${Math.round(23+ratio*57)},${Math.round(175-ratio*146)},${Math.round(167-ratio*72)})`;
          ctx.fillRect(x0+x,y0+y,3,3);
        }
        const reduced=window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
        const scan=reduced ? .5 : (now/6500)%1;
        ctx.strokeStyle="#fff9";ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(x0,y0+scan*height);ctx.lineTo(x0+width,y0+scan*height);ctx.stroke();
        ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(x0+scan*width,y0+scan*height,3,0,Math.PI*2);ctx.fill();
        ctx.fillStyle="#a9bad8";ctx.font="10px system-ui";
        ctx.fillText(`Guide time → · diagonal ±${span.toFixed(2)}s scope`,x0,145);
        ctx.save();ctx.translate(12,126);ctx.rotate(-Math.PI/2);ctx.fillText("Output time →",0,0);ctx.restore();
        ctx.fillStyle="#4fe3ce";ctx.fillText("● Free",x0,166);
        ctx.fillStyle="#8ea3d8";ctx.fillText("● Falloff",x0+65,166);
        ctx.fillStyle="#c38cb7";ctx.fillText("● Capped penalty",x0+145,166);
        host.querySelector(".ranges").textContent=active
          ? `Free ±${w.toFixed(2)}s · reaches cap at ±${reach.toFixed(2)}s · λ ${s.toFixed(1)} · cap ${c.toFixed(1)} (not a hard cutoff)`
          : "Temporal bias OFF · RoPE alignment remains active";
      }
      const panel={update(data){
        state={...state,...data};
        host.querySelector(".status").textContent=data.phase === "evaluating"
          ? `● Evaluation ${state.evaluation} running · ${state.motion_rows.toLocaleString()} motion tokens · ${state.backend}`
          : `✓ Evaluation ${state.evaluation} returned · ${state.elapsed_seconds.toFixed(2)}s wall time · waiting for sampler`;
      }};
      panels.set(String(node.id),panel);
      // IDs can change when restoring/copying a workflow.
      const added=node.onAdded;
      node.onAdded=function(){ const r=added?.apply(this,arguments); for(const [id,p] of panels) if(p===panel) panels.delete(id); panels.set(String(node.id),panel); return r; };
      node.properties ??= {};
      const toggle=node.addWidget("toggle", "Show TARA preview", true, value => setVisible(value),
                                  {serialize:false});
      const monitor=node.addDOMWidget("tara_policy_monitor","div",host,
          {serialize:false,hideOnZoom:false,getMinHeight:()=>visible ? 300 : 0,
           getMaxHeight:()=>visible ? 300 : 0});
      const originalComputeSize=monitor.computeSize;
      monitor.computeSize=function(...args){
        return visible ? (originalComputeSize?.apply(this,args) ?? [390,300]) : [0,-4];
      };
      function setVisible(value, resize=true){
        visible=Boolean(value);
        node.properties.bfs_tara_preview=visible;
        toggle.value=visible;
        host.style.display=visible ? "" : "none";
        monitor.hidden=!visible;
        cancelAnimationFrame(raf);
        if(visible && !removed) raf=requestAnimationFrame(draw);
        if(resize){
          const size=node.computeSize();
          node.setSize([Math.max(node.size[0],390),size[1]]);
          node.setDirtyCanvas(true,true);
        }
      }
      const configured=node.onConfigure;
      node.onConfigure=function(){
        const r=configured?.apply(this,arguments);
        for(const [id,p] of panels) if(p===panel) panels.delete(id);
        panels.set(String(node.id),panel);
        setVisible(node.properties.bfs_tara_preview !== false);
        return r;
      };
      node.setSize([Math.max(node.size[0],390),Math.max(node.size[1],610)]);
      const remove=node.onRemoved;
      node.onRemoved=function(){ removed=true;cancelAnimationFrame(raf);for(const [id,p] of panels) if(p===panel) panels.delete(id);return remove?.apply(this,arguments); };
      setVisible(node.properties.bfs_tara_preview !== false, false);
      return result;
    };
  }
});
