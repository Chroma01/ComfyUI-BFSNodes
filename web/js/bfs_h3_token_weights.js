import { app } from "../../scripts/app.js";

app.registerExtension({
  name:"BFS.H3.TokenWeightEditor",
  async beforeRegisterNodeDef(type,data){
    if(data.name!=="BFSH3TokenWeightEditor")return;
    const created=type.prototype.onNodeCreated;
    type.prototype.onNodeCreated=function(){
      const result=created?.apply(this,arguments),node=this;
      const host=document.createElement("div");
      host.style.cssText="width:100%;height:100%;box-sizing:border-box;padding:12px;background:#101725;color:#d9e6ff;border-radius:10px;font:12px system-ui;display:flex;flex-direction:column;gap:8px";
      host.innerHTML=`<b>H3 · Text token weights</b><div>Original tokenizer labels · supports (weighted phrases:1.3)</div><input class="search" placeholder="Filter token / word…" style="color:#eee;background:#1b2940;padding:5px;border:1px solid #375071"><div class="actions" style="display:flex;gap:6px;align-items:center"><button data-action="minus">−0.1</button><button data-action="plus">+0.1</button><button data-action="reset">Reset selected</button><button data-action="clear">Reset overrides</button></div><div class="selection">Click a token; Ctrl/Cmd-click to select several.</div><div class="tokens" style="flex:1;overflow:auto;min-height:150px;display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start"></div><div class="status" style="color:#91b2d9">Queue once in inspect-only mode to load tokens. Changes apply on the next queued run.</div>`;
      const box=host.querySelector(".tokens"),status=host.querySelector(".status"),selected=new Set();let view=null;
      const widget=()=>node.widgets?.find(w=>w.name==="weights_json");
      function doc(){try{return JSON.parse(widget().value||"{}");}catch{return {};}}
      function save(d){const w=widget();w.value=JSON.stringify(d);w.callback?.(w.value);node.setDirtyCanvas(true,true);}
      function render(){
        if(!view)return;
        const d=doc(),valid=!d.prompt_sha256||d.prompt_sha256===view.prompt_sha256;
        const filter=host.querySelector(".search").value.toLowerCase();box.replaceChildren();
        const max=Math.max(1,...view.tokens.map(t=>t.norm));
        for(const token of view.tokens){
          if(filter&&!`${token.label} ${token.word}`.toLowerCase().includes(filter))continue;
          const gain=valid ? Number(d.weights?.[token.index]??token.base_gain??1) : Number(token.base_gain??1);
          const button=document.createElement("button");button.type="button";
          button.textContent=`${token.label.replaceAll("\n","↵") || "∅"} ×${gain.toFixed(2)}`;
          button.title=`Token ${token.index} · ID ${token.token_id}\nOriginal word/fragment: ${token.word}\nEmbedding norm: ${token.norm.toFixed(3)}\nNorm is not attention or importance. Double-click selects this word's occurrences.`;
          button.style.cssText=`font:12px ui-monospace,monospace;white-space:pre-wrap;padding:6px;border-radius:5px;border:1px solid ${selected.has(token.index)?"#fff":gain>1?"#53c9b2":gain<1?"#df9d68":"#304769"};background:linear-gradient(to top,#233d58 ${token.norm/max*100}%,#172235 0);color:${gain>1?"#9ff2d9":gain<1?"#ffca9c":"#cbd9eb"};cursor:pointer`;
          button.onclick=e=>{if(!e.ctrlKey&&!e.metaKey)selected.clear();if(selected.has(token.index))selected.delete(token.index);else selected.add(token.index);render();};
          button.ondblclick=()=>{selected.clear();for(const t of view.tokens)if(t.word===token.word)selected.add(t.index);render();};box.append(button);
        }
        host.querySelector(".selection").textContent=`${selected.size} selected · click tokens; double-click selects matching word fragments`;
      }
      host.querySelector(".search").oninput=render;
      host.querySelector(".actions").onclick=e=>{
        const action=e.target.dataset.action;if(!action||!view)return;
        let d=doc();
        if(action==="clear"){d={prompt_sha256:view.prompt_sha256,weights:{}};selected.clear();}
        else{
          if(d.prompt_sha256&&d.prompt_sha256!==view.prompt_sha256){status.textContent="Prompt changed. Reset all before weighting the new text.";return;}
          d={...d,prompt_sha256:view.prompt_sha256,weights:{...d.weights}};
          for(const i of selected){
            const base=Number(view.tokens.find(t=>t.index===i)?.base_gain??1);
            const gain=action==="reset"?base:Math.min(3,Math.max(.05,Math.round((Number(d.weights[i]??base)+(action==="plus"?.1:-.1))*100)/100));
            if(gain===base)delete d.weights[i];else d.weights[i]=gain;
          }
        }
        save(d);status.textContent="Weights changed. Turn inspect_only off and queue to test. Other parameters and seed should stay fixed.";render();
      };
      this._bfsTokenPanel={update(d){view=d;selected.clear();status.textContent=d.message;render();}};
      this.addDOMWidget("bfs_h3_token_panel","div",host,{serialize:false,hideOnZoom:false,getMinHeight:()=>370});
      this.setSize([Math.max(this.size[0],550),Math.max(this.size[1],660)]);
      return result;
    };
    const executed=type.prototype.onExecuted;
    type.prototype.onExecuted=function(message){executed?.apply(this,arguments);const d=message?.bfs_h3_tokens?.[0];if(d)this._bfsTokenPanel?.update(d);};
  }
});
