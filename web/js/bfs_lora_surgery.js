/**
 * BFS LoRA Surgery: a panel for reading a LoRA's layers and blocks and editing them live.
 *
 * The node keeps its state in a hidden `rules` STRING widget (a JSON list), so a workflow
 * saves and reloads exactly what you set up. Everything visual lives in a Vue app mounted
 * into a DOM widget.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createApp, ref, computed, watch, h } from "./vendor/vue.esm-browser.prod.mjs";

// Starting points, not answers: which range works is per-LoRA, so ablate before trusting one.
const PRESETS = [
  { label: "MLP 50%", rules: [{ enabled: true, match: { type: "*mlp*" }, scale: 0.5 }] },
  { label: "MLP 35%", rules: [{ enabled: true, match: { type: "*mlp*" }, scale: 0.35 }] },
  { label: "Attention only", rules: [{ enabled: true, match: { type: "*mlp*" }, scale: 0 }] },
  { label: "MLP-in off, late half", rules: [{ enabled: true, match: { type: "*gate_up*" }, blocksHalf: true, scale: 0 }] },
];

function styles() {
  if (document.getElementById("bfs-lora-surgery-css")) return;
  const el = document.createElement("style");
  el.id = "bfs-lora-surgery-css";
  el.textContent = `
.bfsls{font:12px/1.45 var(--font-family,system-ui,sans-serif);color:var(--descrip-text,#c9c9cf);
  background:var(--comfy-input-bg,#1b1b1f);border-radius:8px;padding:8px;height:100%;overflow:auto;box-sizing:border-box}
.bfsls h4{margin:0 0 6px;font-size:12px;letter-spacing:.02em;color:var(--input-text,#e6e6ea)}
.bfsls .muted{color:#8b8b93}
.bfsls .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.bfsls .fam{border:1px solid #33333c;border-radius:6px;margin-bottom:6px;overflow:hidden}
.bfsls .fam>summary{cursor:pointer;padding:6px 8px;background:#232329;list-style:none;display:flex;
  justify-content:space-between;align-items:center;user-select:none}
.bfsls .fam>summary::-webkit-details-marker{display:none}
.bfsls .type{display:grid;grid-template-columns:1fr 78px 54px 30px;gap:6px;align-items:center;
  padding:4px 8px;border-top:1px solid #2a2a31}
.bfsls .type:hover{background:#26262c}
.bfsls .name{font-family:ui-monospace,monospace;font-size:11px;color:#dcdce2;overflow:hidden;text-overflow:ellipsis}
.bfsls .bar{height:3px;border-radius:2px;background:#3a86ff;margin-top:3px;opacity:.85}
.bfsls input[type=range]{width:100%;accent-color:#3a86ff}
.bfsls input[type=text],.bfsls input[type=number]{background:#131317;border:1px solid #34343d;color:#e6e6ea;
  border-radius:4px;padding:3px 6px;font-size:11px;width:100%;box-sizing:border-box;font-family:ui-monospace,monospace}
.bfsls button{background:#2b2b33;border:1px solid #3a3a45;color:#dcdce2;border-radius:5px;padding:3px 8px;
  font-size:11px;cursor:pointer}
.bfsls button:hover{background:#35353f}
.bfsls button.pri{background:#2b5fd9;border-color:#3a6ee8;color:#fff}
.bfsls .rule{display:grid;grid-template-columns:18px 1fr 64px 18px 18px 20px;gap:5px;align-items:center;
  padding:4px 6px;border:1px solid #33333c;border-radius:5px;margin-bottom:4px;background:#1f1f25}
.bfsls .rx{font-family:ui-monospace,monospace;font-size:10px;color:#9fd3a0;word-break:break-all}
.bfsls .blocks{display:flex;flex-wrap:wrap;gap:2px;margin:4px 0}
.bfsls .blk{width:19px;height:19px;user-select:none;touch-action:none;border-radius:3px;background:#2a2a32;border:1px solid #3a3a45;
  font-size:9px;display:flex;align-items:center;justify-content:center;cursor:pointer;color:#a8a8b2}
.bfsls .blk.sel{background:#2b5fd9;border-color:#4a7ef0;color:#fff}
.bfsls .err{color:#ff8a8a}
.bfsls .sec{border-top:1px solid #2c2c34;margin-top:8px;padding-top:6px}
.bfsls details.help{border:1px solid #33333c;border-radius:6px;background:#1c1c22;margin-bottom:8px}
.bfsls details.help>summary{cursor:pointer;padding:5px 8px;color:#9fb8ff;list-style:none;user-select:none}
.bfsls details.help>summary::-webkit-details-marker{display:none}
.bfsls .helpbody{padding:2px 10px 8px;color:#b4b4bd;font-size:11px;line-height:1.55}
.bfsls .helpbody code{background:#262630;padding:1px 4px;border-radius:3px;font-size:10px}
.bfsls .helpbody b{color:#dcdce2}
.bfsls .helpbody ol{margin:4px 0 6px 16px;padding:0}
.bfsls .helpbody li{margin:2px 0}
`;
  document.head.appendChild(el);
}

function Panel(props) {
  const struct = ref(null);
  const error = ref("");
  const loading = ref(false);
  const rules = ref([]);
  const selFamily = ref("");
  const selType = ref("*");
  const selBlocks = ref(new Set());
  const scale = ref(0);
  const dragging = ref(false);
  let dragAdds = true;

  function readRules() {
    try {
      const parsed = JSON.parse(props.getRules() || "[]");
      rules.value = Array.isArray(parsed) ? parsed : [];
    } catch { rules.value = []; }
  }
  function writeRules() { props.setRules(JSON.stringify(rules.value)); }

  async function load() {
    const name = props.getLora();
    if (!name) return;
    loading.value = true; error.value = "";
    try {
      const r = await api.fetchApi(`/bfs/lora/structure?name=${encodeURIComponent(name)}`);
      const data = await r.json();
      if (data.error) { error.value = data.error; struct.value = null; }
      else { struct.value = data; selFamily.value = data.families?.[0]?.name || ""; }
    } catch (e) { error.value = String(e); }
    loading.value = false;
  }

  const family = computed(() => (struct.value?.families || []).find(f => f.name === selFamily.value)
                                || struct.value?.families?.[0]);
  const maxNorm = computed(() => Math.max(0.0001, ...((family.value?.types || []).map(t => t.norm || 0))));

  function blocksSpec() {
    const b = [...selBlocks.value].sort((x, y) => x - y);
    if (!b.length) return "";
    const out = []; let s = b[0], p = b[0];
    for (let i = 1; i <= b.length; i++) {
      if (b[i] === p + 1) { p = b[i]; continue; }
      out.push(s === p ? `${s}` : `${s}-${p}`);
      s = p = b[i];
    }
    return out.join(",");
  }
  // Drag across the ruler to pick a range. Click-to-toggle still works, since a click is
  // just a drag of one cell. Shift is deliberately not used: on the ComfyUI canvas it starts
  // a text selection instead.
  function paint(i) {
    if (dragAdds) selBlocks.value.add(i); else selBlocks.value.delete(i);
    selBlocks.value = new Set(selBlocks.value);
  }
  function dragStart(i, ev) {
    ev.preventDefault(); ev.stopPropagation();
    dragging.value = true;
    dragAdds = !selBlocks.value.has(i);
    paint(i);
    const stop = () => { dragging.value = false; window.removeEventListener("pointerup", stop); };
    window.addEventListener("pointerup", stop);
  }
  function dragOver(i, ev) {
    if (!dragging.value) return;
    ev.preventDefault();
    paint(i);
  }
  function setBlocksFromText(text) {
    const next = new Set();
    for (const part of String(text).split(",")) {
      const t = part.trim();
      if (!t) continue;
      if (t.includes("-")) {
        const [a, b] = t.split("-").map(Number);
        if (Number.isFinite(a) && Number.isFinite(b)) for (let k = Math.min(a, b); k <= Math.max(a, b); k++) next.add(k);
      } else if (Number.isFinite(Number(t))) next.add(Number(t));
    }
    selBlocks.value = next;
  }
  function selectAll() { selBlocks.value = new Set(family.value?.blocks || []); }
  function selectHalf(second) {
    const b = family.value?.blocks || [];
    const mid = Math.floor(b.length / 2);
    selBlocks.value = new Set(second ? b.slice(mid) : b.slice(0, mid));
  }
  function invert() {
    const all = family.value?.blocks || [];
    selBlocks.value = new Set(all.filter(i => !selBlocks.value.has(i)));
  }

  function addRule(asRegex) {
    const fam = family.value?.name || "";
    const spec = blocksSpec();
    const match = asRegex
      ? { regex: buildRegex(fam, selType.value, spec) }
      : { family: fam, type: selType.value, blocks: spec };
    rules.value.push({ enabled: true, match, scale: Number(scale.value) });
    writeRules();
  }
  function buildRegex(fam, type, spec) {
    const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    let blk = "\\d+";
    if (spec) {
      const idx = [];
      for (const part of spec.split(",")) {
        if (part.includes("-")) { const [a, b] = part.split("-").map(Number); for (let k = a; k <= b; k++) idx.push(k); }
        else if (part.trim() !== "") idx.push(Number(part));
      }
      if (idx.length) blk = "(" + [...new Set(idx)].sort((a, b) => a - b).join("|") + ")";
    }
    let mod = ".*";
    if (type && type !== "*") mod = type.endsWith("*") ? esc(type.slice(0, -1)) + ".*" : esc(type);
    return `^.*${fam ? esc(fam) : "[\\w.]+"}\\.${blk}\\.${mod}$`;
  }

  readRules();
  load();
  watch(() => props.getLora(), load);


  const help = () => h("details", { class: "help" }, [
    h("summary", {}, "What is this? (read once)"),
    h("div", { class: "helpbody" }, [
      h("p", {}, [
        "A defect a LoRA learned (soft skin, identity drifting when the subject is far away, ",
        "a pose it will not copy) usually lives in ", h("b", {}, "one module family and a range of blocks"),
        ", not in the whole adapter. This node lets you scale or drop that group and generate again, ",
        "so you can find where it lives without retraining. Nothing is written to disk.",
      ]),
      h("p", {}, [h("b", {}, "How to use it")]),
      h("ol", {}, [
        h("li", {}, "Pick a LoRA. The Layers list below shows every family, its block count and its module types."),
        h("li", {}, ["The bar next to each type is ", h("code", {}, "||dW||"),
                     ", how large that group's update is. Big bars are where training invested."]),
        h("li", {}, ["Click ", h("code", {}, "->"), " on a type to select it, then drag across the block ruler to pick a range."]),
        h("li", {}, ["Set a scale and press ", h("code", {}, "add rule"), ". Scale 0 drops the group; 1 leaves it as trained."]),
        h("li", {}, "Queue the prompt and compare against the unmodified LoRA at the same seed."),
      ]),
      h("p", {}, [h("b", {}, "Where to look first. "),
        "The MLP input projection (", h("code", {}, "gate_up"), ", ", h("code", {}, "w1"), "/", h("code", {}, "w3"),
        ", ", h("code", {}, "mlp.gate"), "/", h("code", {}, "mlp.up"),
        ") is the usual culprit for blur and lost detail. The MLP output projection rarely matters, ",
        "and attention is usually innocent. Which block range is the right one differs per LoRA, so test."]),
      h("p", {}, [h("b", {}, "Order matters. "),
        "Later rules override earlier ones where they overlap, so put boosts first and drops last, ",
        "or a boost will undo a drop. Use the arrows to reorder."]),
      h("p", {}, [h("b", {}, "One warning. "),
        "A variant can win every metric and still have destroyed what you trained. Dropping the whole MLP ",
        "path gave the sharpest skin in one real case and made the LoRA stop copying expression from the ",
        "source image. Always check a hard case (a strong expression, an unusual angle) with your eyes."]),
      h("p", {}, [h("b", {}, "Pruning is not the same as training without those layers. "),
        "Excluding them during training may simply fail to converge: the layer is where the defect lodges, ",
        "not where it comes from, which is usually the dataset."]),
    ]),
  ]);

  return () => h("div", { class: "bfsls" }, [
    h("div", { class: "row", style: "justify-content:space-between;margin-bottom:6px" }, [
      h("h4", {}, struct.value ? `${struct.value.file}, ${struct.value.total_modules} modules`
                               : (loading.value ? "reading..." : "no LoRA loaded")),
      h("button", { onClick: load }, "reload"),
    ]),
    error.value ? h("div", { class: "err" }, error.value) : null,
    help(),

    h("div", { class: "sec" }, [h("h4", {}, "Selection")]),
    h("div", { class: "row" }, [
      h("span", { class: "muted" }, "type"),
      h("input", { type: "text", value: selType.value, style: "flex:1",
                   onInput: e => { selType.value = e.target.value; } }),
    ]),
    h("div", { class: "blocks", onPointerleave: () => { dragging.value = false; } },
      (family.value?.blocks || []).map(i =>
        h("div", { class: "blk" + (selBlocks.value.has(i) ? " sel" : ""),
                   title: "click, or drag across to select a range",
                   onPointerdown: e => dragStart(i, e),
                   onPointerenter: e => dragOver(i, e) }, String(i)))),
    h("div", { class: "row", style: "margin-bottom:4px" }, [
      h("span", { class: "muted" }, "blocks"),
      h("input", { type: "text", value: blocksSpec(), placeholder: "all blocks", style: "flex:1",
                   title: "type a selection, e.g. 8-15 or 0,4,7 or 8-15,24-31",
                   onChange: e => setBlocksFromText(e.target.value) }),
    ]),
    h("div", { class: "row", style: "margin-bottom:4px" }, [
      h("button", { onClick: selectAll }, "all"),
      h("button", { onClick: () => selectHalf(false) }, "first half"),
      h("button", { onClick: () => selectHalf(true) }, "last half"),
      h("button", { onClick: invert }, "invert"),
      h("button", { onClick: () => { selBlocks.value = new Set(); } }, "none"),
    ]),
    h("div", { class: "row" }, [
      h("span", { class: "muted" }, "scale"),
      h("input", { type: "range", min: 0, max: 2, step: 0.05, value: scale.value, style: "flex:1",
                   onInput: e => { scale.value = e.target.value; } }),
      h("span", { style: "width:34px;text-align:right" }, Number(scale.value).toFixed(2)),
    ]),
    h("div", { class: "row", style: "margin:6px 0" }, [
      h("button", { class: "pri", onClick: () => addRule(false) }, "add rule"),
      h("button", { onClick: () => addRule(true), title: "same selection, written as a regex" }, "add as regex"),
      h("span", { class: "muted" }, blocksSpec() ? `blocks ${blocksSpec()}` : "all blocks"),
    ]),
    h("div", { class: "row", style: "margin-bottom:6px" }, PRESETS.map(p =>
      h("button", { onClick: () => { rules.value = JSON.parse(JSON.stringify(p.rules)); writeRules(); } }, p.label))),

    h("div", { class: "sec" }, [h("h4", {}, `Rules (${rules.value.length}), later rules override earlier ones`)]),
    rules.value.length > 1
      ? h("div", { class: "muted", style: "margin-bottom:4px" },
          "Order matters: put boosts first and drops last, or a boost will undo a drop.")
      : null,
    ...rules.value.map((r, i) => h("div", { class: "rule" }, [
      h("input", { type: "checkbox", checked: r.enabled !== false,
                   onChange: e => { r.enabled = e.target.checked; writeRules(); } }),
      r.match.regex
        ? h("div", { class: "rx", title: r.match.regex }, r.match.regex)
        : h("div", { class: "name" }, `${r.match.type || "*"} @ ${r.match.blocks || "all"}`),
      h("input", { type: "number", step: 0.05, value: r.scale,
                   onChange: e => { r.scale = Number(e.target.value); writeRules(); } }),
      h("button", { title: "move up, later rules win", disabled: i === 0,
                    onClick: () => { const a = rules.value; [a[i - 1], a[i]] = [a[i], a[i - 1]]; writeRules(); } }, "↑"),
      h("button", { title: "move down, later rules win", disabled: i === rules.value.length - 1,
                    onClick: () => { const a = rules.value; [a[i + 1], a[i]] = [a[i], a[i + 1]]; writeRules(); } }, "↓"),
      h("button", { onClick: () => { rules.value.splice(i, 1); writeRules(); } }, "×"),
    ])),
    rules.value.length
      ? h("div", { class: "row", style: "margin-top:4px" },
          [h("button", { onClick: () => { rules.value = []; writeRules(); } }, "clear all")])
      : h("div", { class: "muted" }, "No rules, the LoRA is applied unchanged."),

    h("div", { class: "sec" }, [h("h4", {}, "Layers")]),
    ...(struct.value?.families || []).map(fam =>
      h("details", { class: "fam", open: fam.name === selFamily.value }, [
        h("summary", { onClick: () => { selFamily.value = fam.name; } }, [
          h("span", { class: "name" }, fam.name || "(root)"),
          h("span", { class: "muted" }, `${fam.blocks.length} blocks · ${fam.types.length} types`),
        ]),
        ...fam.types.map(t => h("div", { class: "type" }, [
          h("div", {}, [
            h("div", { class: "name", title: t.name }, t.name || "(none)"),
            h("div", { class: "bar", style: `width:${Math.round(100 * (t.norm || 0) / maxNorm.value)}%` }),
          ]),
          h("span", { class: "muted" }, t.norm != null ? `||dW|| ${t.norm.toFixed(2)}` : ""),
          h("span", { class: "muted" }, `r${t.rank ?? "?"}`),
          h("button", { title: "select this type", onClick: () => { selFamily.value = fam.name; selType.value = t.name; } }, "->"),
        ])),
      ])),
  ]);
}

app.registerExtension({
  name: "BFSNodes.LoraSurgery",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSLoraSurgery") return;
    styles();
    const rulesWidget = node.widgets?.find(w => w.name === "rules");
    const loraWidget = node.widgets?.find(w => w.name === "lora_name");
    if (rulesWidget) { rulesWidget.type = "hidden"; rulesWidget.computeSize = () => [0, -4]; }

    const host = document.createElement("div");
    host.style.cssText = "width:100%;height:100%;min-height:340px";
    node.addDOMWidget("bfs_surgery_panel", "div", host, { serialize: false, hideOnZoom: false });

    createApp({
      render: Panel({
        getLora: () => loraWidget?.value,
        getRules: () => rulesWidget?.value ?? "[]",
        setRules: v => { if (rulesWidget) rulesWidget.value = v; node.graph?.setDirtyCanvas(true); },
      }),
    }).mount(host);

    node.size = [Math.max(node.size[0], 430), Math.max(node.size[1], 520)];
  },
});
