/**
 * BFS Shot Planner: split a long video into model-sized shots on a timeline, give each shot its own
 * reference image and prompt, and run the rest of the graph once per shot.
 *
 * All state lives in the hidden `plan` STRING widget (JSON) so a workflow saves, reloads and shares
 * exactly what you set up. The panel is a Vue app mounted into a DOM widget; the server does the video
 * work (probe, thumbnails, PySceneDetect cuts, split) through /bfs/shotloop/* routes.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createApp, ref, reactive, computed, onMounted, onBeforeUnmount, h } from "./vendor/vue.esm-browser.prod.mjs";

const GRIDS = { "H3 (17n+5)": [17, 5], "LTX / Wan (8n+1)": [8, 1], "Wan (4n+1)": [4, 1], "any": [1, 0] };
const DEFAULTS = {
  video: "", fps: 24, grid: "H3 (17n+5)", mode: "shots", max_s: 4.5, min_s: 1.0, sensitivity: 0.5,
  max_parts: 0, max_total_s: 0, bounds: [], segs: [], global_ref: "", global_ref2: "", global_prompt: "",
  megapixels: 0.15, multiple: 32, detector: "adaptive", run: "auto", auto_continue: true,
  skip_fill: "original",
  filters: { person: false, min_person_area: 0, max_persons: 0, face: false, skip_dark: false, dark_level: 0.06,
             skip_static: false, static_level: 0.004, min_frames: 0, samples: 6 },
};

const snapUp = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.ceil((n - o) / s);
};
const snapDown = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.floor((n - o) / s);
};
const hue = i => `hsl(${(i * 47 + 200) % 360} 55% 46%)`;
const viewUrl = name => {
  if (!name) return "";
  const i = name.lastIndexOf("/");
  const sub = i >= 0 ? name.slice(0, i) : "", file = i >= 0 ? name.slice(i + 1) : name;
  return api.apiURL(`/view?filename=${encodeURIComponent(file)}&subfolder=${encodeURIComponent(sub)}&type=input`);
};
const fmtT = (f, fps) => { const s = f / fps; return `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`; };

function styles() {
  if (document.getElementById("bfs-shotloop-css")) return;
  const el = document.createElement("style");
  el.id = "bfs-shotloop-css";
  el.textContent = `
.bsl{font:12px/1.45 var(--font-family,system-ui,sans-serif);color:#c9c9cf;background:#17171b;border-radius:10px;
  height:100%;overflow:auto;box-sizing:border-box;padding:10px;position:relative}
.bsl *{box-sizing:border-box}
.bsl .hdr{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.bsl .ttl{font-weight:600;color:#ececf1;font-size:13px;letter-spacing:.01em}
.bsl .pill{font-size:10px;padding:1px 7px;border-radius:999px;background:#25252d;color:#a9a9b4;border:1px solid #33333d}
.bsl .pill.ok{background:#173527;color:#7fe0a8;border-color:#245c40}
.bsl .pill.warn{background:#3a2a12;color:#ffc46b;border-color:#6a4a16}
.bsl .grow{flex:1}
.bsl select,.bsl input[type=number],.bsl input[type=text],.bsl textarea{background:#101014;border:1px solid #33333d;color:#e6e6ea;
  border-radius:6px;padding:4px 7px;font-size:11px;font-family:inherit;width:100%}
.bsl textarea{resize:vertical;min-height:54px;font-family:ui-monospace,monospace}
.bsl input[type=range]{width:100%;accent-color:#5b8cff}
.bsl button{background:#26262e;border:1px solid #393945;color:#dcdce2;border-radius:6px;padding:4px 9px;font-size:11px;cursor:pointer;white-space:nowrap}
.bsl button:hover{background:#30303a}
.bsl button.pri{background:#3a63e0;border-color:#4b74f0;color:#fff}
.bsl button.pri:hover{background:#4672ee}
.bsl button.dng{background:#3a1f22;border-color:#6a2c33;color:#ffb3b3}
.bsl button:disabled{opacity:.45;cursor:default}
.bsl .card{background:#1d1d23;border:1px solid #2c2c35;border-radius:9px;padding:8px;margin-bottom:8px}
.bsl .card h5{margin:0 0 6px;font-size:11px;font-weight:600;color:#b9b9c4;text-transform:uppercase;letter-spacing:.06em;display:flex;gap:6px;align-items:center}
.bsl .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px 8px}
.bsl .fld label{display:block;font-size:10px;color:#8c8c97;margin-bottom:2px}
.bsl .hint{font-size:10px;color:#7d7d88}
.bsl .err{color:#ff8f8f;background:#2a1517;border:1px solid #5a2228;border-radius:6px;padding:5px 8px;margin-bottom:8px}
.bsl .tl{position:relative;overflow-x:auto;overflow-y:hidden;border-radius:7px;background:#111115;border:1px solid #2a2a33}
.bsl .tlin{position:relative}
.bsl .strip{display:flex;height:64px;overflow:hidden}
.bsl .strip img{height:64px;object-fit:cover;flex:none;opacity:.92}
.bsl .spark{display:block;height:26px;width:100%}
.bsl .segs{position:relative;height:30px;margin-top:2px;cursor:crosshair}
.bsl .seg{position:absolute;top:2px;bottom:2px;border-radius:5px;display:flex;align-items:center;justify-content:center;
  font-size:10px;color:#fff;font-weight:600;overflow:hidden;cursor:pointer;border:2px solid transparent;user-select:none}
.bsl .seg.sel{border-color:#fff;box-shadow:0 0 0 2px #5b8cff66}
.bsl .seg.off{opacity:.35;background-image:repeating-linear-gradient(45deg,#0004 0 6px,#0000 6px 12px)}
.bsl .seg.long{outline:2px solid #ff5d5d;outline-offset:-2px}
.bsl .hdl{position:absolute;top:-70px;bottom:0;width:9px;margin-left:-4px;cursor:ew-resize;z-index:3}
.bsl .hdl::after{content:"";position:absolute;left:3px;top:0;bottom:0;width:3px;background:#fff;opacity:.85;border-radius:2px;box-shadow:0 0 4px #000}
.bsl .hdl:hover::after{background:#ffd34d}
.bsl .cut{position:absolute;top:0;height:64px;width:2px;background:#ff4d6d;opacity:.9;pointer-events:none}
.bsl .ph{position:absolute;top:0;bottom:0;width:1px;background:#ffd34d;pointer-events:none;z-index:4}
.bsl .tip{position:absolute;z-index:6;pointer-events:none;background:#0d0d10ee;border:1px solid #3a3a44;border-radius:6px;padding:3px;font-size:10px;color:#ddd}
.bsl .tip img{display:block;height:84px;border-radius:4px}
.bsl .shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
.bsl .sc{background:#202027;border:1px solid #30303a;border-radius:8px;padding:6px;cursor:pointer;position:relative}
.bsl .sc:hover{border-color:#4a4a58}
.bsl .sc.sel{border-color:#5b8cff;box-shadow:0 0 0 1px #5b8cff}
.bsl .sc .bar{height:3px;border-radius:2px;margin-bottom:5px}
.bsl .sc .t{font-size:10px;color:#9a9aa6}
.bsl .sc .p{font-size:10px;color:#c9c9d3;margin-top:3px;height:28px;overflow:hidden}
.bsl .thumbs{display:flex;gap:4px;margin-top:4px}
.bsl .rt{width:34px;height:34px;border-radius:5px;object-fit:cover;background:#2a2a33;border:1px solid #3a3a45}
.bsl .rt.ph2{display:flex;align-items:center;justify-content:center;color:#666;font-size:9px}
.bsl .refbox{display:flex;gap:8px;align-items:center}
.bsl .refslot{width:78px;height:78px;border-radius:8px;border:1px dashed #44444f;background:#141418;display:flex;align-items:center;
  justify-content:center;cursor:pointer;overflow:hidden;color:#6f6f7a;font-size:10px;text-align:center;flex:none}
.bsl .refslot img{width:100%;height:100%;object-fit:cover}
.bsl .refslot:hover{border-color:#5b8cff}
.bsl .modal{position:absolute;inset:0;background:#0b0b0ecc;z-index:20;display:flex;align-items:center;justify-content:center;padding:14px}
.bsl .mbox{background:#1b1b21;border:1px solid #3a3a45;border-radius:10px;width:100%;max-height:100%;display:flex;flex-direction:column}
.bsl .mhd{display:flex;gap:6px;align-items:center;padding:8px;border-bottom:1px solid #2c2c35}
.bsl .mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:6px;padding:8px;overflow:auto}
.bsl .mi{border:2px solid transparent;border-radius:7px;overflow:hidden;cursor:pointer;background:#141418}
.bsl .mi:hover{border-color:#5b8cff}
.bsl .mi img{width:100%;height:84px;object-fit:cover;display:block}
.bsl .mi div{font-size:9px;padding:2px 4px;color:#9a9aa6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bsl .prog{height:7px;border-radius:4px;background:#26262e;overflow:hidden}
.bsl .prog>div{height:100%;background:linear-gradient(90deg,#3a63e0,#7fe0a8)}
.bsl .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.bsl details>summary{cursor:pointer;list-style:none;user-select:none}
.bsl details>summary::-webkit-details-marker{display:none}
`;
  document.head.appendChild(el);
}

function Panel(io) {
  const plan = reactive({ ...DEFAULTS });
  const files = reactive({ videos: [], images: [] });
  const an = ref(null);          // analysis from the server
  const cuts = ref([]);
  const detectorUsed = ref("");
  const size = reactive({ w: 0, h: 0 });
  const busy = ref(""); const error = ref("");
  const sel = ref(0); const zoom = ref(1);
  const hover = ref(null);       // {frame, x}
  const gallery = ref(null);     // {target: index|'global', field: 'ref'|'ref2'}
  const gq = ref("");
  const prog = reactive({ done: 0, count: 0 });
  const tlEl = ref(null);
  const stats = ref([]);         // per-shot content stats from /bfs/shotloop/filters

  const load = () => {
    let p = {};
    try { p = JSON.parse(io.getPlan() || "{}"); } catch { p = {}; }
    Object.assign(plan, DEFAULTS, p);
    plan.filters = { ...DEFAULTS.filters, ...(p.filters || {}) };
    if (!Array.isArray(plan.bounds)) plan.bounds = [];
    if (!Array.isArray(plan.segs)) plan.segs = [];
  };
  const save = () => io.setPlan(JSON.stringify(plan));

  const n = computed(() => {
    if (!an.value) return 0;
    const cap = plan.max_total_s > 0 ? Math.round(plan.max_total_s * plan.fps) : Infinity;
    return Math.min(an.value.n, cap);
  });
  const maxLen = computed(() => snapDown(Math.round(plan.max_s * plan.fps), plan.grid));
  const segs = computed(() => {
    const N = n.value; if (!N) return [];
    const b = [0, ...plan.bounds.filter(x => x > 0 && x < N), N].sort((a, c) => a - c);
    const cs = new Set(cuts.value);
    let out = [];
    for (let i = 0; i < b.length - 1; i++) {
      if (b[i + 1] <= b[i]) continue;
      const m = plan.segs[out.length] || {};
      out.push({ start: b[i], end: b[i + 1], len: b[i + 1] - b[i], gen: snapUp(b[i + 1] - b[i], plan.grid),
                 cut: cs.has(b[i]), enabled: m.enabled !== false, ref: m.ref || "", ref2: m.ref2 || "", prompt: m.prompt || "",
                 force: m.force || "auto" });
    }
    if (plan.max_parts > 0) out = out.slice(0, plan.max_parts);
    return out;
  });
  const statFor = s => stats.value.find(x => x.start === s.start && x.end === s.end) || null;
  const skipWhy = s => {
    if (!s.enabled) return "disabled";
    if (s.force === "run") return "";
    if (s.force === "skip") return "skipped by hand";
    return filtersOn.value ? (statFor(s)?.skip_reason || "") : "";
  };
  const filtersOn = computed(() => { const f = plan.filters; return !!(f.person || f.face || f.skip_dark || f.skip_static || f.max_persons > 0 || f.min_frames > 0); });
  const active = computed(() => segs.value.filter(s => !skipWhy(s)));
  const pxPerFrame = computed(() => {
    const w = (tlEl.value?.clientWidth || 600) - 2;
    return Math.max(0.2, (w / Math.max(1, n.value)) * zoom.value);
  });
  const tlWidth = computed(() => Math.max(1, Math.round(n.value * pxPerFrame.value)));

  const meta = i => { while (plan.segs.length <= i) plan.segs.push({}); return plan.segs[i]; };

  async function refreshFiles() {
    try { const r = await api.fetchApi("/bfs/shotloop/files"); Object.assign(files, await r.json()); } catch (e) { /* ignore */ }
  }
  async function analyze() {
    an.value = null; error.value = "";
    if (!plan.video) return;
    busy.value = "Analysing video…";
    try {
      const r = await api.fetchApi(`/bfs/shotloop/analyze?video=${encodeURIComponent(plan.video)}&fps=${plan.fps}`);
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      an.value = j;
      await autoSplit(plan.bounds.length === 0);
    } catch (e) { error.value = String(e.message || e); }
    busy.value = "";
  }
  async function autoSplit(apply = true) {
    if (!plan.video) return;
    busy.value = "Detecting cuts…"; error.value = "";
    try {
      const body = { plan: { ...plan, bounds: apply ? [] : plan.bounds } };
      const r = await api.fetchApi("/bfs/shotloop/plan", { method: "POST", body: JSON.stringify(body) });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      cuts.value = j.cuts || []; detectorUsed.value = j.detector || ""; size.w = j.width; size.h = j.height;
      if (apply) {
        plan.bounds = j.segs.slice(1).map(s => s.start);
        const keep = plan.segs; plan.segs = j.segs.map((_, i) => ({ enabled: true, ref: keep[i]?.ref || "", ref2: keep[i]?.ref2 || "", prompt: keep[i]?.prompt || "" }));
        sel.value = 0; save();
      }
    } catch (e) { error.value = String(e.message || e); }
    busy.value = "";
  }
  async function analyzeContent() {
    if (!plan.video) return;
    busy.value = "Detecting people and faces…"; error.value = "";
    try {
      const r = await api.fetchApi("/bfs/shotloop/filters", { method: "POST", body: JSON.stringify({ plan: { ...plan } }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      stats.value = j.segs.map(x => ({ ...x.stats, start: x.start, end: x.end, skip_reason: x.skip_reason }));
    } catch (e) { error.value = String(e.message || e); }
    busy.value = "";
  }
  const setFilter = (k, v) => { plan.filters = { ...plan.filters, [k]: v }; save(); if (stats.value.length) analyzeContent(); };
  async function upload(file, cb) {
    const fd = new FormData(); fd.append("image", file); fd.append("type", "input"); fd.append("overwrite", "true");
    busy.value = `Uploading ${file.name}…`;
    try {
      const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
      const j = await r.json();
      await refreshFiles();
      cb(j.subfolder ? `${j.subfolder}/${j.name}` : j.name);
    } catch (e) { error.value = `Upload failed: ${e}`; }
    busy.value = "";
  }
  const pickFile = (accept, cb) => {
    const inp = document.createElement("input"); inp.type = "file"; inp.accept = accept;
    inp.onchange = () => inp.files[0] && upload(inp.files[0], cb); inp.click();
  };
  async function progress(reset = false) {
    if (!plan.video || plan.run !== "queue") return;
    try {
      const r = await api.fetchApi("/bfs/shotloop/progress", { method: "POST", body: JSON.stringify({ plan: { ...plan }, plan_raw: io.getPlan(), reset }) });
      const j = await r.json(); if (!j.error) { prog.done = j.done; prog.count = j.count || active.value.length; }
    } catch { /* ignore */ }
  }

  // ---- editing
  const setVideo = v => { plan.video = v; plan.bounds = []; plan.segs = []; save(); analyze(); };
  const splitAt = f => {
    const N = n.value; f = Math.round(f);
    if (f <= 0 || f >= N || plan.bounds.includes(f)) return;
    const i = segs.value.findIndex(s => f > s.start && f < s.end);
    plan.bounds = [...plan.bounds, f].sort((a, c) => a - c);
    plan.segs.splice(i + 1, 0, { ...(plan.segs[i] || {}) });
    sel.value = i + 1; save();
  };
  const mergeNext = i => {
    const s = segs.value[i]; if (!s || i >= segs.value.length - 1) return;
    plan.bounds = plan.bounds.filter(b => b !== s.end); plan.segs.splice(i + 1, 1); save();
  };
  const setMeta = (i, k, v) => { meta(i)[k] = v; save(); };
  const applyAll = (k) => { const v = meta(sel.value)[k] || ""; segs.value.forEach((_, i) => { meta(i)[k] = v; }); save(); };
  const drag = (k, ev) => {
    ev.preventDefault(); ev.stopPropagation();
    const box = tlEl.value.getBoundingClientRect();
    const move = e => {
      const x = e.clientX - box.left + tlEl.value.scrollLeft;
      const f = Math.round(x / pxPerFrame.value);
      const lo = (plan.bounds[k - 1] ?? 0) + 1, hi = (plan.bounds[k + 1] ?? n.value) - 1;
      plan.bounds[k] = Math.max(lo, Math.min(hi, f));
      hover.value = { frame: plan.bounds[k], x };
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); save(); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  };
  const frameAt = e => { const box = tlEl.value.getBoundingClientRect(); const x = e.clientX - box.left + tlEl.value.scrollLeft; return { frame: Math.max(0, Math.min(n.value - 1, Math.round(x / pxPerFrame.value))), x }; };
  const thumbFor = f => { const t = an.value?.thumbs; if (!t?.length) return ""; let b = t[0]; for (const x of t) { if (x.f <= f) b = x; else break; } return b.src; };

  // ---- queue loop events
  const onProg = e => { prog.done = e.detail.done; prog.count = e.detail.count; };
  const onNext = e => { onProg(e); if (plan.run === "queue" && plan.auto_continue) setTimeout(() => app.queuePrompt(0, 1), 300); };
  onMounted(() => {
    load(); refreshFiles(); analyze(); progress();
    api.addEventListener("bfs-shotloop-progress", onProg); api.addEventListener("bfs-shotloop-next", onNext);
  });
  onBeforeUnmount(() => { api.removeEventListener("bfs-shotloop-progress", onProg); api.removeEventListener("bfs-shotloop-next", onNext); });
  io.expose({ reload: () => { load(); analyze(); progress(); } });

  // ---- view helpers
  const fld = (label, input, hint) => h("div", { class: "fld" }, [h("label", label), input, hint ? h("div", { class: "hint" }, hint) : null]);
  const num = (k, step = 0.1, min = 0) => h("input", { type: "number", step, min, value: plan[k], onChange: e => { plan[k] = parseFloat(e.target.value) || 0; save(); } });
  const sel_ = (k, opts, after) => h("select", { value: plan[k], onChange: e => { plan[k] = e.target.value; save(); after && after(); } },
    opts.map(o => h("option", { value: Array.isArray(o) ? o[0] : o }, Array.isArray(o) ? o[1] : o)));
  const refSlot = (name, label, onClick, onClear) => h("div", { class: "refslot", title: name || label, onClick },
    name ? [h("img", { src: viewUrl(name) })] : [label]);

  return () => {
    const S = segs.value, cur = S[sel.value], N = n.value, ppf = pxPerFrame.value, fps = plan.fps;
    const tooLong = S.filter(s => s.len > maxLen.value).length;
    const totalGen = active.value.reduce((a, s) => a + s.gen, 0);

    const header = h("div", { class: "hdr" }, [
      h("span", { class: "ttl" }, "🎬 Shot Planner"),
      busy.value ? h("span", { class: "pill warn" }, busy.value) : (an.value ? h("span", { class: "pill ok" }, `${active.value.length} shots · ${(N / fps).toFixed(1)}s`) : null),
      h("span", { class: "grow" }),
      plan.run === "queue" ? h("span", { class: "pill" }, "queue loop") : h("span", { class: "pill" }, "auto loop"),
    ]);

    const source = h("div", { class: "card" }, [
      h("h5", ["Source video", detectorUsed.value ? h("span", { class: "pill" }, detectorUsed.value) : null]),
      h("div", { class: "row" }, [
        h("div", { style: "flex:1;min-width:180px" }, [h("select", { value: plan.video, onChange: e => setVideo(e.target.value) },
          [h("option", { value: "" }, "— choose a video —"), ...files.videos.map(v => h("option", { value: v }, v))])]),
        h("button", { onClick: () => pickFile("video/*", setVideo) }, "⬆ Upload"),
        h("button", { onClick: refreshFiles, title: "refresh the input folder" }, "↻"),
      ]),
      an.value ? h("div", { class: "hint", style: "margin-top:4px" },
        `${an.value.width}×${an.value.height} · ${an.value.fps_src.toFixed(2)} fps · ${an.value.duration.toFixed(2)}s → timeline ${fps} fps, ${an.value.n} frames · generate at ${size.w}×${size.h}`) : null,
    ]);

    const settings = h("details", { class: "card", open: true }, [
      h("summary", h("h5", ["Split settings", h("span", { class: "hint", style: "text-transform:none;letter-spacing:0" }, `max ${maxLen.value} frames per shot (${(maxLen.value / fps).toFixed(2)}s)`)])),
      h("div", { class: "grid" }, [
        fld("Mode", sel_("mode", [["shots", "Camera cuts"], ["fixed", "Fixed length"]])),
        fld("Detector", sel_("detector", [["adaptive", "PySceneDetect adaptive"], ["content", "PySceneDetect content"], ["builtin", "Built-in"]])),
        fld(`Sensitivity ${Number(plan.sensitivity).toFixed(2)}`, h("input", { type: "range", min: 0, max: 1, step: 0.05, value: plan.sensitivity, onInput: e => { plan.sensitivity = parseFloat(e.target.value); }, onChange: save })),
        fld("Frame grid", sel_("grid", Object.keys(GRIDS))),
        fld("Max seconds / shot", num("max_s", 0.1, 0.2)),
        fld("Min seconds / shot", num("min_s", 0.1, 0)),
        fld("Timeline fps", num("fps", 1, 1), "re-analyses"),
        fld("Max shots (0 = all)", num("max_parts", 1, 0)),
        fld("Max total seconds (0 = all)", num("max_total_s", 0.5, 0)),
        fld("Megapixels", num("megapixels", 0.01, 0.02)),
        fld("Size multiple", num("multiple", 8, 8)),
        fld("Run", sel_("run", [["auto", "Auto loop (one run)"], ["queue", "Queue loop (one shot per run)"]], progress)),
      ]),
      h("div", { class: "row", style: "margin-top:8px" }, [
        h("button", { class: "pri", disabled: !plan.video || !!busy.value, onClick: () => autoSplit(true) }, "✂ Auto split"),
        h("button", { disabled: !plan.video, onClick: analyze }, "↻ Re-analyse"),
        h("span", { class: "hint" }, "Drag the white handles to move a boundary · double-click the shot bar to split there"),
      ]),
    ]);

    // timeline
    const thumbs = an.value?.thumbs || [];
    const thumbW = Math.max(8, (an.value?.thumb_w || 96) * 0 + (thumbs.length ? tlWidth.value / thumbs.length : 96));
    const score = an.value?.score || [];
    const maxS = Math.max(4, ...score.slice(0, N));
    const sparkPts = score.slice(0, N).map((v, i) => `${(i * ppf).toFixed(1)},${(24 - Math.min(1, v / maxS) * 22).toFixed(1)}`).join(" ");
    const timeline = an.value ? h("div", { class: "card" }, [
      h("h5", ["Timeline", h("span", { class: "grow" }), h("span", { class: "hint", style: "text-transform:none" }, "zoom"),
        h("input", { type: "range", min: 1, max: 8, step: 0.5, value: zoom.value, style: "width:110px", onInput: e => { zoom.value = parseFloat(e.target.value); } })]),
      h("div", { class: "tl", ref: tlEl,
        onPointermove: e => { if (!e.buttons) hover.value = frameAt(e); }, onPointerleave: () => { hover.value = null; } }, [
        h("div", { class: "tlin", style: `width:${tlWidth.value}px` }, [
          h("div", { class: "strip" }, thumbs.map(t => h("img", { src: t.src, style: `width:${thumbW}px` }))),
          ...cuts.value.filter(c => c < N).map(c => h("div", { class: "cut", style: `left:${c * ppf}px`, title: `cut @ ${c}` })),
          h("svg", { class: "spark", viewBox: `0 0 ${tlWidth.value} 26`, preserveAspectRatio: "none" },
            [h("polyline", { points: sparkPts, fill: "none", stroke: "#ff7a90", "stroke-width": 1, "vector-effect": "non-scaling-stroke" })]),
          h("div", { class: "segs", onDblclick: e => splitAt(frameAt(e).frame) }, [
            ...S.map((s, i) => h("div", {
              class: ["seg", i === sel.value && "sel", !!skipWhy(s) && "off", s.len > maxLen.value && "long"],
              style: `left:${s.start * ppf}px;width:${Math.max(2, s.len * ppf - 1)}px;background:${hue(i)}`,
              title: `#${i + 1} · frames ${s.start}-${s.end - 1} · ${s.len} → ${s.gen}${skipWhy(s) ? " · skip: " + skipWhy(s) : ""}`, onClick: () => { sel.value = i; },
            }, s.len * ppf > 26 ? `${i + 1}` : "")),
            ...plan.bounds.filter(b => b < N).map((b, k) => h("div", { class: "hdl", style: `left:${b * ppf}px`, title: `boundary @ ${b}`, onPointerdown: e => drag(k, e) })),
          ]),
          hover.value ? h("div", { class: "ph", style: `left:${hover.value.x}px` }) : null,
        ]),
        hover.value ? h("div", { class: "tip", style: `left:${Math.min(tlWidth.value - 160, Math.max(0, hover.value.x - (tlEl.value?.scrollLeft || 0) - 70))}px;top:2px` },
          [h("img", { src: thumbFor(hover.value.frame) }), h("div", `frame ${hover.value.frame} · ${fmtT(hover.value.frame, fps)}`)]) : null,
      ]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("span", { class: "pill" }, `${S.length} shots`), h("span", { class: "pill" }, `${cuts.value.length} cuts`),
        h("span", { class: "pill" }, `generate ${totalGen} frames`),
        tooLong ? h("span", { class: "pill warn" }, `${tooLong} shot(s) longer than ${maxLen.value} frames`) : null,
      ]),
    ]) : null;

    const F = plan.filters;
    const chk = (k, label) => h("label", { class: "row", style: "gap:4px" }, [h("input", { type: "checkbox", checked: !!F[k], onChange: e => setFilter(k, e.target.checked) }), label]);
    const fnum = (k, step, label, hint) => fld(label, h("input", { type: "number", step, min: 0, value: F[k], onChange: e => setFilter(k, parseFloat(e.target.value) || 0) }), hint);
    const skippedN = S.filter(s => skipWhy(s)).length;
    const filters = an.value ? h("details", { class: "card", open: filtersOn.value || stats.value.length > 0 }, [
      h("summary", h("h5", ["Filters", filtersOn.value ? h("span", { class: "pill warn" }, `${skippedN} skipped`) : h("span", { class: "pill" }, "off"),
        h("span", { class: "hint", style: "text-transform:none;letter-spacing:0" }, "skipped shots do not run; the join fills them with the original video or drops them")])),
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        chk("person", "Needs a person"), chk("face", "Needs a face"), chk("skip_dark", "Skip dark / fades"), chk("skip_static", "Skip static shots"),
      ]),
      h("div", { class: "grid" }, [
        fnum("min_person_area", 0.01, "Min person size (0-1 of frame)", "e.g. 0.03 skips wide shots"),
        fnum("max_persons", 1, "Max people (0 = any)", "skip crowds"),
        fnum("min_frames", 1, "Min frames (0 = off)"),
        fnum("samples", 1, "Frames sampled / shot"),
        fnum("dark_level", 0.01, "Dark below (0-1)"),
        fnum("static_level", 0.001, "Static below"),
        fld("Skipped shots in the output", sel_("skip_fill", [["original", "Keep original video"], ["drop", "Remove them"]])),
      ]),
      h("div", { class: "row", style: "margin-top:8px" }, [
        h("button", { class: "pri", disabled: !!busy.value, onClick: analyzeContent }, "👤 Analyse people & faces"),
        h("span", { class: "hint" }, stats.value.length ? `stats for ${stats.value.length} shots · YOLO person/face from models/ultralytics` : "runs the detectors on a few frames of every shot"),
      ]),
    ]) : null;

    // shot cards
    const cards = S.length ? h("div", { class: "card" }, [
      h("h5", "Shots"),
      h("div", { class: "shots" }, S.map((s, i) => h("div", { class: ["sc", i === sel.value && "sel"], onClick: () => { sel.value = i; } }, [
        h("div", { class: "bar", style: `background:${hue(i)}` }),
        h("div", { class: "row" }, [h("b", `#${i + 1}`), s.cut ? h("span", { class: "pill" }, "cut") : null,
          skipWhy(s) ? h("span", { class: "pill warn", title: skipWhy(s) }, "skip") : h("span", { class: "pill ok" }, "run"),
          s.force !== "auto" ? h("span", { class: "pill" }, s.force) : null]),
        statFor(s) ? h("div", { class: "t", style: "margin-top:2px" },
          `👤 ${statFor(s).persons} · ${(statFor(s).person_area * 100).toFixed(1)}% · 🙂 ${statFor(s).faces} · ☀ ${(statFor(s).brightness * 100).toFixed(0)}%`) : null,
        skipWhy(s) ? h("div", { class: "t", style: "color:#ffc46b" }, skipWhy(s)) : null,
        h("div", { class: "t" }, `${fmtT(s.start, fps)} → ${fmtT(s.end, fps)} · ${s.len}f → ${s.gen}f`),
        h("div", { class: "thumbs" }, [
          s.ref || plan.global_ref ? h("img", { class: "rt", src: viewUrl(s.ref || plan.global_ref), style: s.ref ? "" : "opacity:.45" }) : h("div", { class: "rt ph2" }, "ref"),
          s.ref2 || plan.global_ref2 ? h("img", { class: "rt", src: viewUrl(s.ref2 || plan.global_ref2), style: s.ref2 ? "" : "opacity:.45" }) : h("div", { class: "rt ph2" }, "ref2"),
          h("img", { class: "rt", src: thumbFor(s.start + Math.floor(s.len / 2)), style: "width:56px" }),
        ]),
        h("div", { class: "p" }, s.prompt ? s.prompt : (plan.global_prompt ? "↳ global prompt" : "— no prompt —")),
      ]))),
    ]) : null;

    // editor for the selected shot
    const editor = cur ? h("div", { class: "card" }, [
      h("h5", [`Shot #${sel.value + 1}`, h("span", { class: "hint", style: "text-transform:none" }, `frames ${cur.start}–${cur.end - 1} · ${cur.len} → generate ${cur.gen}`)]),
      h("div", { class: "row", style: "align-items:flex-start;gap:10px" }, [
        h("div", { class: "refbox" }, [
          refSlot(cur.ref, "＋ reference\n(uses global)", () => { gallery.value = { target: sel.value, field: "ref" }; }),
          refSlot(cur.ref2, "＋ ref 2\n(uses global)", () => { gallery.value = { target: sel.value, field: "ref2" }; }),
        ]),
        h("div", { style: "flex:1;min-width:200px" }, [
          h("textarea", { placeholder: "Prompt for this shot (empty = global prompt)", value: cur.prompt, onChange: e => setMeta(sel.value, "prompt", e.target.value) }),
        ]),
      ]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("select", { value: cur.force, style: "width:auto", title: "Override the content filters for this shot",
          onChange: e => setMeta(sel.value, "force", e.target.value) },
          [h("option", { value: "auto" }, "filters decide"), h("option", { value: "run" }, "always run"), h("option", { value: "skip" }, "always skip")]),
        h("button", { onClick: () => setMeta(sel.value, "enabled", !cur.enabled) }, cur.enabled ? "⏸ Disable" : "▶ Enable"),
        h("button", { onClick: () => splitAt(cur.start + Math.floor(cur.len / 2)) }, "✂ Split in half"),
        h("button", { disabled: sel.value >= S.length - 1, onClick: () => mergeNext(sel.value) }, "⇥ Merge with next"),
        h("button", { onClick: () => applyAll("ref") }, "Ref → all"), h("button", { onClick: () => applyAll("prompt") }, "Prompt → all"),
        h("button", { class: "dng", onClick: () => { ["ref", "ref2", "prompt"].forEach(k => { meta(sel.value)[k] = ""; }); save(); } }, "Use global"),
      ]),
    ]) : null;

    const globals = h("div", { class: "card" }, [
      h("h5", "Global (used by shots without their own)"),
      h("div", { class: "row", style: "align-items:flex-start;gap:10px" }, [
        h("div", { class: "refbox" }, [
          refSlot(plan.global_ref, "＋ global\nreference", () => { gallery.value = { target: "global", field: "global_ref" }; }),
          refSlot(plan.global_ref2, "＋ global\nref 2", () => { gallery.value = { target: "global", field: "global_ref2" }; }),
        ]),
        h("div", { style: "flex:1;min-width:200px" }, [h("textarea", { placeholder: "Global prompt (a connected `prompt` input overrides it)", value: plan.global_prompt, onChange: e => { plan.global_prompt = e.target.value; save(); } })]),
      ]),
      h("div", { class: "hint", style: "margin-top:4px" }, "Connected ref_image / ref_image_2 / prompt inputs on the node override these defaults."),
    ]);

    const queue = plan.run === "queue" ? h("div", { class: "card" }, [
      h("h5", ["Queue loop", h("span", { class: "grow" }), h("span", { class: "hint", style: "text-transform:none" }, `${prog.done}/${prog.count || active.value.length} shots done`)]),
      h("div", { class: "prog" }, [h("div", { style: `width:${(100 * prog.done / Math.max(1, prog.count || active.value.length)).toFixed(1)}%` })]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("label", { class: "row" }, [h("input", { type: "checkbox", checked: plan.auto_continue, onChange: e => { plan.auto_continue = e.target.checked; save(); } }), "Auto-queue the next shot"]),
        h("button", { onClick: () => progress(false) }, "↻ Status"), h("button", { class: "dng", onClick: () => progress(true) }, "⟲ Reset loop"),
      ]),
      h("div", { class: "hint", style: "margin-top:4px" }, "Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video."),
    ]) : null;

    const modal = gallery.value ? h("div", { class: "modal", onClick: e => { if (e.target === e.currentTarget) gallery.value = null; } }, [
      h("div", { class: "mbox" }, [
        h("div", { class: "mhd" }, [
          h("b", "Choose reference"), h("input", { type: "text", placeholder: "search…", value: gq.value, onInput: e => { gq.value = e.target.value; }, style: "flex:1" }),
          h("button", { onClick: () => pickFile("image/*", name => { choose(name); }) }, "⬆ Upload"),
          h("button", { onClick: () => choose("") }, "None"), h("button", { onClick: () => { gallery.value = null; } }, "✕"),
        ]),
        h("div", { class: "mgrid" }, files.images.filter(f => f.toLowerCase().includes(gq.value.toLowerCase())).slice(0, 400).map(f =>
          h("div", { class: "mi", onClick: () => choose(f), title: f }, [h("img", { src: viewUrl(f), loading: "lazy" }), h("div", f)]))),
      ]),
    ]) : null;
    function choose(name) {
      const g = gallery.value; if (!g) return;
      if (g.target === "global") plan[g.field] = name; else meta(g.target)[g.field] = name;
      save(); gallery.value = null;
    }

    return h("div", { class: "bsl" }, [header, error.value ? h("div", { class: "err" }, error.value) : null, source, settings, timeline, filters, cards, editor, globals, queue, modal]);
  };
}

app.registerExtension({
  name: "BFSNodes.ShotPlanner",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSShotPlanner") return;
    styles();
    const planWidget = node.widgets?.find(w => w.name === "plan");
    if (planWidget) { planWidget.type = "hidden"; planWidget.computeSize = () => [0, -4]; }
    const host = document.createElement("div");
    host.style.cssText = "width:100%;height:100%;min-height:520px";
    node.addDOMWidget("bfs_shot_planner", "div", host, { serialize: false, hideOnZoom: false });
    const handle = {};
    createApp({
      setup: () => Panel({
        getPlan: () => planWidget?.value ?? "{}",
        setPlan: v => { if (planWidget) planWidget.value = v; node.graph?.setDirtyCanvas(true); },
        expose: o => Object.assign(handle, o),
      }),
    }).mount(host);
    if (planWidget) {
      planWidget.options = planWidget.options || {};
      planWidget.options.serialize = true;
      planWidget.serializeValue = () => planWidget.value;
    }
    const prev = node.onConfigure;
    node.onConfigure = function () {
      const r = prev?.apply(this, arguments);
      setTimeout(() => handle.reload?.(), 0);
      return r;
    };
    node.size = [Math.max(node.size[0], 780), Math.max(node.size[1], 860)];
  },
});
