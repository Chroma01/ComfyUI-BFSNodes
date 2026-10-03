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
import { showGuide } from "./bfs_md_view.js";
import { createApp, ref, reactive, computed, watch, onMounted, onBeforeUnmount, h, Teleport } from "./vendor/vue.esm-browser.prod.mjs";

const GRIDS = { "H3 (17n+5)": [17, 5], "LTX / Wan (8n+1)": [8, 1], "Wan (4n+1)": [4, 1], "any": [1, 0] };
const DEFAULTS = {
  video: "", fps: 24, grid: "H3 (17n+5)", mode: "shots", max_s: 4.5, min_s: 1.0, sensitivity: 0.5,
  max_parts: 0, max_total_s: 0, bounds: [], segs: [], global_ref: "", global_ref2: "", global_prompt: "",
  megapixels: 0.15, multiple: 32, detector: "adaptive", run: "auto", auto_continue: true,
  skip_fill: "original", audio_mode: "auto", ref_details: {}, mask_cfg: {}, vlm_cfg: {}, cast: {}, cast_assign: true, cast_split: false, cast_only: false,
  filters: { person: false, min_person_area: 0, max_persons: 0, face: false, skip_dark: false, dark_level: 0.06,
             skip_static: false, static_level: 0.004, min_frames: 0, samples: 6 },
};

// global mask settings (same defaults as the server's DEFAULT_MASK)
const MASK_DEFAULTS = { threshold: 0.5, max_objects: 4, invert: false, fill_holes: true, temporal_expand: 2, blockify: 0,
                        padding: 0.15, expand: 16, feather: 12, paste: "mask" };

const VLM_DEFAULTS = { enabled: false, frames: 3, max_tokens: 1024, auto_segment: true, auto_shot: true, instruction: "",
                       describe_preset: "full body", describe_custom: "" };

// hover help for every setting (the 📖 Guide button opens the full documentation)
const TIPS = {
  "Mode": "Camera cuts: shots start at the detected cuts, long ones are split evenly. Fixed length: equal parts no longer than the maximum.",
  "Detector": "Cut detector. PySceneDetect adaptive works on most footage (fast camera moves don't trigger it); content is stricter on hard cuts; built-in needs no package.",
  "Sensitivity": "How easily a change counts as a cut. Higher finds more cuts (and more false ones).",
  "Frame grid": "Frame counts the model accepts. H3: 17n+5, LTX/Wan: 8n+1, Wan: 4n+1. Each shot is generated at the next valid length and trimmed back.",
  "Max seconds / shot": "Longest shot sent to the model. Keep it within what the model / LoRA was trained on (H3 body swap: about 4.5 s).",
  "Min seconds / shot": "Shorter shots merge into a neighbour when the merge still fits the maximum.",
  "Timeline fps": "The planner works on this frame rate (the source is resampled). Generated video and audio use it too.",
  "Max shots": "Only the first N shots (0 = all). Handy for quick tests.",
  "Max total seconds": "Only the first N seconds of the video (0 = all).",
  "Megapixels": "Generation size: the source aspect ratio at this pixel area (0.15 ≈ 512×288). Cropped shots use it for the crop.",
  "Size multiple": "Width and height snap to this multiple (32 for H3 and most video models).",
  "Run": "Auto loop: every shot in one queue run. Queue loop: one shot per run, stored on disk, the join outputs the video after the last shot.",
  "Skipped shots in the output": "Shots that do not run: keep the original video there, or remove them (with their audio).",
  "Crop padding": "Context kept around the mask's box, as a % of the box. More context blends better; less gives the model more pixels on the subject.",
  "Expand": "Grow the mask by this many pixels before pasting the result back (covers hair, edges, motion blur).",
  "Feather": "Soft edge of the paste-back, in pixels.",
  "Temporal expand": "Hold the mask this many frames before and after: removes flicker where SAM 3 drops a frame.",
  "Blockify": "Square blocks of this size (0 = off). 16 matches H3's latent grid, useful as an inpainting mask.",
  "Detection threshold": "SAM 3 score needed to keep a text match. Lower finds more (and more wrong) objects.",
  "Max objects": "How many matches of the text prompt are tracked together.",
  "Paste back": "Only the mask: the result replaces the masked pixels. The whole box: the full crop is pasted with feathered borders.",
  "Frames per shot": "How many frames of each shot the VLM looks at.",
  "Max tokens": "Longest VLM answer. Raise it for long descriptions.",
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
.bsl:focus{outline:none}
.bsl .qm{color:#7d8bb0;cursor:help}
.bsl .busybar{position:sticky;top:0;z-index:5;margin:6px 0;padding:6px 8px;border-radius:8px;background:#22222b;border:1px solid #3d3d4a}
.bsl .busybar .brow{display:flex;gap:6px;align-items:center}
.bsl .busybar .spin{width:12px;height:12px;border-radius:50%;border:2px solid #5b8cff;border-top-color:transparent;animation:bslspin .8s linear infinite;flex:none}
.bsl .busybar .bprog{height:4px;background:#33333d;border-radius:3px;margin-top:5px;overflow:hidden;position:relative}
.bsl .busybar .bprog>div{height:100%;background:#5b8cff;transition:width .2s}
.bsl .busybar .bprog.ind>div{position:absolute;width:30%;animation:bslind 1.2s ease-in-out infinite}
@keyframes bslspin{to{transform:rotate(360deg)}}
@keyframes bslind{0%{left:-30%}100%{left:100%}}
.bsl .dlist{display:flex;flex-direction:column;gap:6px;margin-top:6px}
.bsl .drow{display:flex;gap:8px;align-items:flex-start}
.bsl .dthumbs{display:flex;gap:3px}
.bsl .dthumbs img{width:44px;height:58px;object-fit:cover;border-radius:5px;border:1px solid #3a3a45}
.bsl .vsug{margin-top:6px;padding:6px 8px;border:1px dashed #4a4a58;border-radius:6px;display:flex;flex-direction:column;gap:3px}
.bsl .vsug button{margin-left:6px;padding:1px 6px}
.bsl .mstrip{display:flex;gap:4px;margin-top:6px;align-items:center;flex-wrap:wrap}
.bsl .mstrip img{height:90px;border-radius:5px;border:1px solid #3a3a45}
.bsl .mmodal{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center}
.bsl .mbox{background:#1d1d23;border:1px solid #3a3a45;border-radius:10px;padding:12px;width:min(960px,94vw);max-height:94vh;overflow:auto}
.bsl .mimg{position:relative;cursor:crosshair;user-select:none;line-height:0}
.bsl .mimg img{width:100%;border-radius:6px}
.bsl .pt{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;border:2px solid #fff;cursor:pointer}
.bsl .pt.pos{background:#3ccf6b}.bsl .pt.neg{background:#e8455a}
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
.bsl .rslot{display:flex;flex-direction:column;gap:3px;align-items:center}
.bsl .rbtns{display:flex;gap:3px}
.bsl .rbtns button{padding:1px 6px;font-size:10px}
.bsl .player{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}
.bsl .player video{max-height:240px;max-width:100%;border-radius:8px;background:#000;border:1px solid #2c2c35}
.bsl .pinfo{flex:1;min-width:200px;display:flex;flex-direction:column;gap:6px}
.bsl .tc{font-family:ui-monospace,monospace;font-size:12px;color:#e6e6ea;background:#101014;border:1px solid #2c2c35;border-radius:6px;padding:6px 8px;line-height:1.6}
.bsl .tc b{color:#ffd34d}
.bsl .playhead{position:absolute;top:0;bottom:0;width:2px;background:#ffd34d;box-shadow:0 0 6px #ffd34d;pointer-events:none;z-index:5}
.bsl .sc .play{position:absolute;top:5px;right:5px;padding:1px 6px;font-size:10px}
.bsl .seg.playing{box-shadow:0 0 0 2px #ffd34d}
.bsl .recent{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;align-items:center}
.bsl .recent img{width:30px;height:30px;border-radius:5px;object-fit:cover;cursor:pointer;border:1px solid #3a3a45}
.bsl .recent img:hover{border-color:#5b8cff}
.bsl .recent.sm img{width:22px;height:22px}
.bsl .cast{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.bsl .person{background:#141418;border:1px solid #2c2c35;border-radius:8px;padding:6px;display:flex;flex-direction:column;gap:4px;align-items:flex-start}
.bsl .person.ign{opacity:.45}
.bsl .person .face{width:64px;height:64px;border-radius:50%;object-fit:cover;cursor:pointer;border:2px solid #3a3a45}
.bsl .refslot.sm{width:56px;height:56px;font-size:10px}
.bsl .who{display:flex;gap:3px;margin-top:3px;align-items:center}
.bsl .who img{width:22px;height:22px;border-radius:50%;object-fit:cover;border:1px solid #3a3a45;opacity:.7}
.bsl .who img.main{width:26px;height:26px;opacity:1;border-color:#fff}
.bsl .who img.lk{border-color:#8fd18f}
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
  // loading bar: what the server reports (stage, done/total) and how long it has been busy
  const status = reactive({ stage: "", done: 0, total: 0 });
  const busySince = ref(0); const now = ref(Date.now());
  const upPct = ref(-1);
  watch(busy, v => { if (v) { busySince.value = Date.now(); Object.assign(status, { stage: "", done: 0, total: 0 }); } else { busySince.value = 0; upPct.value = -1; } });
  const sel = ref(0); const zoom = ref(1);
  const hover = ref(null);       // {frame, x}
  const prog = reactive({ done: 0, count: 0 });
  const tlEl = ref(null);
  const stats = ref([]);         // per-shot content stats from /bfs/shotloop/filters
  const people = ref([]);        // cast from /bfs/shotloop/cast: [{id, thumb, share, first, last}]
  const maskPrev = ref({});      // "start-end" -> {frames, box, coverage, empty} from /bfs/shotloop/mask
  const showMasks = ref(true);   // overlay the mask preview on the shot cards
  const vlmSug = ref({});        // "start-end" -> {segment, shot, people, recommend, reason} from the VLM
  const modal = reactive({ open: false, idx: -1, key: 0, points: [], text: "", prev: null, busy: "" });
  const segPeople = ref({});     // "start-end" -> {people, main} from /bfs/shotloop/plan

  const load = () => {
    let p = {};
    try { p = JSON.parse(io.getPlan() || "{}"); } catch { p = {}; }
    Object.assign(plan, DEFAULTS, p);
    plan.filters = { ...DEFAULTS.filters, ...(p.filters || {}) };
    plan.cast = { ...(p.cast || {}) };
    plan.mask_cfg = { ...MASK_DEFAULTS, ...(p.mask_cfg || {}) };
    plan.vlm_cfg = { ...VLM_DEFAULTS, ...(p.vlm_cfg || {}) };
    plan.ref_details = { ...(p.ref_details || {}) };
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
                 force: m.force || "auto", chain: m.chain || "off", chainFrame: m.chain_frame || "first",
                 crop: !!m.crop, mask: m.mask || {} });
    }
    if (plan.max_parts > 0) out = out.slice(0, plan.max_parts);
    return out;
  });
  const whoIn = s => segPeople.value[`${s.start}-${s.end}`] || null;
  const castOf = id => plan.cast[String(id)] || {};
  const linked = id => !!castOf(id).ref && !castOf(id).ignore;
  // reference a shot gets from its main person when it has none of its own
  const castRef = (s, field) => {
    if (!plan.cast_assign) return "";
    const who = whoIn(s); if (!who || who.main < 0 || castOf(who.main).ignore) return "";
    return castOf(who.main)[field] || "";
  };
  const personOf = id => people.value.find(p => p.id === id) || null;
  const statFor = s => stats.value.find(x => x.start === s.start && x.end === s.end) || null;
  const skipWhy = s => {
    if (!s.enabled) return "disabled";
    if (s.force === "run") return "";
    if (s.force === "skip") return "skipped by hand";
    const why = filtersOn.value ? (statFor(s)?.skip_reason || "") : "";
    if (why) return why;
    const who = whoIn(s);
    if (plan.cast_only && who && !who.people.some(p => linked(p))) return "no linked person";
    return "";
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
    try { const r = await api.fetchApi("/bfs/shotloop/files"); const j = await r.json(); files.videos = j.videos || []; } catch (e) { /* ignore */ }
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
      people.value = []; segPeople.value = {};
      if (Object.keys(plan.cast).length || plan.cast_split || plan.cast_only) await findPeople(false);   // restore the cast of a saved plan
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
      segPeople.value = Object.fromEntries(j.segs.filter(s => s.main !== undefined && (s.people.length || people.value.length))
        .map(s => [`${s.start}-${s.end}`, { people: s.people, main: s.main }]));
      if (apply) {
        plan.bounds = j.segs.slice(1).map(s => s.start);
        const keep = plan.segs; plan.segs = j.segs.map((_, i) => ({ enabled: true, ref: keep[i]?.ref || "", ref2: keep[i]?.ref2 || "", prompt: keep[i]?.prompt || "",
          chain: keep[i]?.chain || "off", chain_frame: keep[i]?.chain_frame || "first",
          crop: !!keep[i]?.crop, mask: keep[i]?.mask || {} }));
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
  async function findPeople(resplit = true) {
    if (!plan.video) return;
    busy.value = "Finding people by face…"; error.value = "";
    try {
      const r = await api.fetchApi("/bfs/shotloop/cast", { method: "POST", body: JSON.stringify({ plan: { ...plan } }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      people.value = j.people || [];
    } catch (e) { error.value = String(e.message || e); busy.value = ""; return; }
    busy.value = "";
    await autoSplit(resplit && !!plan.cast_split);
  }
  const setCast = (id, k, v) => {
    plan.cast = { ...plan.cast, [String(id)]: { ...castOf(id), [k]: v } };
    if (k !== "ignore") remember(v);
    save();
  };
  const segKey = s => `${s.start}-${s.end}`;
  async function previewMask(i, spec, intoModal = false) {
    const s = segs.value[i]; if (!s) return;
    const label = "Segmenting with SAM 3… (the first time downloads/loads the model)";
    if (intoModal) { modal.busy = label; busySince.value = Date.now(); Object.assign(status, { stage: "", done: 0, total: 0 }); } else busy.value = label;
    error.value = "";
    try {
      const r = await api.fetchApi("/bfs/shotloop/mask", { method: "POST", body: JSON.stringify({ plan: { ...plan }, index: i, mask: spec || null }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      if (intoModal) modal.prev = j; else maskPrev.value = { ...maskPrev.value, [segKey(s)]: j };
    } catch (e) { error.value = String(e.message || e); }
    if (intoModal) modal.busy = ""; else busy.value = "";
  }
  const setMask = (i, k, v) => { const m = meta(i); m.mask = { ...(m.mask || {}), [k]: v }; save(); };
  const setMaskCfg = (k, v) => { plan.mask_cfg = { ...plan.mask_cfg, [k]: v }; save(); maskPrev.value = {}; };
  const openPoints = i => {
    const s = segs.value[i]; if (!s) return;
    Object.assign(modal, { open: true, idx: i, key: s.mask.key ?? Math.floor(s.len / 2), points: [...(s.mask.points || [])],
                           text: s.mask.text || "", prev: null, busy: "" });
  };
  const savePoints = () => {
    const m = meta(modal.idx); m.mask = { ...(m.mask || {}), points: modal.points, key: modal.key, text: modal.text }; save();
    if (modal.prev) maskPrev.value = { ...maskPrev.value, [segKey(segs.value[modal.idx])]: modal.prev };
    modal.open = false;
  };
  const mergeSug = list => { const o = { ...vlmSug.value }; for (const x of list || []) o[`${x.start}-${x.end}`] = x; vlmSug.value = o; };
  async function analyseVLM() {
    busy.value = "Asking the VLM about every shot…"; error.value = "";
    try {
      const r = await api.fetchApi("/bfs/shotloop/vlm", { method: "POST", body: JSON.stringify({ plan: { ...plan } }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      mergeSug(j.segs);
    } catch (e) { error.value = String(e.message || e); }
    busy.value = "";
  }
  // reference sets used by the plan (global, every shot's, every cast person's): one description each
  const refSets = computed(() => {
    const out = new Map();
    const add = (a, b, where) => { if (!a && !b) return; const k = `${a || ""}|${b || ""}`; if (!out.has(k)) out.set(k, { a, b, where: [] }); out.get(k).where.push(where); };
    add(plan.global_ref, plan.global_ref2, "global");
    segs.value.forEach((x, i) => add(x.ref || plan.global_ref, x.ref2 || plan.global_ref2, `#${i + 1}`));
    Object.entries(plan.cast || {}).forEach(([id, c]) => { if (!c.ignore) add(c.ref, c.ref2, `person ${id}`); });
    return [...out.entries()].map(([key, v]) => ({ key, ...v }));
  });
  async function describeRefs() {
    busy.value = "Describing the references with the VLM…"; error.value = "";
    try {
      const sets = refSets.value.map(x => [x.a || "", x.b || ""]);
      const r = await api.fetchApi("/bfs/shotloop/describe", { method: "POST", body: JSON.stringify({ plan: { ...plan }, sets }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      plan.ref_details = { ...plan.ref_details, ...j.texts }; save();
    } catch (e) { error.value = String(e.message || e); }
    busy.value = "";
  }
  const setDetail = (k, v) => { plan.ref_details = { ...plan.ref_details, [k]: v }; save(); };
  const setVlmCfg = (k, v) => { plan.vlm_cfg = { ...plan.vlm_cfg, [k]: v }; save(); };
  const applySug = (i, what) => {
    const s = segs.value[i], g = s && vlmSug.value[`${s.start}-${s.end}`]; if (!g) return;
    if (what === "segment" && g.segment) setMask(i, "text", g.segment);
    if (what === "skip") setMeta(i, "force", g.recommend === "skip" ? "skip" : "auto");
  };
  const applyAllSug = () => {
    segs.value.forEach((s, i) => {
      const g = vlmSug.value[`${s.start}-${s.end}`]; if (!g) return;
      const m = meta(i);
      if (g.segment && !(m.mask?.text || (m.mask?.points || []).length)) m.mask = { ...(m.mask || {}), text: g.segment };
      if (g.recommend === "skip" && (m.force || "auto") === "auto") m.force = "skip";
    });
    save();
  };
  const setCastOpt = (k, v) => { plan[k] = v; save(); if (k === "cast_split") autoSplit(true); };
  const setFilter = (k, v) => { plan.filters = { ...plan.filters, [k]: v }; save(); if (stats.value.length) analyzeContent(); };
  // XHR instead of fetch: big videos show how much has been sent
  const postWithProgress = (url, body, onPct) => new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", api.apiURL(url));
    if (api.user) x.setRequestHeader("Comfy-User", api.user);
    x.upload.onprogress = e => { if (e.lengthComputable) onPct(e.loaded / e.total); };
    x.onload = () => (x.status >= 200 && x.status < 300) ? resolve(JSON.parse(x.responseText || "{}")) : reject(new Error(`HTTP ${x.status}`));
    x.onerror = () => reject(new Error("network error"));
    x.send(body);
  });
  async function upload(file, cb) {
    const fd = new FormData(); fd.append("image", file); fd.append("type", "input"); fd.append("overwrite", "true");
    const mb = (file.size / 1e6).toFixed(1);
    busy.value = `Uploading ${file.name} (${mb} MB)…`; upPct.value = 0;
    try {
      const j = await postWithProgress("/upload/image", fd, f => { upPct.value = f; });
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
  // Delete / Backspace on a selected shot: remove the cut at its start (merge into the previous shot;
  // the first shot merges with the next one)
  const removeCut = i => {
    const S = segs.value, s = S[i]; if (!s || S.length < 2) return;
    if (i === 0) { mergeNext(0); sel.value = 0; return; }
    plan.bounds = plan.bounds.filter(b => b !== s.start); plan.segs.splice(i, 1);
    sel.value = i - 1; save();
  };
  const onKey = e => {
    if (e.key !== "Delete" && e.key !== "Backspace") return;
    const t = e.target, tag = (t?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || t?.isContentEditable) return;
    e.preventDefault(); e.stopPropagation();   // keep ComfyUI from deleting the node
    removeCut(sel.value);
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

  // ---- preview player
  const vid = ref(null);
  const play = reactive({ mode: "", idx: -1, frame: 0, loop: false });
  const stopAt = () => { const s = segs.value[play.idx]; return s ? s.end / plan.fps : Infinity; };
  const tick = () => {
    const v = vid.value; if (!v) return;
    play.frame = Math.floor(v.currentTime * plan.fps + 1e-3);
    if (!play.mode) return;
    if (play.mode === "all") {
      const i = segs.value.findIndex(s => play.frame >= s.start && play.frame < s.end);
      if (i >= 0 && skipWhy(segs.value[i])) {            // jump over shots that will not run
        const nxt = segs.value.slice(i + 1).find(s => !skipWhy(s));
        if (nxt) v.currentTime = nxt.start / plan.fps + 1e-3; else { v.pause(); play.mode = ""; }
        return;
      }
      if (i >= 0) play.idx = i;
      if (play.frame >= n.value) { v.pause(); play.mode = ""; }
    } else if (v.currentTime >= stopAt() - 1e-3) {
      const s = segs.value[play.idx];
      if (play.loop && s) v.currentTime = s.start / plan.fps + 1e-3; else { v.pause(); play.mode = ""; v.currentTime = stopAt() - 0.5 / plan.fps; }
    }
  };
  let raf = 0;
  const loop = () => { tick(); raf = requestAnimationFrame(loop); };
  const playShot = i => {
    const v = vid.value, s = segs.value[i]; if (!v || !s) return;
    sel.value = i; play.idx = i; play.mode = "shot";
    v.currentTime = s.start / plan.fps + 1e-3; v.play();
  };
  const playAll = () => {
    const v = vid.value; if (!v) return;
    play.mode = "all"; const first = segs.value.find(s => !skipWhy(s)) || segs.value[0];
    play.idx = segs.value.indexOf(first); v.currentTime = (first?.start || 0) / plan.fps + 1e-3; v.play();
  };
  const stop = () => { vid.value?.pause(); play.mode = ""; };
  const seek = f => { const v = vid.value; if (v) { v.currentTime = f / plan.fps + 1e-3; play.frame = f; } };

  // ---- queue loop events
  const onProg = e => { prog.done = e.detail.done; prog.count = e.detail.count; };
  const onStatus = e => { if (busy.value || modal.busy) Object.assign(status, e.detail || {}); };
  let clock = 0;
  const onVlm = e => { if (e.detail?.video === plan.video) mergeSug(e.detail.segs); };
  const onNext = e => { onProg(e); if (plan.run === "queue" && plan.auto_continue) setTimeout(() => app.queuePrompt(0, 1), 300); };
  onMounted(() => {
    load(); refreshFiles(); analyze(); progress(); raf = requestAnimationFrame(loop);
    api.addEventListener("bfs-shotloop-progress", onProg); api.addEventListener("bfs-shotloop-next", onNext);
    api.addEventListener("bfs-shotloop-vlm", onVlm);
    api.addEventListener("bfs-shotloop-status", onStatus);
    clock = setInterval(() => { if (busy.value || modal.busy) now.value = Date.now(); }, 500);
  });
  onBeforeUnmount(() => { cancelAnimationFrame(raf); api.removeEventListener("bfs-shotloop-progress", onProg); api.removeEventListener("bfs-shotloop-next", onNext); api.removeEventListener("bfs-shotloop-vlm", onVlm); api.removeEventListener("bfs-shotloop-status", onStatus); clearInterval(clock); });
  io.expose({ reload: () => { load(); analyze(); progress(); } });

  // ---- view helpers
  const fld = (label, input, hint) => {
    const tip = Object.entries(TIPS).find(([k]) => String(label).startsWith(k))?.[1] || "";
    return h("div", { class: "fld", title: tip }, [h("label", tip ? [label, h("span", { class: "qm" }, " ⓘ")] : label), input, hint ? h("div", { class: "hint" }, hint) : null]);
  };
  const num = (k, step = 0.1, min = 0) => h("input", { type: "number", step, min, value: plan[k], onChange: e => { plan[k] = parseFloat(e.target.value) || 0; save(); } });
  const sel_ = (k, opts, after) => h("select", { value: plan[k], onChange: e => { plan[k] = e.target.value; save(); after && after(); } },
    opts.map(o => h("option", { value: Array.isArray(o) ? o[0] : o }, Array.isArray(o) ? o[1] : o)));
  // target: shot index or "global"; field: ref | ref2 (global_ref | global_ref2 for "global")
  // the last 10 references picked, remembered across workflows (per browser)
  const RECENT_KEY = "bfs.shotloop.recentRefs";
  const recent = ref((() => { try { return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); } catch { return []; } })());
  const remember = name => {
    if (!name) return;
    recent.value = [name, ...recent.value.filter(x => x !== name)].slice(0, 10);
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(recent.value)); } catch { /* private mode */ }
  };
  const setRef = (target, field, name) => {
    if (target === "global") plan[field] = name; else meta(target)[field] = name;
    remember(name); save();
  };
  const uploadRef = (target, field) => pickFile("image/*", name => setRef(target, field, name));
  const refAll = (field, name) => { segs.value.forEach((_, i) => { meta(i)[field] = name; }); save(); };
  const usedRefs = computed(() => {
    const set = new Set([...recent.value, plan.global_ref, plan.global_ref2, ...plan.segs.flatMap(m => [m?.ref, m?.ref2])].filter(Boolean));
    return [...set].slice(0, 10);
  });
  const refSlot = (name, label, target, field, perShot) => h("div", { class: "rslot" }, [
    h("div", { class: "refslot", title: name ? name + " (click to replace)" : "click to upload", onClick: () => uploadRef(target, field) },
      name ? [h("img", { src: viewUrl(name) })] : [label]),
    h("div", { class: "rbtns" }, [
      name ? h("button", { title: "remove", onClick: () => setRef(target, field, "") }, "✕") : null,
      name && perShot ? h("button", { title: "use this reference for every shot", onClick: () => refAll(field, name) }, "→ all") : null,
    ]),
  ]);

  return () => {
    const S = segs.value, cur = S[sel.value], N = n.value, ppf = pxPerFrame.value, fps = plan.fps;
    const tooLong = S.filter(s => s.len > maxLen.value).length;
    const totalGen = active.value.reduce((a, s) => a + s.gen, 0);

    const busyBar = (msg) => {
      const secs = busySince.value ? Math.max(0, Math.round((now.value - busySince.value) / 1000)) : 0;
      const pct = upPct.value >= 0 ? upPct.value : (status.total > 0 ? Math.min(1, status.done / status.total) : -1);
      const stage = status.stage && !msg.startsWith(status.stage) ? status.stage : "";
      return h("div", { class: "busybar" }, [
        h("div", { class: "brow" }, [h("span", { class: "spin" }), h("b", msg), stage ? h("span", { class: "hint" }, `· ${stage}`) : null,
          h("span", { class: "grow" }), pct >= 0 ? h("span", `${Math.round(pct * 100)}%`) : null, h("span", { class: "hint" }, `${secs}s`)]),
        h("div", { class: ["bprog", pct < 0 && "ind"] }, [h("div", { style: pct >= 0 ? `width:${(pct * 100).toFixed(1)}%` : "" })]),
      ]);
    };
    const guideUrl = new URL("./docs/BFSShotPlanner.md", import.meta.url).href;
    const header = h("div", { class: "hdr" }, [
      h("span", { class: "ttl" }, "🎬 Shot Planner"),
      busy.value ? h("span", { class: "pill warn" }, busy.value) : (an.value ? h("span", { class: "pill ok" }, `${active.value.length} shots · ${(N / fps).toFixed(1)}s`) : null),
      h("span", { class: "grow" }),
      plan.run === "queue" ? h("span", { class: "pill" }, "queue loop") : h("span", { class: "pill" }, "auto loop"),
      h("button", { title: "Open the full guide (every setting explained, with examples)", onClick: () => showGuide(guideUrl) }, "📖 Guide"),
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
      an.value ? (au => h("div", { class: "row", style: "margin-top:4px;gap:8px" }, [
        au?.has_audio
          ? h("span", { class: "pill ok", title: au.codec || "" }, `🔊 audio · ${au.sample_rate ? (au.sample_rate / 1000).toFixed(1) + " kHz" : "?"} · ${au.channels || "?"} ch`)
          : h("span", { class: "pill warn", title: "The outputs carry a silent track of the right length, so Create Video works" }, `🔇 ${au?.reason || "no audio"} → silent track`),
        h("select", { value: plan.audio_mode, style: "width:auto", disabled: !au?.has_audio,
          title: "Audio of the planner's and the join's outputs",
          onChange: e => { plan.audio_mode = e.target.value; save(); } },
          [h("option", { value: "auto" }, "use the video's audio"), h("option", { value: "silent" }, "silent track")]),
      ]))(an.value.audio) : null,
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
        h("span", { class: "hint" }, "Drag the white handles to move a boundary · double-click the shot bar to split there · select a shot and press Delete to remove its cut"),
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
        onPointermove: e => { if (!e.buttons) hover.value = frameAt(e); }, onPointerleave: () => { hover.value = null; },
        onClick: e => { if (e.target.classList.contains("hdl")) return; seek(frameAt(e).frame); } }, [
        h("div", { class: "tlin", style: `width:${tlWidth.value}px` }, [
          h("div", { class: "strip" }, thumbs.map(t => h("img", { src: t.src, style: `width:${thumbW}px` }))),
          ...cuts.value.filter(c => c < N).map(c => h("div", { class: "cut", style: `left:${c * ppf}px`, title: `cut @ ${c}` })),
          h("svg", { class: "spark", viewBox: `0 0 ${tlWidth.value} 26`, preserveAspectRatio: "none" },
            [h("polyline", { points: sparkPts, fill: "none", stroke: "#ff7a90", "stroke-width": 1, "vector-effect": "non-scaling-stroke" })]),
          h("div", { class: "segs", onDblclick: e => splitAt(frameAt(e).frame) }, [
            ...S.map((s, i) => h("div", {
              class: ["seg", i === sel.value && "sel", !!skipWhy(s) && "off", s.len > maxLen.value && "long", play.mode && play.idx === i && "playing"],
              style: `left:${s.start * ppf}px;width:${Math.max(2, s.len * ppf - 1)}px;background:${hue(i)}`,
              title: `#${i + 1} · frames ${s.start}-${s.end - 1} · ${s.len} → ${s.gen}${skipWhy(s) ? " · skip: " + skipWhy(s) : ""}`, onClick: () => { sel.value = i; },
            }, s.len * ppf > 26 ? `${i + 1}` : "")),
            ...plan.bounds.filter(b => b < N).map((b, k) => h("div", { class: "hdl", style: `left:${b * ppf}px`, title: `boundary @ ${b}`, onPointerdown: e => drag(k, e) })),
          ]),
          hover.value ? h("div", { class: "ph", style: `left:${hover.value.x}px` }) : null,
          vid.value && plan.video ? h("div", { class: "playhead", style: `left:${Math.min(N, play.frame) * ppf}px` }) : null,
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

    // cast: people found by face, each linked to a reference
    const castN = people.value.filter(p => linked(p.id)).length;
    const cchk = (k, label, title) => h("label", { class: "row", style: "gap:4px", title }, [h("input", { type: "checkbox", checked: !!plan[k], onChange: e => setCastOpt(k, e.target.checked) }), label]);
    const cast = an.value ? h("details", { class: "card", open: people.value.length > 0 }, [
      h("summary", h("h5", ["Cast", people.value.length ? h("span", { class: "pill" }, `${people.value.length} people · ${castN} linked`) : h("span", { class: "pill" }, "not analysed"),
        h("span", { class: "hint", style: "text-transform:none;letter-spacing:0" }, "track faces, link each person to a reference, split and assign shots by who is on screen")])),
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        cchk("cast_assign", "Shots use their main person's reference", "a shot without its own reference takes the reference of the person with the most screen time in it"),
        cchk("cast_split", "Split where the main person changes", "adds a boundary when the biggest face switches to someone else (re-splits the plan)"),
        cchk("cast_only", "Only run shots with a linked person", "shots where no linked person appears are skipped"),
      ]),
      people.value.length ? h("div", { class: "cast" }, people.value.map(p => {
        const c = castOf(p.id);
        return h("div", { class: ["person", c.ignore && "ign"] }, [
          h("img", { class: "face", src: p.thumb, title: `first seen ${fmtT(p.first, fps)} (click to seek)`, onClick: () => seek(p.first) }),
          h("div", { class: "t" }, [h("b", `Person ${p.id}`), ` · ${(p.share * 100).toFixed(1)}%`]),
          h("div", { class: "t" }, `${fmtT(p.first, fps)} → ${fmtT(p.last, fps)}`),
          h("div", { class: "refbox" }, ["ref", "ref2"].map(k => h("div", { class: "rslot" }, [
            h("div", { class: "refslot sm", title: c[k] ? c[k] + " (click to replace)" : "click to upload",
              onClick: () => pickFile("image/*", name => setCast(p.id, k, name)) }, c[k] ? [h("img", { src: viewUrl(c[k]) })] : [k === "ref" ? "⬆ ref" : "⬆ ref 2"]),
            c[k] ? h("div", { class: "rbtns" }, [h("button", { title: "remove", onClick: () => setCast(p.id, k, "") }, "✕")]) : null,
          ]))),
          usedRefs.value.length ? h("div", { class: "recent sm" }, usedRefs.value.map(n => h("img", { src: viewUrl(n), title: `${n} (click = ref, shift+click = ref 2)`,
            onClick: e => setCast(p.id, e.shiftKey ? "ref2" : "ref", n) }))) : null,
          h("label", { class: "row", style: "gap:4px" }, [h("input", { type: "checkbox", checked: !!c.ignore, onChange: e => setCast(p.id, "ignore", e.target.checked) }), "ignore"]),
        ]);
      })) : null,
      h("div", { class: "row", style: "margin-top:8px" }, [
        h("button", { class: "pri", disabled: !!busy.value, onClick: () => findPeople(true) }, people.value.length ? "↻ Re-analyse people" : "👥 Find people"),
        h("span", { class: "hint" }, people.value.length ? "faces sampled every 0.5 s and grouped by identity (InsightFace, CPU)" : "detects faces across the video and groups them into people"),
      ]),
    ]) : null;

    // global mask settings
    const MC = plan.mask_cfg || {};
    const mnum = (k, step, label, hint, scale = 1) => fld(label, h("input", { type: "number", step, min: 0, value: +(MC[k] * scale).toFixed(3),
      onChange: e => setMaskCfg(k, (parseFloat(e.target.value) || 0) / scale) }), hint);
    const mchk = (k, label, title) => h("label", { class: "row", style: "gap:4px", title }, [h("input", { type: "checkbox", checked: !!MC[k], onChange: e => setMaskCfg(k, e.target.checked) }), label]);
    const cropN = S.filter(x => x.crop).length;
    const masks = an.value ? h("details", { class: "card", open: cropN > 0 }, [
      h("summary", h("h5", ["Mask & crop", cropN ? h("span", { class: "pill warn" }, `${cropN} cropped`) : h("span", { class: "pill" }, "off"),
        h("span", { class: "hint", style: "text-transform:none;letter-spacing:0" }, "SAM 3 per shot (text or points) · generate only the masked region and paste it back")])),
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        mchk("fill_holes", "Fill holes", "Fill enclosed gaps so each region is solid"),
        mchk("invert", "Invert", "Use everything except the segmented object"),
        h("label", { class: "row", style: "gap:4px", title: "Overlay the mask preview on the shot cards" }, [
          h("input", { type: "checkbox", checked: showMasks.value, onChange: e => { showMasks.value = e.target.checked; } }), "Show masks on shots"]),
      ]),
      h("div", { class: "grid" }, [
        mnum("padding", 1, "Crop padding (%)", "context around the mask's box", 100),
        mnum("expand", 1, "Expand (px)", "grow the mask before pasting back"),
        mnum("feather", 1, "Feather (px)", "soft edge of the paste"),
        mnum("temporal_expand", 1, "Temporal expand (frames)", "hold the mask a few frames: less flicker"),
        mnum("blockify", 1, "Blockify (px, 0 = off)", "square blocks; 16 matches H3's latent grid"),
        mnum("threshold", 0.05, "Detection threshold", "SAM 3 score to keep an object"),
        mnum("max_objects", 1, "Max objects", "how many matches of the text are tracked"),
        fld("Paste back", h("select", { value: MC.paste, onChange: e => setMaskCfg("paste", e.target.value) },
          [h("option", { value: "mask" }, "only the mask (feathered)"), h("option", { value: "box" }, "the whole box (feathered)")])),
      ]),
      h("div", { class: "hint", style: "margin-top:4px" }, "Uses the official sam3.1_multiplex_fp16 checkpoint (models/checkpoints), downloaded from Comfy-Org/sam3.1 the first time. Each shot sets what to segment in its editor."),
    ]) : null;

    // VLM
    const VC = plan.vlm_cfg || {};
    const vchk = (k, label, title) => h("label", { class: "row", style: "gap:4px", title }, [h("input", { type: "checkbox", checked: !!VC[k], onChange: e => setVlmCfg(k, e.target.checked) }), label]);
    const sugN = Object.keys(vlmSug.value).length;
    const vlmCard = an.value ? h("details", { class: "card", open: !!VC.enabled || sugN > 0 }, [
      h("summary", h("h5", ["VLM", VC.enabled ? h("span", { class: "pill ok" }, "on") : h("span", { class: "pill" }, "off"),
        sugN ? h("span", { class: "pill" }, `${sugN} suggestions`) : null,
        h("span", { class: "hint", style: "text-transform:none;letter-spacing:0" }, "connect a Qwen3-VL (CLIPLoader) to the planner's vlm input: it looks at every shot and suggests settings")])),
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        vchk("enabled", "Use the VLM when the workflow runs", "At run time the planner asks the VLM about the shots it runs and applies the options below"),
        vchk("auto_segment", "Fill the mask text", "Shots without a mask text or points get the VLM's segment suggestion"),
        vchk("auto_shot", "Fill {shot} in the prompt", "Write {shot} in a prompt: it becomes the VLM's description of that shot (camera, framing, action)"),
      ]),
      h("div", { class: "grid" }, [
        fld("Frames per shot", h("input", { type: "number", min: 1, max: 8, step: 1, value: VC.frames, onChange: e => setVlmCfg("frames", parseInt(e.target.value) || 3) })),
        fld("Max tokens", h("input", { type: "number", min: 64, max: 2048, step: 32, value: VC.max_tokens, onChange: e => setVlmCfg("max_tokens", parseInt(e.target.value) || 1024) })),
      ]),
      h("textarea", { style: "margin-top:6px;min-height:44px", placeholder: "Extra instruction for the VLM (optional), e.g. 'segment the woman, not the man'",
        value: VC.instruction || "", onChange: e => setVlmCfg("instruction", e.target.value) }),
      h("h5", { style: "margin-top:10px" }, ["Describe references", h("span", { class: "hint", style: "text-transform:none;letter-spacing:0" },
        "write {details} in any prompt: each shot gets the description of its own references")]),
      h("div", { class: "row", style: "gap:8px" }, [
        h("select", { value: VC.describe_preset, style: "width:auto", onChange: e => setVlmCfg("describe_preset", e.target.value) },
          ["full body", "head / face", "face attributes", "outfit", "custom"].map(o => h("option", { value: o }, o))),
        h("button", { class: "pri", disabled: !!busy.value || !refSets.value.length, onClick: describeRefs }, "📝 Describe refs"),
        h("span", { class: "hint" }, `${refSets.value.length} reference set(s) · editable below`),
      ]),
      VC.describe_preset === "custom" ? h("textarea", { style: "margin-top:6px;min-height:52px", value: VC.describe_custom || "",
        placeholder: "Your instruction for the VLM, e.g. 'Describe the character's costume and props piece by piece…'",
        onChange: e => setVlmCfg("describe_custom", e.target.value) }) : null,
      refSets.value.length ? h("div", { class: "dlist" }, refSets.value.map(x => h("div", { class: "drow" }, [
        h("div", { class: "dthumbs" }, [x.a ? h("img", { src: viewUrl(x.a) }) : null, x.b ? h("img", { src: viewUrl(x.b) }) : null]),
        h("div", { style: "flex:1" }, [
          h("div", { class: "hint" }, x.where.length > 4 ? `${x.where.slice(0, 4).join(", ")} +${x.where.length - 4}` : x.where.join(", ")),
          h("textarea", { style: "min-height:44px", value: plan.ref_details?.[x.key] || "",
            placeholder: "no description yet: Describe refs, write your own, or leave it to the VLM at run time",
            onChange: e => setDetail(x.key, e.target.value) }),
        ]),
      ]))) : null,
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("button", { class: "pri", disabled: !!busy.value, onClick: analyseVLM }, "🤖 Analyse shots"),
        h("button", { disabled: !sugN, onClick: applyAllSug, title: "Mask text for shots without one, skip where the VLM says skip" }, "Apply suggestions → all"),
        h("span", { class: "hint" }, "The VLM buttons work after the workflow has run once with the VLM connected (ComfyUI only hands models to nodes when they run)."),
      ]),
    ]) : null;

    // shot cards
    const cards = S.length ? h("div", { class: "card" }, [
      h("h5", "Shots"),
      h("div", { class: "shots" }, S.map((s, i) => h("div", { class: ["sc", i === sel.value && "sel"], onClick: () => { sel.value = i; } }, [
        h("button", { class: "play", title: "play this shot", onClick: e => { e.stopPropagation(); playShot(i); } }, "▶"),
        h("div", { class: "bar", style: `background:${hue(i)}` }),
        h("div", { class: "row" }, [h("b", `#${i + 1}`), s.cut ? h("span", { class: "pill" }, "cut") : null,
          skipWhy(s) ? h("span", { class: "pill warn", title: skipWhy(s) }, "skip") : h("span", { class: "pill ok" }, "run"),
          s.force !== "auto" ? h("span", { class: "pill" }, s.force) : null,
          i > 0 && s.chain !== "off" ? h("span", { class: "pill", title: `continues from the previous shot's ${s.chainFrame} frame as ${s.chain}` }, "⛓") : null,
          s.crop ? h("span", { class: "pill", title: `cropped to: ${s.mask.text || ((s.mask.points || []).length + " points")}` }, "✂") : null]),
        whoIn(s) ? h("div", { class: "who" }, whoIn(s).people.length ? whoIn(s).people.map(id => h("img", {
          src: personOf(id)?.thumb || "", title: `Person ${id}${id === whoIn(s).main ? " (main)" : ""}${linked(id) ? " · linked" : ""}`,
          class: [id === whoIn(s).main && "main", linked(id) && "lk"] })) : [h("span", { class: "t" }, "no faces")]) : null,
        statFor(s) ? h("div", { class: "t", style: "margin-top:2px" },
          `👤 ${statFor(s).persons} · ${(statFor(s).person_area * 100).toFixed(1)}% · 🙂 ${statFor(s).faces} · ☀ ${(statFor(s).brightness * 100).toFixed(0)}%`) : null,
        skipWhy(s) ? h("div", { class: "t", style: "color:#ffc46b" }, skipWhy(s)) : null,
        h("div", { class: "t" }, `${fmtT(s.start, fps)} → ${fmtT(s.end, fps)} · ${s.len}f → ${s.gen}f`),
        h("div", { class: "thumbs" }, [
          ...["ref", "ref2"].map(k => {
            const own = s[k], via = castRef(s, k), name = own || via || plan["global_" + k];
            return name ? h("img", { class: "rt", src: viewUrl(name), title: own ? name : via ? `${name} (from the person)` : `${name} (global)`,
              style: own ? "" : via ? "outline:1px dashed #8fd18f" : "opacity:.45" }) : h("div", { class: "rt ph2" }, k);
          }),
          h("img", { class: "rt", style: "width:56px", src: showMasks.value && maskPrev.value[`${s.start}-${s.end}`]?.frames?.length
            ? maskPrev.value[`${s.start}-${s.end}`].frames[Math.floor(maskPrev.value[`${s.start}-${s.end}`].frames.length / 2)].src
            : thumbFor(s.start + Math.floor(s.len / 2)) }),
        ]),
        h("div", { class: "p" }, s.prompt ? s.prompt : (plan.global_prompt ? "↳ global prompt" : "— no prompt —")),
      ]))),
    ]) : null;

    // editor for the selected shot
    const ps = S[play.idx] || cur;
    const player = an.value && plan.video ? h("div", { class: "card" }, [
      h("h5", ["Preview", play.mode ? h("span", { class: "pill warn" }, play.mode === "all" ? "playing all" : "playing shot") : null]),
      h("div", { class: "player" }, [
        h("video", { ref: vid, src: viewUrl(plan.video), preload: "metadata", muted: false, playsinline: true,
          onPause: () => { if (play.mode) play.mode = ""; } }),
        h("div", { class: "pinfo" }, [
          h("div", { class: "row" }, [
            h("button", { class: "pri", disabled: !cur, onClick: () => playShot(sel.value) }, `▶ Play shot #${sel.value + 1}`),
            h("button", { onClick: playAll }, "▶ Play all"),
            h("button", { onClick: stop }, "■ Stop"),
            h("label", { class: "row", style: "gap:4px" }, [h("input", { type: "checkbox", checked: play.loop, onChange: e => { play.loop = e.target.checked; } }), "loop shot"]),
          ]),
          ps ? h("div", { class: "tc" }, [
            h("div", ["Shot ", h("b", `#${S.indexOf(ps) + 1}`), skipWhy(ps) ? `  (skipped: ${skipWhy(ps)})` : ""]),
            h("div", ["start ", h("b", fmtT(ps.start, fps)), `  (frame ${ps.start})`]),
            h("div", ["end   ", h("b", fmtT(ps.end, fps)), `  (frame ${ps.end - 1})  ·  ${(ps.len / fps).toFixed(2)}s`]),
            h("div", ["now   ", h("b", fmtT(Math.min(N, play.frame), fps)), `  (frame ${Math.min(N, play.frame)})`]),
          ]) : null,
          h("div", { class: "hint" }, "Click the timeline to seek. Play all skips shots that will not run."),
        ]),
      ]),
    ]) : null;

    const editor = cur ? h("div", { class: "card" }, [
      h("h5", [`Shot #${sel.value + 1}`, h("span", { class: "hint", style: "text-transform:none" }, `frames ${cur.start}–${cur.end - 1} · ${cur.len} → generate ${cur.gen}`)]),
      h("div", { class: "row", style: "align-items:flex-start;gap:10px" }, [
        h("div", { class: "refbox" }, [
          refSlot(cur.ref, "⬆ reference\n(uses global)", sel.value, "ref", true),
          refSlot(cur.ref2, "⬆ ref 2\n(uses global)", sel.value, "ref2", true),
        ]),
        h("div", { style: "flex:1;min-width:200px" }, [
          h("textarea", { placeholder: "Prompt for this shot (empty = global prompt)", value: cur.prompt, onChange: e => setMeta(sel.value, "prompt", e.target.value) }),
        ]),
      ]),
      usedRefs.value.length ? h("div", { class: "recent" }, [h("span", { class: "hint" }, "recent (click = ref, shift+click = ref 2):"),
        ...usedRefs.value.map(n => h("img", { src: viewUrl(n), title: n, onClick: e => setRef(sel.value, e.shiftKey ? "ref2" : "ref", n) }))]) : null,
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("select", { value: cur.force, style: "width:auto", title: "Override the content filters for this shot",
          onChange: e => setMeta(sel.value, "force", e.target.value) },
          [h("option", { value: "auto" }, "filters decide"), h("option", { value: "run" }, "always run"), h("option", { value: "skip" }, "always skip")]),
        h("button", { onClick: () => setMeta(sel.value, "enabled", !cur.enabled) }, cur.enabled ? "⏸ Disable" : "▶ Enable"),
        h("button", { onClick: () => splitAt(cur.start + Math.floor(cur.len / 2)) }, "✂ Split in half"),
        h("button", { disabled: sel.value >= S.length - 1, onClick: () => mergeNext(sel.value) }, "⇥ Merge with next"),
        h("button", { onClick: () => applyAll("prompt") }, "Prompt → all"),
        h("button", { class: "dng", onClick: () => { ["ref", "ref2", "prompt"].forEach(k => { meta(sel.value)[k] = ""; }); save(); } }, "Use global"),
      ]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("span", { class: "hint", title: "Uses a frame of the PREVIOUS shot's generated result. Works in the queue loop, and in the auto loop with BFS Shot H3 Duet (it renders shot by shot)." }, "Continuity:"),
        h("select", { value: cur.chain, style: "width:auto", disabled: sel.value === 0,
          title: "reference: the previous result's frame becomes one more <Picture n> after this shot's own references. first frame: it is anchored at frame 0 of this shot.",
          onChange: e => setMeta(sel.value, "chain", e.target.value) },
          [h("option", { value: "off" }, "off"), h("option", { value: "reference" }, "previous shot as reference"),
           h("option", { value: "first frame" }, "previous shot as first frame")]),
        h("select", { value: cur.chainFrame, style: "width:auto", disabled: sel.value === 0 || cur.chain === "off",
          title: "Which frame of the previous shot's result", onChange: e => setMeta(sel.value, "chain_frame", e.target.value) },
          [h("option", { value: "first" }, "its first frame"), h("option", { value: "middle" }, "its middle frame"), h("option", { value: "last" }, "its last frame")]),
        h("button", { title: "Use this continuity setting for every shot after the first", onClick: () => {
          const c = meta(sel.value).chain || "off", f = meta(sel.value).chain_frame || "first";
          segs.value.forEach((_, i) => { if (i > 0) { meta(i).chain = c; meta(i).chain_frame = f; } }); save();
        } }, "Continuity → all"),
        sel.value === 0 ? h("span", { class: "hint" }, "the first shot uses only its references") : null,
      ]),
      h("div", { class: "row", style: "margin-top:8px;align-items:center" }, [
        h("label", { class: "row", style: "gap:4px", title: "Generate only the masked region: the shot is cropped to one box around the SAM 3 mask (the union over all its frames), and BFS Shot Join pastes the result back, feathered by the mask." }, [
          h("input", { type: "checkbox", checked: cur.crop, onChange: e => setMeta(sel.value, "crop", e.target.checked) }), "✂ Crop to mask"]),
        h("input", { type: "text", value: cur.mask.text || "", style: "flex:1;min-width:160px",
          placeholder: "what to segment, in English: person in white, red car…",
          title: "SAM 3 text prompt (up to 32 tokens; separate several things with commas). Points, when set, take priority.",
          onChange: e => setMask(sel.value, "text", e.target.value) }),
        h("button", { title: "Pick positive / negative points on a frame of this shot", onClick: () => openPoints(sel.value) },
          (cur.mask.points || []).length ? `🎯 Points (${cur.mask.points.length})` : "🎯 Points…"),
        h("button", { disabled: !!busy.value || !(cur.mask.text || (cur.mask.points || []).length), onClick: () => previewMask(sel.value) }, "👁 Preview mask"),
        h("button", { title: "Use this text prompt for every shot (points stay per shot)", onClick: () => {
          const t = cur.mask.text || ""; segs.value.forEach((_, i) => { const m = meta(i); m.mask = { ...(m.mask || {}), text: t }; }); save();
        } }, "Mask → all"),
        h("button", { title: "Crop every shot that has a mask", onClick: () => {
          segs.value.forEach((x, i) => { if (x.mask.text || (x.mask.points || []).length || cur.mask.text) meta(i).crop = cur.crop; }); save();
        } }, "Crop → all"),
      ]),
      maskPrev.value[`${cur.start}-${cur.end}`] ? h("div", { class: "mstrip" }, [
        ...maskPrev.value[`${cur.start}-${cur.end}`].frames.map(f => h("img", { src: f.src, title: `frame ${f.f}` })),
        h("span", { class: "hint" }, maskPrev.value[`${cur.start}-${cur.end}`].empty ? "nothing found: the shot runs uncropped"
          : `mask covers ${(maskPrev.value[`${cur.start}-${cur.end}`].coverage * 100).toFixed(1)}% · yellow = crop box`),
      ]) : null,
      vlmSug.value[`${cur.start}-${cur.end}`] ? (g => h("div", { class: "vsug" }, [
        h("div", ["🤖 ", h("b", "segment: "), g.segment || "—", g.segment ? h("button", { onClick: () => applySug(sel.value, "segment") }, "Use as mask") : null]),
        h("div", [h("b", "shot: "), g.shot || "—", g.shot ? h("button", { title: "copy (the {shot} placeholder in the prompt gets it automatically at run time)",
          onClick: () => navigator.clipboard?.writeText(g.shot) }, "Copy") : null]),
        h("div", [h("b", "recommend: "), `${g.recommend}${g.people != null ? " · " + g.people + " people" : ""}${g.reason ? " · " + g.reason : ""}`,
          g.recommend === "skip" ? h("button", { onClick: () => applySug(sel.value, "skip") }, "Skip this shot") : null]),
        g.raw ? h("div", { class: "hint" }, "unparsed answer: " + g.raw.slice(0, 200)) : null,
      ]))(vlmSug.value[`${cur.start}-${cur.end}`]) : null,
    ]) : null;

    const globals = h("div", { class: "card" }, [
      h("h5", "Global (used by shots without their own)"),
      h("div", { class: "row", style: "align-items:flex-start;gap:10px" }, [
        h("div", { class: "refbox" }, [
          refSlot(plan.global_ref, "⬆ global\nreference", "global", "global_ref", false),
          refSlot(plan.global_ref2, "⬆ global\nref 2", "global", "global_ref2", false),
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

    const ms = S[modal.idx];
    const pointsModal = modal.open && ms ? h(Teleport, { to: "body" }, h("div", { class: "bsl", style: "background:none;padding:0" }, h("div", { class: "mmodal", onKeydown: e => e.stopPropagation(), onPointerdown: e => { if (e.target === e.currentTarget) modal.open = false; } }, [
      h("div", { class: "mbox" }, [
        h("h5", [`Shot #${modal.idx + 1} · points`, h("span", { class: "hint", style: "text-transform:none" },
          "click = keep (green) · right-click or shift+click = exclude (red)")]),
        h("div", { class: "mimg", onContextmenu: e => e.preventDefault(), onPointerdown: e => {
          const box = e.currentTarget.getBoundingClientRect();
          const x = (e.clientX - box.left) / box.width, y = (e.clientY - box.top) / box.height;
          modal.points = [...modal.points, { x: +x.toFixed(4), y: +y.toFixed(4), label: (e.button === 2 || e.shiftKey) ? 0 : 1 }];
          modal.prev = null;
        } }, [
          h("img", { draggable: false, src: api.apiURL(`/bfs/shotloop/frame?video=${encodeURIComponent(plan.video)}&fps=${plan.fps}&f=${ms.start + modal.key}&w=900`) }),
          ...modal.points.map((p, k) => h("div", { class: ["pt", p.label ? "pos" : "neg"], style: `left:${p.x * 100}%;top:${p.y * 100}%`,
            title: "click to remove", onPointerdown: e => { e.stopPropagation(); modal.points = modal.points.filter((_, j) => j !== k); modal.prev = null; } })),
        ]),
        h("div", { class: "row", style: "margin-top:6px" }, [
          h("span", { class: "hint" }, `frame ${ms.start + modal.key} (${modal.key + 1}/${ms.len})`),
          h("input", { type: "range", min: 0, max: ms.len - 1, step: 1, value: modal.key, style: "flex:1",
            onInput: e => { modal.key = parseInt(e.target.value); modal.prev = null; } }),
        ]),
        h("div", { class: "row", style: "margin-top:6px" }, [
          h("input", { type: "text", value: modal.text, style: "flex:1", placeholder: "text prompt (used when there are no points)",
            onChange: e => { modal.text = e.target.value; } }),
          h("button", { onClick: () => { modal.points = []; modal.prev = null; } }, "Clear points"),
          h("button", { class: "pri", disabled: !!modal.busy || !(modal.points.length || modal.text),
            onClick: () => previewMask(modal.idx, { points: modal.points, key: modal.key, text: modal.text }, true) }, "👁 Segment"),
        ]),
        modal.busy ? busyBar(modal.busy) : null,
        modal.prev ? h("div", { class: "mstrip" }, [...modal.prev.frames.map(f => h("img", { src: f.src, title: `frame ${f.f}` })),
          h("span", { class: "hint" }, modal.prev.empty ? "nothing found" : `covers ${(modal.prev.coverage * 100).toFixed(1)}%`)]) : null,
        h("div", { class: "row", style: "margin-top:8px;justify-content:flex-end" }, [
          h("button", { onClick: () => { modal.open = false; } }, "Cancel"),
          h("button", { class: "pri", onClick: savePoints }, "Save"),
        ]),
      ]),
    ]))) : null;

    return h("div", { class: "bsl", tabindex: 0, onKeydown: onKey }, [header, busy.value ? busyBar(busy.value) : null, error.value ? h("div", { class: "err" }, error.value) : null, source, settings, timeline, player, filters, cast, masks, vlmCard, cards, editor, globals, queue, pointsModal]);
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
