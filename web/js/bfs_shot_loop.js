/**
 * BFS Shot Planner: split a long video into model-sized shots on a timeline, give each shot its own
 * reference image and prompt, and run the rest of the graph once per shot.
 *
 * All state lives in the hidden `plan` STRING widget (JSON) so a workflow saves, reloads and shares
 * exactly what you set up. The panel is a Vue app mounted into a DOM widget; the server does the video
 * work (probe, thumbnails, PySceneDetect cuts, split) through /bfs/shotloop/* routes.
 *
 * This file holds the state and the actions; the views (one per tab) live in ./shotloop/.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { showGuide } from "./bfs_md_view.js";
import { wheelToCanvas } from "./bfs_wheel.js";
import { createApp, ref, reactive, computed, watch, nextTick, onMounted, onBeforeUnmount } from "./vendor/vue.esm-browser.prod.mjs";
import { h, rawH, t, uiLanguage, setUiLanguage, DEFAULTS, MASK_DEFAULTS, VLM_DEFAULTS, snapUp, snapDown, segKey, hasMask, pill } from "./shotloop/common.js";
import { styles } from "./shotloop/styles.js";
// history lengths H3's VAE encodes whole (17k+5); old plans saved 17/34/51, read as the next valid one up
const historyLen = n => (n = Number(n) || 22) <= 5 ? 5 : 5 + Math.ceil((n - 5) / 17) * 17;
import { timelineCard } from "./shotloop/view_timeline.js";
import { videoTab } from "./shotloop/view_video.js";
import { shotsTab } from "./shotloop/view_shots.js";
import { peopleTab } from "./shotloop/view_people.js";
import { promptsTab } from "./shotloop/view_prompts.js";
import { runTab } from "./shotloop/view_run.js";
import { busyBar, pointsModal, tourModal } from "./shotloop/view_modal.js";

const viewUrl = name => {
  if (!name) return "";
  const i = name.lastIndexOf("/");
  const sub = i >= 0 ? name.slice(0, i) : "", file = i >= 0 ? name.slice(i + 1) : name;
  return api.apiURL(`/view?filename=${encodeURIComponent(file)}&subfolder=${encodeURIComponent(sub)}&type=input`);
};
const store = {   // per-browser UI conveniences (never the plan itself)
  get: (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const DEFAULT_W = 1180, MIN_W = 1040, MIN_H = 1000;   // node size: wide enough for the four-column settings
const TABS = [["video", "🎞 Video"], ["shots", "🎬 Shots"], ["people", "👥 People / Filters / Masks"], ["prompts", "📝 Prompts & refs"], ["run", "▶ Run"]];

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
  const rootEl = ref(null);
  // keep the selected shot's card in view in the sideways strip
  watch(sel, () => nextTick(() => rootEl.value?.querySelector(".shots .sc.sel")?.scrollIntoView({ block: "nearest", inline: "nearest" })));
  const hover = ref(null);       // {frame, x}
  const prog = reactive({ done: 0, count: 0 });
  const cache = reactive({ ok: [], stored: [], mb: 0, folder: "", temp: false, combined: null });
  // what the server learned about memory on this GPU (BFS Shot Join records every shot it receives)
  const vram = reactive({ gpu: "", total_gb: 0, free_gb: 0, max_tokens: 0, max_peak_gb: 0, runs: 0, ram_free_gb: 0 });
  async function refreshVram() {
    try { const r = await api.fetchApi("/bfs/shotloop/vram"); const j = await r.json(); if (!j.error) Object.assign(vram, j); } catch { /* optional */ }
  }
  const tlEl = ref(null);
  const stats = ref([]);         // per-shot content stats from /bfs/shotloop/filters
  const people = ref([]);        // cast from /bfs/shotloop/cast: [{id, thumb, share, first, last}]
  const maskPrev = ref({});      // "start-end" -> {frames, box, coverage, empty} from /bfs/shotloop/mask
  const showMasks = ref(true);   // overlay the mask preview on the shot cards
  const vlmSug = ref({});        // "start-end" -> {segment, shot, people, recommend, reason} from the VLM
  const modal = reactive({ open: false, idx: -1, key: 0, points: [], text: "", region: 1, groups: {}, prev: null, busy: "" });
  const segPeople = ref({});     // "start-end" -> {people, main} from /bfs/shotloop/plan
  const targetCrop = ref({});    // "start-end" -> data URL of the subject the description was made from
  const tab = ref(store.get("bfs.shotloop.tab", "shots"));
  watch(tab, v => store.set("bfs.shotloop.tab", v));
  const tour = ref(0); // opened only by the user's Tour button; never shown automatically
  // what "Copy to other shots" copies (remembered per browser)
  const copyOpts = reactive(store.get("bfs.shotloop.copy", { ref: true, prompt: false, mask_text: false, points: false, crop: false,
                                                              target: false, chain: false, color_guide: false, scope: "all" }));
  watch(copyOpts, v => store.set("bfs.shotloop.copy", { ...v }), { deep: true });

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
    Object.values(plan.cast).forEach(person => { if (person && typeof person === "object") { delete person.ref; delete person.ref2; delete person.guide_color; } });
    plan.segs.forEach(meta => { if (meta) { delete meta.color_guide_person; delete meta.color_guide_person2; } });
  };
  // any change of the plan (a new video, a prompt, a setting...) is a new loop run on the server: show its progress
  let progTimer = 0;
  const save = () => { io.setPlan(JSON.stringify(plan)); clearTimeout(progTimer); progTimer = setTimeout(() => progress(), 400); };

  const n = computed(() => (an.value ? an.value.n : 0));      // the whole video on the timeline
  // the part the plan works on [lo, hi): start / end (0 = the end), then "max total seconds" counted from the start
  const range = computed(() => {
    const N = n.value; if (!N) return { lo: 0, hi: 0 };
    // frames (older plans saved seconds)
    const a = plan.start_f != null ? +plan.start_f : Math.round((+plan.start_s || 0) * plan.fps);
    const b = plan.end_f != null ? +plan.end_f : (+plan.end_s > 0 ? Math.round(+plan.end_s * plan.fps) : 0);
    const lo = Math.max(0, Math.min(N - 1, a));
    let hi = b > 0 ? Math.max(lo + 1, Math.min(N, b)) : N;
    if (+plan.max_total_s > 0) hi = Math.min(hi, lo + Math.max(1, Math.round(+plan.max_total_s * plan.fps)));
    return { lo, hi };
  });
  const maxLen = computed(() => snapDown(Math.round(plan.max_s * plan.fps), plan.grid));
  const segs = computed(() => {
    const N = n.value; if (!N) return [];
    const { lo, hi } = range.value;
    const b = [lo, ...plan.bounds.filter(x => x > lo && x < hi), hi].sort((a, c) => a - c);
    const cs = new Set(cuts.value);
    let out = [];
    for (let i = 0; i < b.length - 1; i++) {
      if (b[i + 1] <= b[i]) continue;
      const m = plan.segs[out.length] || {};
      out.push({ start: b[i], end: b[i + 1], len: b[i + 1] - b[i], gen: snapUp(b[i + 1] - b[i], plan.grid),
                 cut: cs.has(b[i]), enabled: m.enabled !== false, ref: m.ref || "", ref2: m.ref2 || "", prompt: m.prompt || "",
                 force: m.force || "auto", chain: m.chain || "head", chainFrame: m.chain_frame || "first", chainHistory: historyLen(m.chain_history),
                 color_guide: !!m.color_guide, color_guide_color: m.color_guide_color || "",
                 color_guide_opacity: m.color_guide_opacity ?? 0.35,
                 color_guide_text2: m.color_guide_text2 || "", color_guide_color2: m.color_guide_color2 || "",
                 color_guide_points2: m.color_guide_points2 || [], color_guide_key2: m.color_guide_key2,
                 crop: !!m.crop, inpaint: !!m.inpaint, paste: !!m.paste, pasteText: m.paste_text || "", strength: m.strength ?? 1, mask: m.mask || {}, target: m.target || "",
                 extMask: !!plan.mask_video || !!(plan.track?.points?.length), trackMask: !!(plan.track?.points?.length) });
    }
    if (plan.max_parts > 0) out = out.slice(0, plan.max_parts);
    return out;
  });
  // shots on the timeline before "Max shots" keeps the first ones: more than segs means some are left out
  const allSegCount = computed(() => {
    if (!n.value) return 0;
    const { lo, hi } = range.value;
    const b = [lo, ...plan.bounds.filter(x => x > lo && x < hi), hi].sort((a, c) => a - c);
    return b.reduce((k, x, i) => k + (i && x > b[i - 1] ? 1 : 0), 0);
  });
  const whoIn = s => segPeople.value[segKey(s)] || null;
  const castOf = id => plan.cast[String(id)] || {};
  const included = id => !!personOf(id) && !castOf(id).ignore;
  const personOf = id => people.value.find(p => p.id === id) || null;
  const statFor = s => stats.value.find(x => x.start === s.start && x.end === s.end) || null;
  const filtersOn = computed(() => { const f = plan.filters; return !!(f.person || f.face || f.skip_dark || f.skip_static || f.max_persons > 0 || f.min_frames > 0); });
  const skipWhy = s => {
    if (!s.enabled) return "disabled";
    if (s.force === "run") return "";
    if (s.force === "skip") return "skipped by hand";
    const why = filtersOn.value ? (statFor(s)?.skip_reason || "") : "";
    if (why) return why;
    const who = whoIn(s);
    if (plan.cast_only && who && !who.people.some(p => included(p))) return "no detected person";
    return "";
  };
  const active = computed(() => segs.value.filter(s => !skipWhy(s)));
  const pxPerFrame = computed(() => {
    const w = (tlEl.value?.clientWidth || 600) - 2;
    return Math.max(0.2, (w / Math.max(1, n.value)) * zoom.value);
  });
  const tlWidth = computed(() => Math.max(1, Math.round(n.value * pxPerFrame.value)));

  // how a shot is generated: crop (the planner crops, the join pastes back) x inpaint (only the mask is regenerated,
// read by BFS Shot H3 Conditioning with inpaint = per shot)
// [key, name, crop, inpaint, paste]; paste = the whole frame is generated and BFS Shot Join pastes only the person back
const MODES = [["frame", "Full frame", false, false, false], ["paste", "Frame + paste", false, false, true], ["mask", "Mask only", false, true, false],
               ["crop", "Crop", true, false, false], ["cropmask", "Crop + mask", true, true, false]];
const modeOf = s => (MODES.find(m => m[2] === !!s.crop && m[3] === !!s.inpaint && m[4] === (!!s.paste && !s.crop && !s.inpaint))
                     || MODES[0])[0];
const modeName = s => MODES.find(m => m[0] === modeOf(s))[1];

const meta = i => { while (plan.segs.length <= i) plan.segs.push({}); return plan.segs[i]; };
  // the reference a shot uses: its own or the global one
  const refOf = (s, k) => s[k] || plan["global_" + k] || "";
  const promptOf = s => s.prompt || plan.global_prompt || "";
  const detailsOf = s => {
    const value = plan.ref_details?.[`${refOf(s, "ref")}|${refOf(s, "ref2")}`] || "";
    return typeof value === "object" ? Object.values(value).some(Boolean) : !!value;
  };

  // things worth fixing before a run, per shot (only shots that run)
  // ---- memory status: each shot's size (latent tokens = generated frames x resolution) against the biggest shot that
  // already ran on this GPU. Not a GB figure: ComfyUI offloads weights as needed, so only what ran here is a reference.
  const H3G = "H3 (17n+5)";
  const headOf = (s, i) => plan.run !== "auto" && s.chain === "head" && i > 0 && !s.cut && (plan.grid || H3G) === H3G
    ? Math.min(s.chainHistory || 22, 39) : 0;
  const shotTokens = (s, i) => {
    const hd = headOf(s, i), gen = hd ? snapUp(s.len + hd, plan.grid) : s.gen;
    const steps = (plan.grid || H3G) === H3G ? 5 * Math.ceil(gen / 17) - 3 : Math.ceil(gen / 4);
    return Math.max(1, steps) * Math.ceil((size.h || 0) / 16) * Math.ceil((size.w || 0) / 16);
  };
  const memory = computed(() => {
    const list = segs.value.map((s, i) => ({ i, tok: skipWhy(s) ? 0 : shotTokens(s, i), gen: s.gen + headOf(s, i) })).filter(x => x.tok > 0);
    const top = list.reduce((a, b) => (b.tok > a.tok ? b : a), { i: -1, tok: 0, gen: 0 });
    const ratio = vram.max_tokens ? top.tok / vram.max_tokens : 0;
    const level = !top.tok || !size.w ? "" : !vram.max_tokens ? "info" : ratio <= 1.05 ? "ok" : ratio <= 1.3 ? "note" : "warn";
    return { list, top, ratio, level };
  });
  // the final video as images (BFS Shot Join, save_to = images) is held in RAM whole: about 12 bytes per pixel per frame,
  // ~3.5x more with the comparison. Long videos should be written straight to a file instead.
  const joinRam = computed(() => {
    const frames = segs.value.reduce((t, s) => t + (skipWhy(s) && plan.skip_fill === "drop" ? 0 : s.len), 0);
    const gb = frames * (size.w || 0) * (size.h || 0) * 12 / 2 ** 30;
    return { frames, gb, long: vram.ram_free_gb > 0 && gb * 3.5 > 0.6 * vram.ram_free_gb };
  });
  const checks = computed(() => {
    const out = [];
    if (joinRam.value.long && active.value.length)
      out.push({ lvl: "warn", shot: 0, text: `long video: as images the final video needs ~${joinRam.value.gb.toFixed(1)} GB of RAM `
        + `(~${(joinRam.value.gb * 3.5).toFixed(1)} GB with the comparison) and ${vram.ram_free_gb} GB is free. Set BFS Shot Join `
        + `save_to = mp4 file: it writes the video straight to disk, shows it on the node and still gives the result to the next nodes` });
    segs.value.forEach((s, i) => {
      if (skipWhy(s)) return;
      const add = (lvl, text) => out.push({ lvl, shot: i, text });
      const p = promptOf(s);
      if (s.len > maxLen.value) add(plan.mode === "single" ? "note" : "warn", plan.mode === "single"
        ? `one shot of ${s.len} frames (${(s.len / plan.fps).toFixed(1)} s), longer than Max seconds / shot: the model may not hold it`
        : `longer than ${maxLen.value} frames: split it`);
      if ((s.crop || s.inpaint || s.paste) && !hasMask(s)) add("warn", `${modeName(s)} needs a mask (text, points or a mask video; or the planner's mask input)`);
      if (hasMask(s) && maskPrev.value[segKey(s)]?.empty) add("warn", "the mask preview found nothing");
      if (/\{target\}/.test(p) && !s.target) add("warn", "the prompt uses {target} but the shot has no target description");
      if (/\{details2?\}/.test(p) && !detailsOf(s) && !plan.vlm_cfg?.enabled) add("info", "the prompt uses {details} but its references have no description yet");
      if (/\{details2?\}/.test(p) && (plan.vlm_cfg?.describe_preset || "short") !== "short")
        add("warn", "{details} sits inside a sentence of the prompt: use the 'short' describe preset (Prompts & refs), the other presets write a paragraph");
      if (!p) add("info", "no prompt (fine when the node's prompt input is connected)");
      if (!refOf(s, "ref")) add("info", "no reference (fine when the node's ref_image input is connected)");
      if (vram.max_tokens && shotTokens(s, i) > 1.3 * vram.max_tokens)
        add("warn", "much bigger than any shot that ran on this GPU (over 1.3×): it may run out of memory. Lower Megapixels or Max seconds / shot, use head 5, or set Between shots = free memory (Run tab)");
      if (i > 0 && plan.run === "auto" && (s.chain === "head" || s.chain === "history") && !s.cut)
        add("warn", `continuity '${s.chain}' does not work in the auto loop (the previous result does not exist yet): use Run = one run, shot by shot`);
    });
    return out;
  });
  const checksFor = i => checks.value.filter(c => c.shot === i);

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
          chain: keep[i]?.chain || "head", chain_frame: keep[i]?.chain_frame || "first", chain_history: historyLen(keep[i]?.chain_history),
          crop: !!keep[i]?.crop, inpaint: !!keep[i]?.inpaint, paste: !!keep[i]?.paste, paste_text: keep[i]?.paste_text || "", strength: keep[i]?.strength ?? 1, mask: keep[i]?.mask || {}, target: keep[i]?.target || "",
          color_guide: !!keep[i]?.color_guide, color_guide_color: keep[i]?.color_guide_color || "", color_guide_opacity: keep[i]?.color_guide_opacity ?? 0.35,
          color_guide_text2: keep[i]?.color_guide_text2 || "", color_guide_color2: keep[i]?.color_guide_color2 || "",
          color_guide_points2: keep[i]?.color_guide_points2 || [], color_guide_key2: keep[i]?.color_guide_key2 ?? 0 }));
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
    save();
  };
  async function describeTarget(i) {
    const s = segs.value[i]; if (!s) return;
    busy.value = "Selecting the person (SAM 3) and describing them (VLM)…"; busySince.value = Date.now(); error.value = "";
    try {
      const r = await api.fetchApi("/bfs/shotloop/target", { method: "POST", body: JSON.stringify({ plan: { ...plan }, index: i }) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      targetCrop.value = { ...targetCrop.value, [segKey(s)]: j.crop };
      if (j.desc) setMeta(i, "target", j.desc);
      else error.value = "The SAM 3 target cutout was created (shown below), but no VLM is loaded to describe it. Connect Qwen3-VL to the Shot Planner's vlm input and queue the workflow once; then click Describe target again. You can also type the target description yourself.";
    } catch (e) { error.value = String(e.message || e); }
    busy.value = "";
  }
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
  const setMask = (i, k, v) => {
    const m = meta(i); if ((m.mask || {})[k] === v) return;
    m.mask = { ...(m.mask || {}), [k]: v }; save();
    const s = segs.value[i];   // the old preview no longer matches
    if (s && maskPrev.value[segKey(s)]) { const mp = { ...maskPrev.value }; delete mp[segKey(s)]; maskPrev.value = mp; }
  };
  const setMaskCfg = (k, v) => { plan.mask_cfg = { ...plan.mask_cfg, [k]: v }; save(); maskPrev.value = {}; };
  const openPoints = (i, region = 1) => {
    const s = segs.value[i]; if (!s) return;
    const g1 = { key: s.mask.key ?? Math.floor(s.len / 2), points: [...(s.mask.points || [])], text: s.mask.text || "" };
    const g2 = { key: g1.key, points: [...(s.color_guide_points2 || [])], text: s.color_guide_text2 || "" };
    Object.assign(modal, { open: true, idx: i, region: s.color_guide ? region : 1, groups: { 1: g1, 2: g2 },
      key: (region === 2 && s.color_guide ? g2 : g1).key, points: [...((region === 2 && s.color_guide ? g2 : g1).points)],
      text: (region === 2 && s.color_guide ? g2 : g1).text, prev: null, busy: "" });
  };
  const switchPointRegion = region => {
    const old = modal.groups[modal.region] || {};
    modal.groups[modal.region] = { key: modal.key, points: [...modal.points], text: modal.text };
    const next = modal.groups[region] || { key: modal.key, points: [], text: "" };
    modal.region = region; modal.points = [...next.points]; modal.text = next.text; modal.prev = null;
  };
  const commitPointGroups = i => {
    const m = meta(i), g1 = modal.groups[1] || {}, g2 = modal.groups[2] || {};
    m.mask = { ...(m.mask || {}), points: [...(modal.region === 1 ? modal.points : g1.points || [])],
      key: modal.region === 1 ? modal.key : g1.key, text: modal.region === 1 ? modal.text : g1.text };
    if (segs.value[i]?.color_guide) {
      m.color_guide_points2 = [...(modal.region === 2 ? modal.points : g2.points || [])];
      m.color_guide_key2 = modal.region === 2 ? modal.key : (g1.key ?? modal.key);
      m.color_guide_text2 = modal.region === 2 ? modal.text : g2.text || "";
    }
  };
  const savePoints = () => {
    const i = modal.idx; commitPointGroups(i); save();
    if (modal.prev) maskPrev.value = { ...maskPrev.value, [segKey(segs.value[modal.idx])]: modal.prev };
    modal.open = false;
  };
  const targets = (scope, from) => segs.value.map((_, i) => i).filter(i => i !== from && (scope === "all" || i > from));
  // the same selection on other shots (a subject that stays in place): each shot gets the points on its frame at the
  // same relative position as the key frame here
  const pointsTo = (points, key, text, fromIdx, scope = "all") => {
    const src = segs.value[fromIdx]; const rel = src ? key / Math.max(1, src.len - 1) : 0.5;
    [fromIdx, ...targets(scope, fromIdx)].forEach(i => {
      const x = segs.value[i], m = meta(i);
      m.mask = { ...(m.mask || {}), points: points.map(p => ({ ...p })), key: Math.round(rel * Math.max(0, x.len - 1)), text: text ?? (m.mask || {}).text };
    });
    maskPrev.value = {}; save();
  };
  const savePointsAll = () => {
    if (segs.value[modal.idx]?.color_guide) { commitPointGroups(modal.idx); save(); }
    else pointsTo(modal.points, modal.key, modal.text, modal.idx);
    modal.open = false;
  };
  // one selection tracked by SAM 3 through the whole video (or range): every shot without its own selection takes its
  // part of that track, so the subject is found wherever it moves (copied points can fall on the background)
  const saveTrack = () => {
    const s0 = segs.value[modal.idx]; if (!s0 || !modal.points.length) return;
    const own = segs.value.filter((x, i) => i !== modal.idx && (x.mask.text || (x.mask.points || []).length)).length;
    if (own && !window.confirm(`${own} other shot(s) have their own mask text or points, which would win over the track. Clear them so every shot uses the track?`)) {
      // keep them: those shots keep their own selection
    } else {
      segs.value.forEach((_, i) => { const m = meta(i); if (m.mask) { m.mask = { ...m.mask, points: [], text: "" }; } });
    }
    const m = meta(modal.idx); m.mask = { ...(m.mask || {}), points: [], text: "" };
    plan.track = { points: modal.points.map(p => ({ ...p })), frame: s0.start + modal.key, text: modal.text || "",
                   threshold: plan.mask_cfg?.threshold ?? 0.5 };
    maskPrev.value = {}; save(); modal.open = false;
  };
  const clearTrack = () => { plan.track = null; maskPrev.value = {}; save(); };
  const clearMask = i => {
    const m = meta(i); m.mask = {}; m.color_guide_points2 = []; m.color_guide_key2 = 0; m.color_guide_text2 = "";
    m.crop = false; m.inpaint = false; m.paste = false;
    const k = segs.value[i] ? segKey(segs.value[i]) : null;
    if (k) { const mp = { ...maskPrev.value }; delete mp[k]; maskPrev.value = mp; }
    save();
  };
  // copy the chosen settings of shot `from` to the other shots (all of them, or the ones after it)
  const COPY_FIELDS = { ref: "references", prompt: "prompt", mask_text: "mask text", points: "mask points", mask_video: "mask video", crop: "mode",
                        target: "{target}", chain: "continuity", color_guide: "Colored Mark" };
  const copyFrom = (from, fields = copyOpts, scope = copyOpts.scope) => {
    const src = segs.value[from]; if (!src) return 0;
    const to = targets(scope, from);
    to.forEach(i => {
      const m = meta(i), x = segs.value[i];
      if (fields.ref) { m.ref = src.ref; m.ref2 = src.ref2; }
      if (fields.prompt) m.prompt = src.prompt;
      if (fields.mask_text) m.mask = { ...(m.mask || {}), text: src.mask.text || "" };
      if (fields.mask_video) m.mask = { ...(m.mask || {}), video: src.mask.video || "" };
      if (fields.crop && (hasMask(x) || src.mask.text || fields.mask_text)) { m.crop = src.crop; m.inpaint = src.inpaint; m.paste = src.paste; m.paste_text = src.pasteText; m.strength = src.strength; }
      if (fields.target) m.target = src.target;
      if (fields.color_guide) {
        m.color_guide = src.color_guide; m.color_guide_color = src.color_guide_color;
        m.color_guide_opacity = src.color_guide_opacity;
        m.color_guide_text2 = src.color_guide_text2; m.color_guide_color2 = src.color_guide_color2;
        m.color_guide_points2 = [...(src.color_guide_points2 || [])]; m.color_guide_key2 = src.color_guide_key2;
      }
      if (fields.chain && i > 0) { m.chain = src.chain; m.chain_frame = src.chainFrame; m.chain_history = src.chainHistory; }
    });
    if (fields.points && (src.mask.points || []).length) pointsTo(src.mask.points, src.mask.key ?? Math.floor(src.len / 2), undefined, from, scope);
    if (fields.mask_text || fields.points || fields.mask_video) maskPrev.value = {};
    save();
    return to.length;
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
  // reference sets used by the plan (global and per shot): one description each
  const refSets = computed(() => {
    const out = new Map();
    const add = (a, b, where) => { if (!a && !b) return; const k = `${a || ""}|${b || ""}`; if (!out.has(k)) out.set(k, { a, b, where: [] }); out.get(k).where.push(where); };
    add(plan.global_ref, plan.global_ref2, "global");
    segs.value.forEach((x, i) => add(x.ref || plan.global_ref, x.ref2 || plan.global_ref2, `#${i + 1}`));
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
    const s = segs.value[i], g = s && vlmSug.value[segKey(s)]; if (!g) return;
    if (what === "segment" && g.segment) setMask(i, "text", g.segment);
    if (what === "skip") setMeta(i, "force", g.recommend === "skip" ? "skip" : "auto");
  };
  const applyAllSug = () => {
    segs.value.forEach((s, i) => {
      const g = vlmSug.value[segKey(s)]; if (!g) return;
      const m = meta(i);
      if (g.segment && !(m.mask?.text || (m.mask?.points || []).length)) m.mask = { ...(m.mask || {}), text: g.segment };
      if (g.recommend === "skip" && (m.force || "auto") === "auto") m.force = "skip";
    });
    save();
  };
  const setCastOpt = (k, v) => { plan[k] = v; save(); if (k === "cast_split") autoSplit(true); };
  const setFilter = (k, v) => { plan.filters = { ...plan.filters, [k]: v }; save(); if (stats.value.length) analyzeContent(); };
  const setPlan = (k, v, after) => { plan[k] = v; save(); after && after(); };
  // the range changed: split again inside it (the per-shot settings stay with the shot at the same position)
  // start / end in timeline frames (end 0 = the end of the video); the old seconds fields are cleared
  const setRange = (k, v) => { plan[k] = Math.max(0, Math.round(+v || 0)); plan.start_s = 0; plan.end_s = 0; save(); autoSplit(true); };
  const setGlobalMaskVideo = v => { plan.mask_video = v; save(); maskPrev.value = {}; };
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
  // big files go in 8 MB pieces: ComfyUI refuses any single request over --max-upload-size (100 MB by default)
  const CHUNK = 8 * 1024 * 1024, CHUNKED_FROM = 32 * 1024 * 1024;
  async function uploadChunked(file, onPct) {
    const id = `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 12)}`;
    for (let off = 0; off < file.size; off += CHUNK) {
      const piece = file.slice(off, Math.min(file.size, off + CHUNK));
      for (let tries = 0; ; tries++) {          // a dropped piece is sent again (it is written at its offset)
        try {
          await postWithProgress(`/bfs/upload/chunk?id=${id}&offset=${off}`, piece,
                                 f => onPct((off + f * piece.size) / file.size));
          break;
        } catch (e) { if (tries >= 3) throw e; await new Promise(r => setTimeout(r, 1000 * (tries + 1))); }
      }
    }
    const r = await fetch(api.apiURL("/bfs/upload/finish"), { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, name: file.name, size: file.size }) });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
    return j.name;
  }
  async function upload(file, cb) {
    const mb = (file.size / 1e6).toFixed(1);
    busy.value = `Uploading ${file.name} (${mb} MB)…`; upPct.value = 0;
    try {
      let name;
      if (file.size >= CHUNKED_FROM) name = await uploadChunked(file, f => { upPct.value = f; });
      else {
        const fd = new FormData(); fd.append("image", file); fd.append("type", "input"); fd.append("overwrite", "true");
        const j = await postWithProgress("/upload/image", fd, f => { upPct.value = f; });
        name = j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
      }
      await refreshFiles();
      cb(name);
    } catch (e) { error.value = `Upload failed: ${e}`; }
    busy.value = "";
  }
  const pickFile = (accept, cb) => {
    const inp = document.createElement("input"); inp.type = "file"; inp.accept = accept;
    inp.onchange = () => inp.files[0] && upload(inp.files[0], cb); inp.click();
  };
  async function progress(reset = false) {
    if (!plan.video || plan.run === "auto") { prog.done = 0; prog.count = 0; Object.assign(cache, { ok: [], stored: [], mb: 0 }); return; }
    try {
      const r = await api.fetchApi("/bfs/shotloop/progress", { method: "POST", body: JSON.stringify({ plan: { ...plan }, plan_raw: io.getPlan(), reset }) });
      const j = await r.json();
      if (!j.error) { prog.done = j.done; prog.count = j.count || active.value.length; Object.assign(cache, j); }
    } catch { /* ignore */ }
  }
  // ---- stored shots: redo some, or combine what is stored into an MP4 (low memory, on the server)
  async function redo(shots, cascade = true) {
    try {
      const r = await api.fetchApi("/bfs/shotloop/redo", { method: "POST", body: JSON.stringify({ plan: { ...plan }, shots, cascade }) });
      const j = await r.json(); if (j.error) error.value = j.error; else { Object.assign(cache, j); prog.done = j.done; }
    } catch (e) { error.value = `${e}`; }
  }
  async function combine() {
    busy.value = "Combining the stored shots into an MP4";
    try {
      const r = await api.fetchApi("/bfs/shotloop/combine", { method: "POST", body: JSON.stringify({ plan: { ...plan },
        crossfade: plan.combine_crossfade ?? 4, crf: plan.combine_crf ?? 17 }) });
      const j = await r.json();
      if (j.error) error.value = j.error;
      else cache.combined = { ...j, url: api.apiURL(`/view?filename=${encodeURIComponent(j.filename)}&subfolder=${encodeURIComponent(j.subfolder)}&type=output`) };
    } catch (e) { error.value = `${e}`; }
    busy.value = "";
  }
  // a shot of the panel (any, skipped ones too) -> its cache state: "ok" stored and up to date, "run" runs next time
  const cacheOf = s => { const k = active.value.indexOf(s); return k < 0 || !cache.ok.length ? "" : cache.ok[k] ? "ok" : "run"; };

  // ---- editing
  const setVideo = v => {
    plan.video = v; plan.bounds = []; plan.segs = [];
    if (v) {
      // A new source starts as one manageable clip. The user can widen the range or ask for an automatic split later;
      // do not surprise them with a wall of detected shots on first load.
      plan.mode = "single";
      plan.start_f = 0;
      plan.end_f = Math.round(15 * plan.fps);
      plan.start_s = 0; plan.end_s = 0; plan.max_total_s = 0;
      tab.value = "video";
    }
    save(); analyze();
  };
  const splitAt = f => {
    const { lo, hi } = range.value; f = Math.round(f);
    if (f <= lo || f >= hi || plan.bounds.includes(f)) return;
    const i = segs.value.findIndex(s => f > s.start && f < s.end);
    // a split you make by hand always shows: "Max shots" (e.g. 1 in a one-shot workflow) only limits detected shots
    if (plan.max_parts > 0 && allSegCount.value >= plan.max_parts) plan.max_parts = allSegCount.value + 1;
    plan.bounds = [...plan.bounds, f].sort((a, c) => a - c);
    plan.segs.splice(i + 1, 0, { ...(plan.segs[i] || {}) });
    sel.value = i + 1; save();
  };
  const mergeNext = i => {
    const s = segs.value[i]; if (!s || i >= segs.value.length - 1) return;
    plan.bounds = plan.bounds.filter(b => b !== s.end); plan.segs.splice(i + 1, 1); save();
  };
  // Delete / Backspace on a selected shot: remove the cut at its start (merge into the previous shot;
  // the first shot merges with the next one)
  const removeCut = i => {
    const S = segs.value, s = S[i]; if (!s || S.length < 2) return;
    if (i === 0) { mergeNext(0); sel.value = 0; return; }
    plan.bounds = plan.bounds.filter(b => b !== s.start); plan.segs.splice(i, 1);
    sel.value = i - 1; save();
  };
  const select = i => { if (i >= 0 && i < segs.value.length) sel.value = i; };
  const onKey = e => {
    const t = e.target, tag = (t?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || t?.isContentEditable) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {     // previous / next shot
      e.preventDefault(); e.stopPropagation(); select(sel.value + (e.key === "ArrowLeft" ? -1 : 1)); return;
    }
    if (e.key !== "Delete" && e.key !== "Backspace") return;
    e.preventDefault(); e.stopPropagation();   // keep ComfyUI from deleting the node
    removeCut(sel.value);
  };
  const setMeta = (i, k, v) => { meta(i)[k] = v; save(); };
  const setMode = (i, mode) => { const m = MODES.find(x => x[0] === mode); Object.assign(meta(i), { crop: m[2], inpaint: m[3], paste: m[4] }); save(); };
  const drag = (k, ev) => {
    ev.preventDefault(); ev.stopPropagation();
    const box = tlEl.value.getBoundingClientRect();
    const move = e => {
      const x = e.clientX - box.left + tlEl.value.scrollLeft;
      const f = Math.round(x / pxPerFrame.value);
      const lo = (plan.bounds[k - 1] ?? range.value.lo) + 1, hi = (plan.bounds[k + 1] ?? range.value.hi) - 1;
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
  const onProg = e => { prog.done = e.detail.done; prog.count = e.detail.count; refreshVram(); };
  const onStatus = e => { if (busy.value || modal.busy) Object.assign(status, e.detail || {}); };
  let clock = 0;
  const onVlm = e => { if (e.detail?.video === plan.video) mergeSug(e.detail.segs); };
  const onNext = e => { onProg(e); if (plan.run === "queue" && plan.auto_continue) setTimeout(() => app.queuePrompt(0, 1), 300); };
  onMounted(() => {
    load(); refreshFiles(); analyze(); progress(); refreshVram(); raf = requestAnimationFrame(loop);
    if (!plan.video) tab.value = "video";
    api.addEventListener("bfs-shotloop-progress", onProg); api.addEventListener("bfs-shotloop-next", onNext);
    api.addEventListener("bfs-shotloop-vlm", onVlm);
    api.addEventListener("bfs-shotloop-status", onStatus);
    clock = setInterval(() => { if (busy.value || modal.busy) now.value = Date.now(); }, 500);
  });
  onBeforeUnmount(() => { cancelAnimationFrame(raf); api.removeEventListener("bfs-shotloop-progress", onProg); api.removeEventListener("bfs-shotloop-next", onNext); api.removeEventListener("bfs-shotloop-vlm", onVlm); api.removeEventListener("bfs-shotloop-status", onStatus); clearInterval(clock); });
  io.expose({ reload: () => { load(); analyze(); progress(); } });

  // 🧠 the memory status at the top: the plan's biggest shot against the biggest one that ran on this GPU
  const kTok = n => (n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : `${n}`);
  function memTip(m) {
    const gpu = vram.gpu ? `${vram.gpu} · ${vram.total_gb} GB (${vram.free_gb} GB free now)` : "no CUDA GPU seen by the server";
    const top = m.top.i >= 0 ? `Biggest shot in this plan: #${m.top.i + 1}, ${m.top.gen} frames at ${size.w}×${size.h} = ${kTok(m.top.tok)} tokens` : "";
    const ref = vram.max_tokens
      ? `Biggest shot that ran on this GPU: ${kTok(vram.max_tokens)} tokens, peak ${vram.max_peak_gb} GB (${vram.runs} run(s) recorded)`
      : "No shot has run on this GPU yet: after the first one, this status compares every shot with what already ran here.";
    const say = { ok: "✓ Every shot is within what already ran on this GPU.",
      note: "A bit bigger than anything that ran here: it will probably fit (ComfyUI offloads weights), maybe slower.",
      warn: "⚠ Much bigger than anything that ran here: it may run out of memory. Lower Megapixels or Max seconds / shot, use head 5, or set Between shots = free memory (Run tab).",
      info: "" }[m.level];
    return [gpu, top, ref, say, "Size = generated frames (with the continuity head) × resolution; skipped shots are not counted."].filter(Boolean).join("\n");
  }
  function memPill() {
    const m = memory.value;
    if (!m.level) return null;
    const text = { ok: "🧠 memory ✓", note: `🧠 ${m.ratio.toFixed(2)}× biggest run`, warn: `🧠 ⚠ ${m.ratio.toFixed(1)}× biggest run · may OOM`,
      info: "🧠 no memory data yet" }[m.level];
    return h("span", { class: ["pill", { ok: "ok", warn: "warn" }[m.level] || ""], style: "cursor:pointer", title: memTip(m),
      onClick: () => { tab.value = "run"; refreshVram(); } }, text);
  }

  // ---- references
  // the last 10 references picked, remembered across workflows (per browser)
  const RECENT_KEY = "bfs.shotloop.recentRefs";
  const isColorValue = name => typeof name === "string" && /^#[0-9a-f]{6}$/i.test(name);
  const savedRecent = store.get(RECENT_KEY, []);
  const recent = ref((Array.isArray(savedRecent) ? savedRecent : []).filter(name => typeof name === "string" && !isColorValue(name)));
  if (recent.value.length !== (Array.isArray(savedRecent) ? savedRecent.length : 0)) store.set(RECENT_KEY, recent.value);
  const remember = name => {
    if (!name || typeof name !== "string" || isColorValue(name)) return;
    recent.value = [name, ...recent.value.filter(x => x !== name)].slice(0, 10);
    store.set(RECENT_KEY, recent.value);
  };
  // target: shot index or "global"; field: ref | ref2 (global_ref | global_ref2 for "global")
  const setRef = (target, field, name) => {
    if (isColorValue(name)) return; // colors belong to Colored Mark settings, never to image references
    if (target === "global") plan[field] = name; else meta(target)[field] = name;
    remember(name); save();
  };
  const uploadRef = (target, field) => pickFile("image/*", name => setRef(target, field, name));
  const clearRecent = () => { recent.value = []; store.set(RECENT_KEY, []); };
  // everything: the recent list and every reference this plan uses (global and per shot)
  const clearAllRefs = () => {
    if (!confirm("Remove every reference picture from this plan (global, every shot) and forget the recent list?")) return;
    clearRecent();
    plan.global_ref = ""; plan.global_ref2 = "";
    plan.segs.forEach(m => { if (m) { m.ref = ""; m.ref2 = ""; } });
    save();
  };
  const usedRefs = computed(() => {
    const set = new Set([...recent.value, plan.global_ref, plan.global_ref2, ...plan.segs.flatMap(m => [m?.ref, m?.ref2])].filter(Boolean));
    return [...set].slice(0, 10);
  });

  // everything the views need
  const c = {
    api, plan, files, an, cuts, detectorUsed, size, busy, error, status, busySince, now, upPct, sel, zoom, hover, prog, tlEl,
    stats, people, maskPrev, showMasks, vlmSug, modal, segPeople, targetCrop, tab, copyOpts, COPY_FIELDS, vid, play, recent, clearRecent, clearAllRefs,
    n, range, setRange, maxLen, segs, allSegCount, active, pxPerFrame, tlWidth, filtersOn, refSets, usedRefs, checks, tour, memory, vram, refreshVram, cache, cacheOf, redo, combine,
    MODES, modeOf, modeName, viewUrl, save, meta, setMeta, setMode, setPlan, whoIn, castOf, included, personOf, statFor, skipWhy, refOf, promptOf, checksFor,
    refreshFiles, analyze, autoSplit, analyzeContent, findPeople, setCast, describeTarget, previewMask, setMask, setMaskCfg,
    openPoints, switchPointRegion, savePoints, savePointsAll, saveTrack, clearTrack, pointsTo, clearMask, copyFrom, analyseVLM, describeRefs, setDetail, setVlmCfg,
    applySug, applyAllSug, setCastOpt, setGlobalMaskVideo, setFilter, pickFile, progress, setVideo, splitAt, mergeNext, removeCut, select, drag,
    frameAt, thumbFor, playShot, playAll, stop, seek, setRef, uploadRef,
  };

  return () => {
    const S = segs.value, N = n.value, fps = plan.fps;
    const warns = checks.value.filter(x => x.lvl === "warn").length;
    const guideUrl = new URL("./docs/BFSShotPlanner.md", import.meta.url).href;
    const header = h("div", { class: "hdr" }, [
      h("span", { class: "ttl" }, "🎬 Shot Planner"),
      an.value ? pill(`${active.value.length}/${S.length} shots · ${((range.value.hi - range.value.lo) / fps).toFixed(1)}s`
        + (range.value.lo > 0 || range.value.hi < N ? " (range)" : ""), "ok") : null,
      warns ? h("span", { class: "pill warn", style: "cursor:pointer", title: "open the checks (Run tab)", onClick: () => { tab.value = "run"; } }, `⚠ ${warns}`) : null,
      h("span", { class: "grow" }),
      h("label", { class: "langpick", title: t("Language") }, [
        h("span", { class: "hint" }, "🌐"),
        h("select", { value: uiLanguage.value, "aria-label": t("Language"), onChange: e => setUiLanguage(e.target.value) }, [
          rawH("option", { value: "en" }, "English"), rawH("option", { value: "pt" }, "Português"),
          rawH("option", { value: "es" }, "Español"), rawH("option", { value: "zh" }, "中文"),
        ]),
      ]),
      memPill(),
      pill({ queue: "queue loop", recursive: "shot by shot" }[plan.run] || "auto loop"),
      h("button", { title: t("Shot Planner tour"), onClick: () => { tour.value = 1; tab.value = "video"; } }, t("🧭 Tour")),
      h("button", { title: "Open the full guide (every setting explained, with examples)", onClick: () => showGuide(guideUrl) }, "📖 Guide"),
    ]);
    const counts = { shots: S.length || null, people: people.value.length || null };
    const tabs = h("div", { class: "tabs" }, TABS.map(([k, label]) => h("button", {
      class: ["tab", tab.value === k && "on"], disabled: k !== "video" && !an.value, onClick: () => { tab.value = k; } }, [
      label, counts[k] ? h("span", { class: "n" }, counts[k]) : null, k === "run" && warns ? h("span", { class: "dot", title: `${warns} warning(s)` }) : null,
    ])));
    const views = { video: videoTab, shots: shotsTab, people: peopleTab, prompts: promptsTab, run: runTab };
    const cur = an.value || tab.value === "video" ? tab.value : "video";
    return h("div", { class: "bsl root", ref: rootEl, tabindex: 0, onKeydown: onKey }, [
      header, tabs,
      busy.value ? busyBar(c, busy.value) : null,
      error.value ? h("div", { class: "err" }, [h("span", { class: "grow" }, error.value), h("button", { class: "ghost", onClick: () => { error.value = ""; } }, "✕")]) : null,
      an.value ? timelineCard(c) : null,
      views[cur](c),
      pointsModal(c),
      tourModal({ tour, tab }),
    ]);
  };
}

app.registerExtension({
  name: "BFSNodes.ShotPlanner",
  async nodeCreated(node) {
    if (node.comfyClass !== "BFSShotPlanner") return;
    styles();
    const planWidget = node.widgets?.find(w => w.name === "plan");
    if (planWidget) {   // the JSON plan stays a widget (saved with the workflow) but is never shown, in both node renderers
      planWidget.type = "hidden"; planWidget.hidden = true; planWidget.computeSize = () => [0, -4];
      planWidget.options = { ...(planWidget.options || {}), hidden: true };
    }
    const host = document.createElement("div");
    // the panel is absolutely positioned inside the host and scrolls by itself, so switching tabs never resizes the
    // node (Vue nodes mode sizes widgets by their content); min-width keeps it from collapsing to its content width
    host.style.cssText = "position:relative;width:100%;height:100%;min-width:900px;min-height:640px";
    node.addDOMWidget("bfs_shot_planner", "div", host, { serialize: false, hideOnZoom: false });
    wheelToCanvas(host);   // the wheel zooms the canvas over the panel too, except on something that scrolls
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
      if (this.size[0] < MIN_W) this.setSize([MIN_W, Math.max(this.size[1], MIN_H)]);   // older workflows: too narrow
      return r;
    };
    node.size = [Math.max(node.size[0], DEFAULT_W), Math.max(node.size[1], MIN_H)];
  },
});
