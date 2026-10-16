/**
 * BFS Video Compare: one player for two videos.
 *
 * Wipe (drag the divider), overlay (drag to fade) or side by side, with any of guide / target / difference on top
 * or below. The strip under the scrubber is the difference per frame, measured by the node on the real frames:
 * click it to jump to a frame. The node sends three temporary MP4s and that list when it runs.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SOURCES = ["guide", "target", "difference"];
const MODES = [["wipe", "wipe ↔"], ["wipe-v", "wipe ↕"], ["overlay", "overlay"], ["side", "side by side"]];

function styles() {
  if (document.getElementById("bfs-vcmp-css")) return;
  const el = document.createElement("style");
  el.id = "bfs-vcmp-css";
  el.textContent = `
.bfsvc{display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;padding:6px;
  font:12px/1.4 var(--font-family,system-ui,sans-serif);color:var(--descrip-text,#c9c9cf);
  background:var(--comfy-input-bg,#1b1b1f);border-radius:8px}
.bfsvc .stage{position:relative;flex:1;min-height:160px;background:#0e0e11;border-radius:6px;overflow:hidden;
  cursor:ew-resize;touch-action:none;user-select:none}
.bfsvc .stage.v{cursor:ns-resize}.bfsvc .stage.o{cursor:col-resize}.bfsvc .stage.s{cursor:default;display:grid;
  grid-template-columns:1fr 1fr;gap:4px}
.bfsvc video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none}
.bfsvc .stage.s video{position:relative}
.bfsvc .line{position:absolute;background:#fff;box-shadow:0 0 0 1px #0008;pointer-events:none}
.bfsvc .tag{position:absolute;top:6px;padding:1px 7px;border-radius:4px;background:#000a;color:#fff;font-size:11px;
  pointer-events:none}
.bfsvc .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.bfsvc select,.bfsvc button{background:#131317;border:1px solid #34343d;color:#e6e6ea;border-radius:4px;
  padding:2px 6px;font:inherit;cursor:pointer}
.bfsvc button:hover,.bfsvc select:hover{border-color:#3a86ff}
.bfsvc input[type=range]{flex:1;accent-color:#3a86ff;min-width:80px}
.bfsvc canvas{width:100%;height:26px;cursor:pointer;border-radius:3px;background:#131317;display:block}
.bfsvc .info{font-family:ui-monospace,monospace;font-size:11px;white-space:nowrap}
.bfsvc .empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#8b8b93;
  text-align:center;padding:12px}
`;
  document.head.appendChild(el);
}

const url = f => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder || "")}`
  + `&type=${f.type || "temp"}&t=${Date.now()}`);

function el(tag, attrs = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") e.style.cssText = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) e.append(c);
  return e;
}

function buildPlayer(host) {
  const st = { data: null, top: "target", bottom: "guide", mode: "wipe", pos: 0.5, playing: false, muted: true, frame: 0 };
  const vTop = el("video", { playsinline: "", preload: "auto", loop: "" });
  const vBot = el("video", { playsinline: "", preload: "auto", loop: "" });
  vTop.muted = vBot.muted = true;
  const line = el("div", { class: "line" });
  const tagTop = el("div", { class: "tag" }), tagBot = el("div", { class: "tag" });
  const empty = el("div", { class: "empty" }, "Run the node to load the guide and the target.");
  const stage = el("div", { class: "stage", tabindex: "0" }, [vBot, vTop, line, tagTop, tagBot, empty]);
  const sel = (opts, val, on) => {
    const s = el("select", {}, opts.map(([v, l]) => { const o = el("option", { value: v }, l); if (v === val) o.selected = true; return o; }));
    s.addEventListener("change", () => on(s.value));
    return s;
  };
  const srcOpts = () => SOURCES.map(s => [s, s === "guide" ? (st.data?.labels?.[0] || "guide") : s === "target" ? (st.data?.labels?.[1] || "target") : "difference"]);
  let selTop, selBot;
  const selMode = sel(MODES, st.mode, v => { st.mode = v; layout(); });
  const refreshSelects = () => {
    const t = sel(srcOpts(), st.top, v => { st.top = v; load(); }), b = sel(srcOpts(), st.bottom, v => { st.bottom = v; load(); });
    selTop ? selTop.replaceWith(t) : null; selBot ? selBot.replaceWith(b) : null;
    selTop = t; selBot = b;
  };
  refreshSelects();
  const btnSwap = el("button", { title: "Swap top and bottom", onclick: () => { [st.top, st.bottom] = [st.bottom, st.top]; refreshSelects(); load(); } }, "⇅ swap");
  const btnPlay = el("button", { title: "Play / pause (space)", onclick: () => toggle() }, "▶");
  const btnPrev = el("button", { title: "Previous frame (←)", onclick: () => step(-1) }, "◀|");
  const btnNext = el("button", { title: "Next frame (→)", onclick: () => step(1) }, "|▶");
  const btnMute = el("button", { title: "Sound (from the target)", onclick: () => { st.muted = !st.muted; sound(); } }, "🔇");
  const btnFull = el("button", { title: "Full screen", onclick: () => (document.fullscreenElement ? document.exitFullscreen() : host.requestFullscreen?.())?.catch?.(() => {}) }, "⛶");
  const scrub = el("input", { type: "range", min: "0", max: "0", value: "0", step: "1" });
  scrub.addEventListener("input", () => seek(+scrub.value));
  const info = el("span", { class: "info" }, "");
  const strip = el("canvas", { title: "Difference per frame: click to jump" });
  strip.addEventListener("pointerdown", e => {
    const r = strip.getBoundingClientRect(), n = st.data?.frames || 0;
    if (n) seek(Math.min(n - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * n))));
  });
  host.append(el("div", { class: "bfsvc" }, [
    el("div", { class: "row" }, [el("span", {}, "top"), selTop, btnSwap, el("span", {}, "bottom"), selBot, selMode, btnFull]),
    stage,
    el("div", { class: "row" }, [btnPrev, btnPlay, btnNext, scrub, btnMute, info]),
    strip,
  ]));

  const fps = () => st.data?.fps || 24;
  const frameOf = t => Math.min((st.data?.frames || 1) - 1, Math.max(0, Math.floor(t * fps() + 1e-3)));
  const timeOf = f => (f + 0.5) / fps();
  const videos = () => [vBot, vTop];

  function sound() {
    btnMute.textContent = st.muted ? "🔇" : "🔊";
    for (const v of [vTop, vBot]) v.muted = true;
    if (!st.muted) { const v = st.bottom === "target" ? vBot : st.top === "target" ? vTop : null; if (v) v.muted = false; }
  }

  function layout() {
    const m = st.mode, p = st.pos;
    stage.className = "stage" + (m === "wipe-v" ? " v" : m === "overlay" ? " o" : m === "side" ? " s" : "");
    vTop.style.clipPath = m === "wipe" ? `inset(0 ${(1 - p) * 100}% 0 0)` : m === "wipe-v" ? `inset(0 0 ${(1 - p) * 100}% 0)` : "none";
    vTop.style.opacity = m === "overlay" ? String(p) : "1";
    line.style.display = m === "wipe" || m === "wipe-v" ? "block" : "none";
    if (m === "wipe") line.style.cssText = `display:block;left:calc(${p * 100}% - 1px);top:0;bottom:0;width:2px`;
    if (m === "wipe-v") line.style.cssText = `display:block;top:calc(${p * 100}% - 1px);left:0;right:0;height:2px`;
    stage.prepend(vTop); stage.prepend(vBot);   // under the divider and tags; side by side: bottom left, top right
    const name = s => s === "guide" ? (st.data?.labels?.[0] || "guide") : s === "target" ? (st.data?.labels?.[1] || "target") : "difference";
    tagTop.textContent = name(st.top) + (m === "overlay" ? ` ${Math.round(p * 100)}%` : "");
    tagBot.textContent = name(st.bottom);
    tagTop.style.cssText = m === "side" ? "right:6px" : "left:6px";
    tagBot.style.cssText = m === "side" ? "left:6px" : m === "wipe-v" ? "left:6px;top:auto;bottom:6px" : "right:6px";
    if (m === "overlay") tagBot.style.cssText = "right:6px";
  }

  function drag(e) {
    if (st.mode === "side" || !st.data) return;
    const r = stage.getBoundingClientRect();
    st.pos = Math.min(1, Math.max(0, st.mode === "wipe-v" ? (e.clientY - r.top) / r.height : (e.clientX - r.left) / r.width));
    layout();
  }
  stage.addEventListener("pointerdown", e => { stage.setPointerCapture(e.pointerId); drag(e); });
  stage.addEventListener("pointermove", e => { if (stage.hasPointerCapture(e.pointerId)) drag(e); });
  stage.addEventListener("keydown", e => {
    if (e.key === " ") { toggle(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
  });

  function drawStrip() {
    const d = st.data, ctx = strip.getContext("2d");
    const w = strip.width = Math.max(50, strip.clientWidth * devicePixelRatio), h = strip.height = 26 * devicePixelRatio;
    ctx.clearRect(0, 0, w, h);
    if (!d?.mae?.length) return;
    // from the least to the most different frame, so the changes along the video stand out
    const mx = Math.max(...d.mae), mn = Math.min(...d.mae), span = Math.max(mx - mn, 1e-6), n = d.mae.length;
    ctx.fillStyle = "#e0573a";
    d.mae.forEach((v, i) => { const bh = 2 + (v - mn) / span * (h - 4); ctx.fillRect(i / n * w, h - bh, Math.max(1, w / n), bh); });
    ctx.fillStyle = "#3a86ff";
    ctx.fillRect(st.frame / n * w, 0, Math.max(2, w / n), h);
  }

  function show(f) {
    st.frame = f;
    scrub.value = String(f);
    const d = st.data;
    const t = (f / fps());
    info.textContent = d ? `frame ${f}/${d.frames - 1}  ${t.toFixed(2)}s  diff ${(d.mae?.[f] ?? 0).toFixed(1)}/255` : "";
    drawStrip();
  }

  function seek(f) {
    if (!st.data) return;
    if (st.playing) toggle();
    for (const v of videos()) v.currentTime = timeOf(f);
    show(f);
  }
  const step = k => seek(Math.min(st.data ? st.data.frames - 1 : 0, Math.max(0, st.frame + k)));

  function toggle() {
    if (!st.data) return;
    st.playing = !st.playing;
    btnPlay.textContent = st.playing ? "⏸" : "▶";
    if (st.playing) { vTop.currentTime = vBot.currentTime; for (const v of videos()) v.play().catch(() => {}); tick(); }
    else for (const v of videos()) v.pause();
  }

  function tick() {   // the bottom video is the clock; the top one follows it within half a frame
    if (!st.playing) return;
    if (Math.abs(vTop.currentTime - vBot.currentTime) > 0.5 / fps()) vTop.currentTime = vBot.currentTime;
    show(frameOf(vBot.currentTime));
    requestAnimationFrame(tick);
  }

  function load() {
    const d = st.data;
    if (!d) return;
    const f = st.frame, was = st.playing;
    if (was) toggle();
    let pending = 2;
    const ready = () => { if (--pending === 0) { for (const v of videos()) v.currentTime = timeOf(f); show(f); sound(); if (was) toggle(); } };
    for (const [v, s] of [[vTop, st.top], [vBot, st.bottom]]) {
      v.addEventListener("loadeddata", ready, { once: true });
      v.src = url(d[s]);
      v.load();
    }
    layout();
  }

  return {
    setData(d) {
      st.data = d;
      empty.style.display = "none";
      scrub.max = String(Math.max(0, d.frames - 1));
      refreshSelects();
      st.frame = Math.min(st.frame, d.frames - 1);
      load();
    },
    layout, drawStrip,
  };
}

app.registerExtension({
  name: "BFSNodes.VideoCompare",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "BFSVideoCompare") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      styles();
      const host = el("div", { style: "width:100%;height:100%;min-height:320px" });
      this._bfsCompare = buildPlayer(host);
      this.addDOMWidget("bfs_video_compare", "div", host, { serialize: false, hideOnZoom: false, getMinHeight: () => 320 });
      this.setSize([Math.max(this.size[0], 560), Math.max(this.size[1], 620)]);
      new ResizeObserver(() => this._bfsCompare?.drawStrip()).observe(host);
      return r;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      const d = message?.bfs_compare?.[0];
      if (d) this._bfsCompare?.setData(d);
    };
  },
});
