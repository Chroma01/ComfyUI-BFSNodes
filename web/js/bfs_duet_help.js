// "❓ How to use" for the H3 duet nodes: a window with drawn examples of every setting.
import { app } from "../../scripts/app.js";
import { showGuide } from "./bfs_md_view.js";

const NODES = ["BFSH3Duet", "BFSShotH3Duet", "BFSH3SidePanel", "BFSH3DuetConditioning"];

// ---- small SVG diagrams -------------------------------------------------------------------
const C = { video: "#3b6fd8", panel: "#d88a2b", grey: "#7a7a7a", ink: "#e8e8ee", dim: "#9a9aa8", bg: "#1b1b21" };

function canvas({ position = "left", size = 1.0, fit = "contain", gap = 0, w = 150, h = 96, label = "" }) {
  // video area keeps 16:9-ish; the strip sits on one side
  const horiz = position === "left" || position === "right";
  const vw = horiz ? w / (1 + size + gap * 0.08) : w, vh = horiz ? h : h / (1 + size + gap * 0.08);
  const sw = horiz ? vw * size : vw, sh = horiz ? vh : vh * size, g = gap * (horiz ? vw : vh) * 0.08;
  const vx = position === "left" ? sw + g : 0, vy = position === "top" ? sh + g : 0;
  const px = position === "right" ? vw + g : 0, py = position === "bottom" ? vh + g : 0;
  // panel content (the source clip, 16:9) placed in the strip
  const ar = vw / vh;
  let cw = sw, ch = sh;
  if (fit === "contain") { if (sw / sh > ar) cw = sh * ar; else ch = sw / ar; }
  const cx = px + (sw - cw) / 2, cy = py + (sh - ch) / 2;
  const W = horiz ? vw + sw + g : vw, H = horiz ? vh : vh + sh + g;
  const clip = fit === "cover" && Math.abs(sw / sh - ar) > 1e-3;
  // cover: the clip is scaled to fill the strip; what sticks out is cropped (dashed)
  let ow = sw, oh = sh;
  if (clip) { if (sw / sh > ar) oh = sw / ar; else ow = sh * ar; }
  const ox = px + (sw - ow) / 2, oy = py + (sh - oh) / 2;
  const pad = clip ? Math.max(0, (ow - sw) / 2, (oh - sh) / 2) + 2 : 2;
  return `<svg viewBox="${-pad} ${-pad} ${W + 2 * pad} ${H + 16 + 2 * pad}" width="${W + 2 * pad}" height="${H + 16 + 2 * pad}">
    <rect x="${px}" y="${py}" width="${sw}" height="${sh}" fill="${C.grey}" opacity=".45"/>
    ${clip ? `<rect x="${ox}" y="${oy}" width="${ow}" height="${oh}" fill="none" stroke="${C.panel}" stroke-dasharray="3 2"/>
             <rect x="${px}" y="${py}" width="${sw}" height="${sh}" fill="${C.panel}"/>
             <text x="${ox + 2}" y="${oy + 9}" fill="${C.dim}" font-size="8">cropped</text>` :
             `<rect x="${cx}" y="${cy}" width="${cw}" height="${ch}" fill="${C.panel}"/>`}
    <text x="${px + sw / 2}" y="${py + sh / 2 + 4}" fill="#fff" font-size="10" text-anchor="middle">panel</text>
    ${g ? `<rect x="${horiz ? (position === "left" ? sw : vw) : 0}" y="${horiz ? 0 : (position === "top" ? sh : vh)}" width="${horiz ? g : vw}" height="${horiz ? vh : g}" fill="${C.grey}"/>` : ""}
    <rect x="${vx}" y="${vy}" width="${vw}" height="${vh}" fill="${C.video}"/>
    <text x="${vx + vw / 2}" y="${vy + vh / 2 + 4}" fill="#fff" font-size="10" text-anchor="middle">video (output)</text>
    <text x="${W / 2}" y="${H + 13}" fill="${C.dim}" font-size="10" text-anchor="middle">${label}</text></svg>`;
}

function rope(mode, gap) {
  const n = 7, step = 18, y = 22;
  let s = "", x = 6;
  for (let i = 0; i < n; i++) { s += `<circle cx="${x}" cy="${y}" r="5" fill="${C.panel}"/>`; x += step; }
  if (mode === "shifted") x += gap * step * 0.5;
  const gx0 = x - step;
  for (let i = 0; i < n; i++) { s += `<circle cx="${x}" cy="${y}" r="5" fill="${C.video}"/>`; x += step; }
  const label = mode === "canvas" ? "canvas: one wide grid, panel and video side by side"
    : `shifted: video keeps its own positions, panel further out${gap ? ` (gap ${gap})` : ""}`;
  const W = Math.max(x + 4, 340);
  return `<svg viewBox="0 0 ${W} 46" width="${W}" height="46">${s}
    ${mode === "shifted" && gap ? `<line x1="${gx0 + 8}" y1="${y}" x2="${gx0 + gap * step * 0.5 + step - 8}" y2="${y}" stroke="${C.dim}" stroke-dasharray="3 3"/>` : ""}
    <text x="4" y="42" fill="${C.dim}" font-size="10">${label}</text></svg>`;
}

function hold(kind) {
  const n = 7; let s = "";
  for (let i = 0; i < n; i++) {
    const held = kind === "all frames" || i === 0;
    s += `<rect x="${4 + i * 20}" y="6" width="16" height="16" fill="${held ? C.panel : C.grey}" opacity="${held ? 1 : .45}"/>`;
  }
  return `<svg viewBox="0 0 260 40" width="260" height="40">${s}<text x="4" y="36" fill="${C.dim}" font-size="10">${kind}: panel held on the orange frames</text></svg>`;
}

const row = (...cells) => `<div class="bdh-row">${cells.map(c => `<div class="bdh-cell">${c}</div>`).join("")}</div>`;

// ---- content ------------------------------------------------------------------------------
function html(type) {
  const shot = type === "BFSShotH3Duet";
  return `
  <h2>H3 Duet: how to use</h2>
  <p>The <b>panel</b> (a clip or a picture) is pinned next to the video in one wide canvas. MiniMax H3 keeps the panel
  exactly and generates the video beside it, <b>in sync</b>: the new video copies the panel's motion, camera, timing and
  cuts. The panel is cut off before decoding, so only the video comes out. No LoRA is needed.</p>
  ${row(canvas({ label: "the canvas the model sees" }), `<svg width="40" height="96"><text x="8" y="52" fill="${C.ink}" font-size="22">→</text></svg>`,
        canvas({ w: 75, size: 0.0001, label: "what you get" }).replace(">panel<", "><"))}

  <h3>What goes in</h3>
  <ul>
    <li><b>panel</b>${shot ? " (the shot's own clip, set by the mode)" : ""}: usually the <b>source clip</b> you want to restyle or recast. A single picture also works but copies less.</li>
    <li><b>ref_images</b> → <code>&lt;Picture 1&gt;</code>, <code>&lt;Picture 2&gt;</code>…: the new person, outfit or place.</li>
    <li><b>guide</b> (optional): an aligned latent guide, for LoRAs trained on it. Usually leave it empty with a panel.</li>
  </ul>

  <h3>Prompt, task and instruction</h3>
  <p><b>prompt</b> always wins. Leave it empty and the node writes a <b>draft</b> from <b>task</b> + <b>instruction</b>
  (see the <code>prompt</code> output). The draft cannot see the video, so a prompt written for the clip is better.</p>
  <table>
    <tr><th>task</th><th>instruction: a few words of what is SEEN</th></tr>
    <tr><td>character swap</td><td><i>(empty, or the look)</i> + the person's pictures as references</td></tr>
    <tr><td>style</td><td><code>a 1990s anime cel style</code></td></tr>
    <tr><td>setting</td><td><code>a sunny beach at sunset, waves behind her</code></td></tr>
    <tr><td>appearance</td><td><code>an elderly woman with short grey hair</code></td></tr>
    <tr><td>lighting / weather</td><td><code>night, lit by pink and blue neon signs</code></td></tr>
  </table>
  <p>Rules that matter (render-tested): the panel has <b>no tag</b> (it is not &lt;Picture n&gt; or &lt;Video n&gt;): call it
  "the kept footage" by its side, or write <code>{layout}</code> to insert the sentence. <b>Never describe the panel's person,
  clothes or room</b> (what you describe gets drawn; what you leave out is copied from the panel). Restate the new identity in
  every shot ("her face from &lt;Picture 1&gt;" + 2–3 face/hair/outfit words). Give exact times for cuts
  ("[Shot 2] At 00:03.708, both halves cut together to…"). No negations ("no hat" draws a hat). The full example is in the
  prompt tooltip.</p>

  <h3>Layout</h3>
  <p><b>position</b>: the side of the panel.</p>
  ${row(canvas({ position: "left", label: "left" }), canvas({ position: "right", label: "right" }),
        canvas({ position: "top", w: 96, h: 120, label: "top" }), canvas({ position: "bottom", w: 96, h: 120, label: "bottom" }))}
  <p><b>size</b>: the panel against the video (1.0 = two equal halves). Smaller is cheaper; keep faces readable.</p>
  ${row(canvas({ size: 1.0, label: "size 1.0" }), canvas({ size: 0.5, label: "size 0.5" }), canvas({ size: 0.33, label: "size 0.33" }))}
  <p><b>fit</b> (advanced): <b>contain</b> keeps the whole clip smaller with grey around (nothing lost, default);
  <b>cover</b> fills the panel and crops; stretch distorts. <b>gap</b> (advanced): a grey separator, in 32 px steps.</p>
  ${row(canvas({ size: 0.5, fit: "contain", label: "contain" }), canvas({ size: 0.5, fit: "cover", label: "cover" }), canvas({ size: 1.0, gap: 2, label: "gap 2" }))}

  <h3>How strongly the panel is pinned</h3>
  <p><b>panel_noise</b>: <b>0</b> copies the panel exactly (swaps, light changes). <b>0.1–0.2</b> gives the model room for big
  changes (a style, a creature, a very different body).</p>
  <p><b>hold</b> (advanced): pin the panel for the whole clip, or only at the start (the rest of the strip is free and cropped).</p>
  ${row(hold("all frames"), hold("first latent frame"))}

  <h3>RoPE (advanced)</h3>
  <p><b>rope_mode</b>: <b>canvas</b> = panel and video share one wide grid (the original duet). <b>shifted</b> = the video keeps
  exactly the positions of a render without the panel, and the panel sits past its edge; <b>rope_gap</b> moves it further
  out. Same quality in tests; keep the gap small at low resolution (a gap close to the video's width makes the model draw
  its own split screen).</p>
  ${row(rope("canvas", 0), rope("shifted", 0), rope("shifted", 3))}

  <h3>Defaults that work</h3>
  <p>euler · beta · 20 steps · CFG 1 · size 1.0 · contain · panel_noise 0 · ${shot ? "mode duet" : "no guide"}.
  ${shot ? "Planner → this node → BFS Shot Join: every shot is pinned and generated in turn; the join cuts the panel off." :
           "Outputs: <code>images</code> + <code>audio</code> (silent when there is no audio VAE) for Create Video; <code>canvas</code> to check the sync; <code>prompt</code> = what was sent."}</p>`;
}

function styles() {
  if (document.getElementById("bfs-duet-help-css")) return;
  const st = document.createElement("style"); st.id = "bfs-duet-help-css";
  st.textContent = `
  .bdh-back{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:10000;display:flex;align-items:center;justify-content:center}
  .bdh{background:${C.bg};color:${C.ink};border:1px solid #3a3a45;border-radius:12px;width:min(860px,94vw);max-height:92vh;overflow:auto;
       padding:18px 22px;font:13px/1.55 system-ui,sans-serif}
  .bdh h2{margin:0 0 8px;font-size:18px}.bdh h3{margin:16px 0 6px;font-size:14px;color:#ffd27a}
  .bdh code{background:#2a2a33;padding:1px 4px;border-radius:4px}
  .bdh table{border-collapse:collapse;margin:6px 0}.bdh td,.bdh th{border:1px solid #33333d;padding:3px 8px;text-align:left}
  .bdh-row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin:6px 0}
  .bdh-cell{background:#141418;border-radius:8px;padding:6px}
  .bdh-close{float:right;cursor:pointer;border:1px solid #444;background:#26262e;color:#ddd;border-radius:6px;padding:2px 10px}`;
  document.head.appendChild(st);
}

function open(type) {
  styles();
  const back = document.createElement("div"); back.className = "bdh-back";
  const guide = new URL("./docs/BFSH3Duet.md", import.meta.url).href;
  back.innerHTML = `<div class="bdh"><button class="bdh-close">close</button>${html(type)}
    <p style="margin-top:14px"><a href="#" class="bdh-guide" style="color:#8fb3ff">Full guide (credits, all settings, a complete example prompt)</a></p></div>`;
  back.addEventListener("pointerdown", e => { if (e.target === back) back.remove(); });
  back.querySelector(".bdh-close").addEventListener("click", () => back.remove());
  back.querySelector(".bdh-guide").addEventListener("click", e => { e.preventDefault(); back.remove(); showGuide(guide); });
  document.body.appendChild(back);
}

app.registerExtension({
  name: "BFSNodes.DuetHelp",
  async nodeCreated(node) {
    if (!NODES.includes(node.comfyClass)) return;
    const w = node.addWidget("button", "❓ How to use", null, () => open(node.comfyClass));
    w.serialize = false;
  },
});
