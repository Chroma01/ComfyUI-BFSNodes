// Shot Planner: constants and small helpers shared by the panel and its views (no side effects: ComfyUI loads every
// .js under web/js as an extension, so this module only exports).
import { h } from "../vendor/vue.esm-browser.prod.mjs";

export const GRIDS = { "H3 (17n+5)": [17, 5], "LTX / Wan (8n+1)": [8, 1], "Wan (4n+1)": [4, 1], "any": [1, 0] };
export const DEFAULTS = {
  video: "", fps: 24, grid: "H3 (17n+5)", mode: "shots", max_s: 4.5, min_s: 1.0, sensitivity: 0.5,
  max_parts: 0, max_total_s: 0, bounds: [], segs: [], global_ref: "", global_ref2: "", global_prompt: "", mask_video: "",
  megapixels: 0.15, multiple: 32, detector: "adaptive", run: "auto", auto_continue: true,
  skip_fill: "original", audio_mode: "auto", ref_details: {}, mask_cfg: {}, vlm_cfg: {}, cast: {}, cast_assign: true, cast_split: false, cast_only: false,
  filters: { person: false, min_person_area: 0, max_persons: 0, face: false, skip_dark: false, dark_level: 0.06,
             skip_static: false, static_level: 0.004, min_frames: 0, samples: 6 },
};

// global mask settings (same defaults as the server's DEFAULT_MASK)
export const MASK_DEFAULTS = { threshold: 0.5, max_objects: 4, invert: false, fill_holes: true, temporal_expand: 2, blockify: 0,
                               padding: 0.15, expand: 16, feather: 12, paste: "mask" };

export const VLM_DEFAULTS = { enabled: false, frames: 3, max_tokens: 1024, auto_segment: true, auto_shot: true, instruction: "",
                              describe_preset: "full body", describe_custom: "" };

// hover help for every setting (the 📖 Guide button opens the full documentation)
export const TIPS = {
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
  "Expand": "Grow the mask by this many pixels: before pasting a cropped result back, and for the generation mask (inpaint 'only the mask'). Raise it when the new person is bigger than the old one (longer hair, wider body).",
  "Feather": "Soft edge of the paste-back, in pixels.",
  "Temporal expand": "Hold the mask this many frames before and after: removes flicker where SAM 3 drops a frame.",
  "Blockify": "Square blocks of this size (0 = off). 16 matches H3's latent grid, useful as an inpainting mask.",
  "Detection threshold": "SAM 3 score needed to keep a text match. Lower finds more (and more wrong) objects.",
  "Max objects": "How many matches of the text prompt are tracked together.",
  "Paste back": "Only the mask: the result replaces the masked pixels. The whole box: the full crop is pasted with feathered borders.",
  "Frames per shot": "How many frames of each shot the VLM looks at.",
  "Max tokens": "Longest VLM answer. Raise it for long descriptions.",
};

export const snapUp = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.ceil((n - o) / s);
};
export const snapDown = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.floor((n - o) / s);
};
export const hue = i => `hsl(${(i * 47 + 200) % 360} 55% 46%)`;
export const fmtT = (f, fps) => { const s = f / fps; return `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`; };
export const segKey = s => `${s.start}-${s.end}`;
// a shot has a mask: its own mask video, SAM 3 text or points, or the plan's mask video for the whole video
// (the planner's `mask` input is only known when the workflow runs)
export const hasMask = s => !!(s.mask?.video || s.mask?.text || (s.mask?.points || []).length || s.extMask);
export const maskSource = s => s.mask?.video ? "video" : (s.mask?.text || (s.mask?.points || []).length) ? "sam" : s.extMask ? "global" : "";

// ---- small view helpers
export const pill = (text, kind = "", title = "") => h("span", { class: ["pill", kind], title }, text);
export const hint = (text, style = "") => h("span", { class: "hint", style }, text);
export const fld = (label, input, help) => {
  const tip = Object.entries(TIPS).find(([k]) => String(label).startsWith(k))?.[1] || "";
  return h("div", { class: "fld", title: tip }, [h("label", tip ? [label, h("span", { class: "qm" }, " ⓘ")] : label), input,
    help ? h("div", { class: "hint" }, help) : null]);
};
export const check = (checked, onChange, label, title = "") =>
  h("label", { class: "chk", title }, [h("input", { type: "checkbox", checked: !!checked, onChange: e => onChange(e.target.checked) }), label]);
export const select = (value, options, onChange, attrs = {}) => h("select", { value, onChange: e => onChange(e.target.value), ...attrs },
  options.map(o => h("option", { value: Array.isArray(o) ? o[0] : o }, Array.isArray(o) ? o[1] : o)));
// a titled section inside a card; `open` makes it a collapsible <details>
export const section = (title, body, { sub = "", right = null, open = null, cls = "" } = {}) => {
  const head = [h("span", { class: "sect" }, title), sub ? hint(sub) : null, h("span", { class: "grow" }), right];
  return open === null
    ? h("div", { class: ["sec", cls] }, [h("div", { class: "sech" }, head), ...[body].flat()])
    : h("details", { class: ["sec", cls], open }, [h("summary", { class: "sech" }, [h("span", { class: "caret" }, "▸"), ...head]), ...[body].flat()]);
};
