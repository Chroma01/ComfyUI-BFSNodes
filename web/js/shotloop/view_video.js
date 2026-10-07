// 🎞 Video tab: the source video, how it is split into shots, and the generation size.
import { h } from "../vendor/vue.esm-browser.prod.mjs";
import { GRIDS, fld, select, section, pill, hint } from "./common.js";

export function videoTab(c) {
  const { plan, an, files, size, busy, detectorUsed, maxLen } = c;
  const fps = plan.fps;
  const num = (k, step = 0.1, min = 0) => h("input", { type: "number", step, min, value: plan[k], onChange: e => c.setPlan(k, parseFloat(e.target.value) || 0) });
  const sel_ = (k, opts, after) => select(plan[k], opts, v => c.setPlan(k, v, after));
  const au = an.value?.audio;

  const source = section("Source video", [
    h("div", { class: "row" }, [
      h("div", { style: "flex:1;min-width:180px" }, [h("select", { value: plan.video, onChange: e => c.setVideo(e.target.value) },
        [h("option", { value: "" }, "— choose a video from the input folder —"), ...files.videos.map(v => h("option", { value: v }, v))])]),
      h("button", { class: plan.video ? "" : "pri", onClick: () => c.pickFile("video/*", c.setVideo) }, "⬆ Upload"),
      h("button", { onClick: c.refreshFiles, title: "refresh the input folder" }, "↻"),
    ]),
    an.value ? h("div", { class: "row", style: "margin-top:6px" }, [
      pill(`${an.value.width}×${an.value.height}`), pill(`${an.value.fps_src.toFixed(2)} fps`), pill(`${an.value.duration.toFixed(2)} s`),
      pill(`timeline ${fps} fps · ${an.value.n} frames`), pill(`generate at ${size.w}×${size.h}`, "acc"),
      au?.has_audio
        ? pill(`🔊 audio · ${au.sample_rate ? (au.sample_rate / 1000).toFixed(1) + " kHz" : "?"} · ${au.channels || "?"} ch`, "ok", au.codec || "")
        : pill(`🔇 ${au?.reason || "no audio"} → silent track`, "warn", "The outputs carry a silent track of the right length, so Create Video works"),
      h("select", { value: plan.audio_mode, style: "width:auto", disabled: !au?.has_audio, title: "Audio of the planner's and the join's outputs",
        onChange: e => c.setPlan("audio_mode", e.target.value) },
        [h("option", { value: "auto" }, "use the video's audio"), h("option", { value: "silent" }, "silent track")]),
    ]) : plan.video ? null : h("div", { class: "empty" }, [h("b", "Pick or upload a video to start"),
      "It is split into shots the model can generate; every shot then gets its own reference, prompt and mask."]),
  ]);

  const split = section("Split into shots", [
    h("div", { class: "grid" }, [
      fld("Mode", sel_("mode", [["shots", "Camera cuts"], ["fixed", "Fixed length"]])),
      fld("Detector", sel_("detector", [["adaptive", "PySceneDetect adaptive"], ["content", "PySceneDetect content"], ["builtin", "Built-in"]])),
      fld(`Sensitivity ${Number(plan.sensitivity).toFixed(2)}`, h("input", { type: "range", min: 0, max: 1, step: 0.05, value: plan.sensitivity,
        onInput: e => { plan.sensitivity = parseFloat(e.target.value); }, onChange: c.save })),
      fld("Frame grid", sel_("grid", Object.keys(GRIDS))),
      fld("Max seconds / shot", num("max_s", 0.1, 0.2), `= ${maxLen.value} frames (${(maxLen.value / fps).toFixed(2)} s)`),
      fld("Min seconds / shot", num("min_s", 0.1, 0)),
      fld("Timeline fps", num("fps", 1, 1), "re-analyses"),
    ]),
    h("div", { class: "row", style: "margin-top:8px" }, [
      h("button", { class: "pri", disabled: !plan.video || !!busy.value, onClick: () => c.autoSplit(true) }, "✂ Auto split"),
      h("button", { disabled: !plan.video, onClick: c.analyze }, "↻ Re-analyse"),
      detectorUsed.value ? hint(`detector: ${detectorUsed.value}`) : null,
      hint("Auto split replaces the boundaries; per-shot settings stay with the shot at the same position."),
    ]),
  ], { sub: "camera cuts or fixed length, on the model's frame grid" });

  const gen = section("Generation size", h("div", { class: "grid" }, [
    fld("Megapixels", num("megapixels", 0.01, 0.02), size.w ? `${size.w}×${size.h}` : ""),
    fld("Size multiple", num("multiple", 8, 8)),
  ]), { sub: "cropped shots use it for the crop" });

  return h("div", { class: "card" }, [source, an.value ? split : null, an.value ? gen : null]);
}
