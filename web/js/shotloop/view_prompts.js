// 📝 Prompts & refs tab: the global reference / prompt, the VLM (shot suggestions) and the reference descriptions ({details}).
import { h, fld, check, section, pill, hint, select, refBox, promptChips, MAX_REFS, MARK_PALETTE, pictureColor } from "./common.js";

const PRESETS = ["short", "object", "full body", "head / face", "face attributes", "outfit", "custom"];

function globalSection(c) {
  const { plan } = c;
  // Colored Mark: every picture has its mark color (a person marked in it becomes that picture)
  const marker = k => h("label", { class: "guidecolor" }, [
    h("span", { class: "hint" }, "Mark color"),
    h("select", { value: pictureColor(plan, k + 1), title: `Colored Mark color of <Picture ${k + 1}>`,
      onChange: e => c.setPictureColor(k + 1, e.target.value) },
      MARK_PALETTE.map(([v, name]) => h("option", { value: v }, name))),
  ]);
  return section("Global reference & prompt", [
    refBox(c, "global", plan.global_refs, [], marker),
    h("div", { class: "pbox", style: "margin-top:6px" }, [
      h("textarea", { style: "min-height:78px", placeholder: "Global prompt (a connected `prompt` input overrides it)",
        value: plan.global_prompt, onChange: e => c.setPlan("global_prompt", e.target.value) }),
      promptChips(c, plan.global_refs.filter(Boolean), v => c.setPlan("global_prompt", v)),
    ]),
    h("div", { class: "hint", style: "margin-top:4px" }, [
      `Used by shots without their own (up to ${MAX_REFS} pictures, <Picture 1> first; click + or drop image files on the box). A connected prompt input on the node overrides the global prompt. Placeholders: `,
      h("b", "{target}"), " who is replaced (per shot) · ", h("b", "{details}"), " description of the shot's references · ",
      h("b", "{details@Picture n}"), " description of picture n alone · ",
      h("b", "{shot}"), " the VLM's description of the shot · ", h("b", "{setting}"), " the setting picture (H3 Conditioning)."]),
    h("div", { class: "hint", style: "margin-top:4px" }, "Mark color: the Colored Mark color of each picture. A person marked for <Picture n> in a shot is drawn in that color, and the prompt says which picture the color becomes. Change it when it is too close to a color in the video."),
  ], { sub: "defaults for every shot" });
}

function vlmSection(c) {
  const { plan, vlmSug, busy, pictures } = c;
  const VC = plan.vlm_cfg || {};
  const opt = (k, label, title) => check(VC[k], v => c.setVlmCfg(k, v), label, title);
  const sugN = Object.keys(vlmSug.value).length;
  return [
    section("VLM", [
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        opt("enabled", "Use the VLM when the workflow runs", "At run time the planner asks the VLM about the shots it runs and applies the options below"),
        opt("auto_segment", "Fill the mask text", "Shots without a mask text or points get the VLM's segment suggestion"),
        opt("auto_shot", "Fill {shot} in the prompt", "Write {shot} in a prompt: it becomes the VLM's description of that shot (camera, framing, action)"),
      ]),
      h("div", { class: "grid" }, [
        fld("Frames per shot", h("input", { type: "number", min: 1, max: 8, step: 1, value: VC.frames, onChange: e => c.setVlmCfg("frames", parseInt(e.target.value) || 3) })),
        fld("Max tokens", h("input", { type: "number", min: 64, max: 2048, step: 32, value: VC.max_tokens, onChange: e => c.setVlmCfg("max_tokens", parseInt(e.target.value) || 1024) })),
        fld("Image max side (VLM)", select(String(VC.image_max_side || 1024),
          [448, 672, 896, 1024, 1280, 1536, 2048].map(n => [String(n), `${n} px`]),
          v => c.setVlmCfg("image_max_side", parseInt(v) || 1024)), "VLM image sizing"),
      ]),
      h("textarea", { style: "margin-top:6px;min-height:44px", placeholder: "Extra instruction for the VLM (optional), e.g. 'segment the woman, not the man'",
        value: VC.instruction || "", onChange: e => c.setVlmCfg("instruction", e.target.value) }),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("button", { class: "pri", disabled: !!busy.value, onClick: c.analyseVLM }, "🤖 Analyse shots"),
        h("button", { disabled: !sugN, onClick: c.applyAllSug, title: "Mask text for shots without one, skip where the VLM says skip" }, "Apply suggestions → all"),
      ]),
      hint("Connect a Qwen3-VL (CLIPLoader) to the planner's vlm input. The VLM buttons work after the workflow has run once with it connected (ComfyUI only hands models to nodes when they run). Each shot's suggestion shows in its editor.", "display:block;margin-top:4px"),
    ], { sub: "looks at every shot and suggests the mask, a description and whether to run it",
         right: h("span", { class: "row" }, [VC.enabled ? pill("on", "ok") : pill("off"), sugN ? pill(`${sugN} suggestions`) : null]) }),
    section("Describe references ({details})", [
      h("div", { class: "row", style: "gap:8px" }, [
        fld("Describe", select(VC.describe_preset, PRESETS, v => c.setVlmCfg("describe_preset", v), { style: "width:auto" }),
          "what every picture gets, unless it has its own below"),
        h("button", { class: "pri", disabled: !!busy.value || !pictures.value.length, onClick: () => c.describePictures() }, "📝 Describe refs"),
        h("button", { disabled: !!busy.value || !pictures.value.length,
          title: "Try other descriptions: a new seed (and some temperature if it is 0), then describe again",
          onClick: () => { if (!(+VC.temperature > 0)) c.setVlmCfg("temperature", 0.7); c.setVlmCfg("seed", (+VC.seed || 0) + 1); c.describePictures(); } }, "🎲"),
        fld("Temperature", h("input", { type: "number", min: 0, max: 1.5, step: 0.1, value: VC.temperature ?? 0, style: "width:70px",
          title: "0 = always the same answer (greedy); 0.5-1 = varied answers, picked by the seed",
          onChange: e => c.setVlmCfg("temperature", Math.max(0, parseFloat(e.target.value) || 0)) })),
        fld("Seed", h("input", { type: "number", min: 0, step: 1, value: VC.seed ?? 0, style: "width:80px",
          title: "Used when the temperature is above 0: another seed, another description",
          onChange: e => c.setVlmCfg("seed", Math.max(0, parseInt(e.target.value) || 0)) })),
        hint(`${pictures.value.length} picture(s) · editable below`),
        check(VC.fit_prompt !== false, v => c.setVlmCfg("fit_prompt", v), "fit the prompt",
          "The VLM gets the global prompt's sentence about that picture ({details}, {details2} or {details@Picture n}; and the shot prompt's {target} sentence for 🧑 Describe target) with a blank in its place, and writes the description to fit it: the description adapts to the prompt, not the other way round."),
      ]),
      VC.describe_preset === "custom" ? h("textarea", { style: "margin-top:6px;min-height:52px", value: VC.describe_custom || "",
        placeholder: "Your instruction for the VLM, e.g. 'Describe the character's costume and props piece by piece…'",
        onChange: e => c.setVlmCfg("describe_custom", e.target.value) }) : null,
      pictures.value.length ? h("div", { class: "dlist" }, pictures.value.map(x => {
        const own = plan.picture_cfg?.[x.name] || {};
        return h("div", { class: "drow" }, [
          h("div", { class: "dthumbs" }, [h("img", { src: c.viewUrl(x.name) })]),
          h("div", { style: "flex:1" }, [
            h("div", { class: "row", style: "gap:6px;margin-bottom:3px" }, [
              hint(`Picture n in ${x.where.length > 4 ? `${x.where.slice(0, 4).join(", ")} +${x.where.length - 4}` : x.where.join(", ")}`),
              h("span", { class: "grow" }),
              select(own.preset || "", [["", `default (${VC.describe_preset || "short"})`], ...PRESETS],
                v => c.setPictureCfg(x.name, "preset", v), { style: "width:auto", title: "What to describe in this picture" }),
              h("button", { class: "sm", disabled: !!busy.value, title: "Describe only this picture again",
                onClick: () => c.describePictures([x.name]) }, "🔄"),
            ]),
            own.preset === "custom" ? h("textarea", { style: "min-height:36px;margin-bottom:3px", value: own.custom || "",
              placeholder: "Instruction for this picture, e.g. 'Describe only the outfit, piece by piece'",
              onChange: e => c.setPictureCfg(x.name, "custom", e.target.value) }) : null,
            h("textarea", { style: "min-height:44px", value: plan.picture_details?.[x.name] || "",
              placeholder: "no description yet: 📝 Describe refs, 🔄, write your own, or leave it to the VLM at run time",
              onChange: e => c.setPictureDetail(x.name, e.target.value) }),
          ]),
        ]);
      })) : hint("Add a reference (global or per shot) first."),
    ], { sub: "{details} = picture 1, {details2} = picture 2, {details@Picture n} = picture n: each shot gets the descriptions of its own pictures" }),
  ];
}

export function promptsTab(c) {
  return h("div", { class: "card" }, [globalSection(c), ...vlmSection(c)]);
}
