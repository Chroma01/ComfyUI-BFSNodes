// 📝 Prompts & refs tab: the global reference / prompt, the VLM (shot suggestions) and the reference descriptions ({details}).
import { h } from "../vendor/vue.esm-browser.prod.mjs";
import { fld, check, section, pill, hint, select } from "./common.js";

function globalSection(c) {
  const { plan } = c;
  const slot = (name, label, field) => h("div", { class: "rslot" }, [
    h("div", { class: "refslot", title: name ? name + " (click to replace)" : "click to upload", onClick: () => c.uploadRef("global", field) },
      name ? [h("img", { src: c.viewUrl(name) })] : [label]),
    h("div", { class: "rbtns" }, [name ? h("button", { title: "remove", onClick: () => c.setRef("global", field, "") }, "✕") : null]),
  ]);
  return section("Global reference & prompt", [
    h("div", { class: "row", style: "align-items:flex-start;gap:10px" }, [
      h("div", { class: "refbox" }, [slot(plan.global_ref, "⬆ global\nreference", "global_ref"), slot(plan.global_ref2, "⬆ global\nref 2", "global_ref2")]),
      h("div", { style: "flex:1;min-width:200px" }, [
        h("textarea", { style: "min-height:78px", placeholder: "Global prompt (a connected `prompt` input overrides it)",
          value: plan.global_prompt, onChange: e => c.setPlan("global_prompt", e.target.value) }),
      ]),
    ]),
    h("div", { class: "hint", style: "margin-top:4px" }, [
      "Used by shots without their own. Connected ref_image / ref_image_2 / prompt inputs on the node override these. Placeholders: ",
      h("b", "{target}"), " who is replaced (per shot) · ", h("b", "{details}"), " description of the shot's references · ",
      h("b", "{shot}"), " the VLM's description of the shot · ", h("b", "{setting}"), " the setting picture (H3 Conditioning)."]),
  ], { sub: "defaults for every shot" });
}

function vlmSection(c) {
  const { plan, vlmSug, busy, refSets } = c;
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
        select(VC.describe_preset, ["full body", "head / face", "face attributes", "outfit", "custom"], v => c.setVlmCfg("describe_preset", v), { style: "width:auto" }),
        h("button", { class: "pri", disabled: !!busy.value || !refSets.value.length, onClick: c.describeRefs }, "📝 Describe refs"),
        hint(`${refSets.value.length} reference set(s) · editable below`),
      ]),
      VC.describe_preset === "custom" ? h("textarea", { style: "margin-top:6px;min-height:52px", value: VC.describe_custom || "",
        placeholder: "Your instruction for the VLM, e.g. 'Describe the character's costume and props piece by piece…'",
        onChange: e => c.setVlmCfg("describe_custom", e.target.value) }) : null,
      refSets.value.length ? h("div", { class: "dlist" }, refSets.value.map(x => h("div", { class: "drow" }, [
        h("div", { class: "dthumbs" }, [x.a ? h("img", { src: c.viewUrl(x.a) }) : null, x.b ? h("img", { src: c.viewUrl(x.b) }) : null]),
        h("div", { style: "flex:1" }, [
          hint(x.where.length > 4 ? `${x.where.slice(0, 4).join(", ")} +${x.where.length - 4}` : x.where.join(", ")),
          h("textarea", { style: "min-height:44px", value: plan.ref_details?.[x.key] || "",
            placeholder: "no description yet: Describe refs, write your own, or leave it to the VLM at run time",
            onChange: e => c.setDetail(x.key, e.target.value) }),
        ]),
      ]))) : hint("Add a reference (global, per shot or per person) first."),
    ], { sub: "write {details} in any prompt: each shot gets the description of its own references" }),
  ];
}

export function promptsTab(c) {
  return h("div", { class: "card" }, [globalSection(c), ...vlmSection(c)]);
}
