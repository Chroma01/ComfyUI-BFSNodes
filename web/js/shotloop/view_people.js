// 👥 People & masks tab: the cast found by face (each person linked to a reference) and the global SAM 3 mask settings.
import { h } from "../vendor/vue.esm-browser.prod.mjs";
import { fmtT, fld, check, section, pill, hint } from "./common.js";

function castSection(c) {
  const { plan, people, busy, usedRefs } = c;
  const fps = plan.fps;
  const castN = people.value.filter(p => c.linked(p.id)).length;
  const opt = (k, label, title) => check(plan[k], v => c.setCastOpt(k, v), label, title);
  return section("Cast", [
    h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
      opt("cast_assign", "Shots use their main person's reference", "a shot without its own reference takes the reference of the person with the most screen time in it"),
      opt("cast_split", "Split where the main person changes", "adds a boundary when the biggest face switches to someone else (re-splits the plan)"),
      opt("cast_only", "Only run shots with a linked person", "shots where no linked person appears are skipped"),
    ]),
    people.value.length ? h("div", { class: "cast" }, people.value.map(p => {
      const cc = c.castOf(p.id);
      return h("div", { class: ["person", cc.ignore && "ign"] }, [
        h("img", { class: "face", src: p.thumb, title: `first seen ${fmtT(p.first, fps)} (click to seek)`, onClick: () => c.seek(p.first) }),
        h("div", { class: "t" }, [h("b", `Person ${p.id}`), ` · ${(p.share * 100).toFixed(1)}%`]),
        h("div", { class: "t" }, `${fmtT(p.first, fps)} → ${fmtT(p.last, fps)}`),
        h("div", { class: "refbox" }, ["ref", "ref2"].map(k => h("div", { class: "rslot" }, [
          h("div", { class: "refslot sm", title: cc[k] ? cc[k] + " (click to replace)" : "click to upload",
            onClick: () => c.pickFile("image/*", name => c.setCast(p.id, k, name)) }, cc[k] ? [h("img", { src: c.viewUrl(cc[k]) })] : [k === "ref" ? "⬆ ref" : "⬆ ref 2"]),
          cc[k] ? h("div", { class: "rbtns" }, [h("button", { title: "remove", onClick: () => c.setCast(p.id, k, "") }, "✕")]) : null,
        ]))),
        usedRefs.value.length ? h("div", { class: "recent sm" }, usedRefs.value.map(n => h("img", { src: c.viewUrl(n), title: `${n} (click = ref, shift+click = ref 2)`,
          onClick: e => c.setCast(p.id, e.shiftKey ? "ref2" : "ref", n) }))) : null,
        check(cc.ignore, v => c.setCast(p.id, "ignore", v), "ignore"),
      ]);
    })) : h("div", { class: "hint" }, "Nobody found yet."),
    h("div", { class: "row", style: "margin-top:8px" }, [
      h("button", { class: "pri", disabled: !!busy.value, onClick: () => c.findPeople(true) }, people.value.length ? "↻ Re-analyse people" : "👥 Find people"),
      hint(people.value.length ? "faces sampled every 0.5 s and grouped by identity (InsightFace, CPU)" : "detects faces across the video and groups them into people"),
    ]),
  ], { sub: "track faces, link each person to a reference, split and assign shots by who is on screen",
       right: people.value.length ? pill(`${people.value.length} people · ${castN} linked`) : pill("not analysed") });
}

function maskSection(c) {
  const { plan, segs, showMasks } = c;
  const MC = plan.mask_cfg || {};
  const mnum = (k, step, label, help, scale = 1) => fld(label, h("input", { type: "number", step, min: 0, value: +(MC[k] * scale).toFixed(3),
    onChange: e => c.setMaskCfg(k, (parseFloat(e.target.value) || 0) / scale) }), help);
  const S = segs.value;
  const counts = { mask: 0, crop: 0, cropmask: 0, paste: 0 };
  S.forEach(s => { const m = c.modeOf(s); if (m in counts) counts[m]++; });
  const masked = S.filter(s => s.mask?.text || (s.mask?.points || []).length || s.mask?.video || s.extMask).length;
  return section("Mask & crop settings", [
    h("div", { class: "note", style: "margin-bottom:8px" }, [
      "Each shot picks what it marks (🎯 text or points, or 🎞 its own mask video) and how it is generated (", h("b", "Full frame · Frame + paste · Mask only · Crop · Crop + mask"),
      ") in the Shots tab. Masks from a video (here, per shot, or the planner's ", h("b", "mask"), " input) replace SAM 3. These settings apply to every shot."]),
    h("div", { class: "row", style: "margin-bottom:8px;align-items:center" }, [
      h("span", { style: "white-space:nowrap", title: "A black/white video of the WHOLE source video (white = the subject), e.g. rotoscoped in another tool. Shots without their own mask (video, text or points) use it instead of SAM 3. The planner's optional mask input does the same from the graph and wins over this." }, "🎞 Mask video (whole video)"),
      h("select", { value: plan.mask_video || "", style: "flex:1;min-width:180px", onChange: e => c.setGlobalMaskVideo(e.target.value) },
        [h("option", { value: "" }, "— none: SAM 3 per shot —"), ...c.files.videos.map(v => h("option", { value: v }, v))]),
      h("button", { onClick: () => c.pickFile("video/*", c.setGlobalMaskVideo) }, "⬆ Upload"),
      plan.mask_video ? h("button", { onClick: () => c.setGlobalMaskVideo("") }, "✕") : null,
    ]),
    h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
      check(MC.fill_holes, v => c.setMaskCfg("fill_holes", v), "Fill holes", "Fill enclosed gaps so each region is solid"),
      check(MC.invert, v => c.setMaskCfg("invert", v), "Invert", "Use everything except the segmented object"),
      check(showMasks.value, v => { showMasks.value = v; }, "Show masks on shots", "Overlay the mask preview on the shot cards"),
    ]),
    h("div", { class: "grid" }, [
      mnum("expand", 1, "Expand (px)", "grow the mask (paste-back and generation mask)"),
      mnum("temporal_expand", 1, "Temporal expand (frames)", "hold the mask a few frames: less flicker"),
      mnum("threshold", 0.05, "Detection threshold", "SAM 3 score to keep an object"),
      mnum("max_objects", 1, "Max objects", "how many matches of the text are tracked"),
      mnum("padding", 1, "Crop padding (%)", "Crop modes: context around the mask's box", 100),
      mnum("feather", 1, "Feather (px)", "Crop modes: soft edge of the paste"),
      fld("Paste back", h("select", { value: MC.paste, onChange: e => c.setMaskCfg("paste", e.target.value) },
        [h("option", { value: "mask" }, "only the mask (feathered)"), h("option", { value: "box" }, "the whole box (feathered)")]), "Crop modes"),
      mnum("blockify", 1, "Blockify (px, 0 = off)", "square blocks; 16 matches H3's latent grid"),
    ]),
    hint("Uses the official sam3.1_multiplex_fp16 checkpoint (models/checkpoints), downloaded from Comfy-Org/sam3.1 the first time.", "display:block;margin-top:6px"),
  ], { sub: "SAM 3",
       right: h("span", { class: "row" }, [pill(`${masked} masked`), counts.mask ? pill(`${counts.mask} mask only`, "acc") : null, counts.paste ? pill(`${counts.paste} frame+paste`, "acc") : null,
         counts.crop + counts.cropmask ? pill(`${counts.crop + counts.cropmask} cropped`, "warn") : null]) });
}

export function peopleTab(c) {
  return h("div", { class: "card" }, [castSection(c), maskSection(c)]);
}
