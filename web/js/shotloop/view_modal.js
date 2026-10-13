// The busy bar and the points window (🎯 Points…: click on a frame to pick the subject for SAM 3).
import { Teleport } from "../vendor/vue.esm-browser.prod.mjs";
import { h, hint, t } from "./common.js";

export function busyBar(c, msg) {
  const { busySince, now, upPct, status } = c;
  const secs = busySince.value ? Math.max(0, Math.round((now.value - busySince.value) / 1000)) : 0;
  const pct = upPct.value >= 0 ? upPct.value : (status.total > 0 ? Math.min(1, status.done / status.total) : -1);
  const stage = status.stage && !msg.startsWith(status.stage) ? status.stage : "";
  return h("div", { class: "busybar" }, [
    h("div", { class: "brow" }, [h("span", { class: "spin" }), h("b", msg), stage ? hint(`· ${stage}`) : null,
      h("span", { class: "grow" }), pct >= 0 ? h("span", `${Math.round(pct * 100)}%`) : null, hint(`${secs}s`)]),
    h("div", { class: ["bprog", pct < 0 && "ind"] }, [h("div", { style: pct >= 0 ? `width:${(pct * 100).toFixed(1)}%` : "" })]),
  ]);
}

export function pointsModal(c) {
  const { modal, segs, plan, api, setMeta } = c;
  const ms = segs.value[modal.idx];
  if (!modal.open || !ms) return null;
  return h(Teleport, { to: "body" }, h("div", { class: "bsl", style: "background:none;padding:0" }, h("div", { class: "mmodal",
    onKeydown: e => e.stopPropagation(), onPointerdown: e => { if (e.target === e.currentTarget) modal.open = false; } }, [
    h("div", { class: "mbox" }, [
      h("div", { class: "row", style: "margin-bottom:6px" }, [h("span", { class: "sect" }, `Shot #${modal.idx + 1} · points`),
        ms.color_guide ? h("label", { class: "row", style: "gap:6px;align-items:center" }, [h("span", { class: "hint" }, `Region ${modal.region} · color`),
          h("select", { value: modal.region === 2 ? (ms.color_guide_color2 || plan.global_color_guide_ref2 || "#9b30ff") : (ms.color_guide_color || plan.global_color_guide_ref1 || "#00ff00"), style: "width:auto", onChange: e => {
            setMeta(modal.idx, modal.region === 2 ? "color_guide_color2" : "color_guide_color", e.target.value);
            if (modal.prev) c.previewMask(modal.idx, { points: modal.points, key: modal.key, text: modal.text,
              preview_color: e.target.value }, true);
          } },
            [["#00ff00", "Green"], ["#9b30ff", "Purple"], ["#ff2020", "Red"], ["#168cff", "Blue"], ["#ffe600", "Yellow"], ["#ff00d4", "Magenta"], ["#00e5c0", "Cyan"]].map(([v, n]) => h("option", { value: v, disabled: ms.color_guide && v === (modal.region === 2 ? (ms.color_guide_color || "#00ff00") : (ms.color_guide_color2 || "#9b30ff")) }, n)))]) : null,
        ms.color_guide ? h("div", { class: "row" }, [hint("Colored Mark keeps two independent point sets; switch regions to edit each person. Both use the frame selected below."),
          h("button", { class: modal.region === 1 ? "pri" : "", onClick: () => c.switchPointRegion(1) }, `Region 1 (${modal.groups[1]?.points?.length || 0})`),
          h("button", { class: modal.region === 2 ? "pri" : "", onClick: () => c.switchPointRegion(2) }, `Region 2 (${modal.groups[2]?.points?.length || 0})`)]) : null,
        hint("click = keep (green) · right-click or shift+click = exclude (red) · click a point to remove it")]),
      h("div", { class: "mimg", onContextmenu: e => e.preventDefault(), onPointerdown: e => {
        const box = e.currentTarget.getBoundingClientRect();
        const x = (e.clientX - box.left) / box.width, y = (e.clientY - box.top) / box.height;
        modal.points = [...modal.points, { x: +x.toFixed(4), y: +y.toFixed(4), label: (e.button === 2 || e.shiftKey) ? 0 : 1 }];
        modal.prev = null;
      } }, [
        h("img", { draggable: false, src: api.apiURL(`/bfs/shotloop/frame?video=${encodeURIComponent(plan.video)}&fps=${plan.fps}&f=${ms.start + modal.key}&w=900`) }),
        ...(ms.color_guide ? [1, 2]
          .flatMap(region => (region === modal.region ? modal.points : modal.groups[region]?.points || []).map((p, k) => h("div", {
          class: ["pt", "cg", region === 2 && "r2", p.label ? "pos" : "neg"],
          style: `left:${p.x * 100}%;top:${p.y * 100}%;--point-color:${region === 2 ? (ms.color_guide_color2 || plan.global_color_guide_ref2 || "#9b30ff") : (ms.color_guide_color || plan.global_color_guide_ref1 || "#00ff00")}`,
          title: `Region ${region}${region === modal.region ? " · click to remove" : ""}`,
          onPointerdown: e => { if (region !== modal.region) return; e.stopPropagation(); modal.points = modal.points.filter((_, j) => j !== k); modal.prev = null; } })))]
          : modal.points.map((p, k) => h("div", { class: ["pt", p.label ? "pos" : "neg"], style: `left:${p.x * 100}%;top:${p.y * 100}%`,
            title: "click to remove", onPointerdown: e => { e.stopPropagation(); modal.points = modal.points.filter((_, j) => j !== k); modal.prev = null; } }))),
      ]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        hint(`frame ${ms.start + modal.key} (${modal.key + 1}/${ms.len})`),
        h("input", { type: "range", min: 0, max: ms.len - 1, step: 1, value: modal.key, style: "flex:1",
          onInput: e => { modal.key = parseInt(e.target.value); modal.prev = null; } }),
      ]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("input", { type: "text", value: modal.text, style: "flex:1", placeholder: "text prompt (used when there are no points)",
          onChange: e => { modal.text = e.target.value; } }),
        h("button", { onClick: () => { modal.points = []; modal.prev = null; } }, "Clear points"),
        h("button", { class: "pri", disabled: !!modal.busy || !(modal.points.length || modal.text),
          onClick: () => c.previewMask(modal.idx, { points: modal.points, key: modal.key, text: modal.text,
            preview_color: modal.region === 2 ? (ms.color_guide_color2 || plan.global_color_guide_ref2 || "#9b30ff")
              : (ms.color_guide_color || plan.global_color_guide_ref1 || "#00ff00") }, true) }, "👁 Segment"),
      ]),
      modal.busy ? busyBar(c, modal.busy) : null,
      modal.prev ? h("div", { class: "mstrip" }, [...modal.prev.frames.map(f => h("img", { src: f.src, title: `frame ${f.f}` })),
        hint(modal.prev.empty ? "nothing found" : `covers ${(modal.prev.coverage * 100).toFixed(1)}%`)]) : null,
      h("div", { class: "row", style: "margin-top:8px;justify-content:flex-end" }, [
        h("button", { onClick: () => { modal.open = false; } }, "Cancel"),
        !ms.color_guide ? h("button", { title: "Apply these points to EVERY shot, each on its frame at the same relative position (a subject that stays in place)",
          onClick: c.savePointsAll }, "Save → all shots") : null,
        h("button", { disabled: !modal.points.length || ms.color_guide,
          title: "Track this selection with SAM 3 through the whole video (or the range), forward and backward from this frame; every shot without its own selection takes its part of the track. Also type a short text (e.g. 'woman in pink top') so the subject is found again after camera cuts.",
          onClick: c.saveTrack }, "🎯 Track → whole video"),
        h("button", { class: "pri", onClick: c.savePoints }, "Save (this shot)"),
      ]),
    ]),
  ])));
}

const TOUR_STEPS = [
  ["video", "Video", "Choose a video, set the frame range, split at camera cuts, then set the generation size."],
  ["shots", "Shots", "Review each shot, assign its references and prompt, and choose how the result is applied."],
  ["people", "People", "Track detected face identities to filter or split shots. Set masks here or per shot."],
  ["prompts", "Prompts", "Set shared references, prompts and optional VLM assistance. Written prompts are kept exactly as authored."],
  ["shots", "Colored Mark", "Colored Mark adds a translucent color overlay to a selected region in the source video. It does not enable inpainting. Choose the color before marking the region with points or a description."],
  ["run", "Run", "Check warnings, choose auto or queue loop, and set filters or quick-run limits."],
];

export function tourModal(c) {
  const { tour, tab } = c;
  const step = tour.value - 1;
  if (step < 0 || step >= TOUR_STEPS.length) return null;
  const [targetTab, title, body] = TOUR_STEPS[step];
  return h(Teleport, { to: "body" }, h("div", { class: "bsl", style: "background:none;padding:0" },
    h("div", { class: "tourback", role: "dialog", "aria-modal": "true", "aria-label": t("Shot Planner tour"),
      onKeydown: e => { if (e.key === "Escape") tour.value = 0; }, onPointerdown: e => { if (e.target === e.currentTarget) tour.value = 0; } }, [
    h("div", { class: "tourbox" }, [
      h("div", { class: "row" }, [h("b", t("Shot Planner tour")), h("span", { class: "grow" }), h("span", { class: "hint" }, `${step + 1} / ${TOUR_STEPS.length}`),
        h("button", { class: "ghost", title: t("Close tour"), "aria-label": t("Close tour"), onClick: () => { tour.value = 0; } }, "✕")]),
      h("h3", t(title)),
      h("p", t(body)),
      h("p", { class: "hint" }, t("This tour is optional. Use Next to visit each main tab; close it whenever you like.")),
      h("div", { class: "row", style: "justify-content:flex-end;margin-top:12px" }, [
        step > 0 ? h("button", { onClick: () => { tour.value--; tab.value = TOUR_STEPS[step - 1][0]; } }, t("Back")) : null,
        h("button", { class: "pri", onClick: () => {
          if (step === TOUR_STEPS.length - 1) tour.value = 0;
          else { tour.value++; tab.value = TOUR_STEPS[step + 1][0]; }
        } }, t(step === TOUR_STEPS.length - 1 ? "Close tour" : "Next")),
      ]),
    ]),
  ])));
}
