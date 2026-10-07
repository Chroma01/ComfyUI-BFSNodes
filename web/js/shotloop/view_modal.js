// The busy bar and the points window (🎯 Points…: click on a frame to pick the subject for SAM 3).
import { h, Teleport } from "../vendor/vue.esm-browser.prod.mjs";
import { hint } from "./common.js";

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
  const { modal, segs, plan, api } = c;
  const ms = segs.value[modal.idx];
  if (!modal.open || !ms) return null;
  return h(Teleport, { to: "body" }, h("div", { class: "bsl", style: "background:none;padding:0" }, h("div", { class: "mmodal",
    onKeydown: e => e.stopPropagation(), onPointerdown: e => { if (e.target === e.currentTarget) modal.open = false; } }, [
    h("div", { class: "mbox" }, [
      h("div", { class: "row", style: "margin-bottom:6px" }, [h("span", { class: "sect" }, `Shot #${modal.idx + 1} · points`),
        hint("click = keep (green) · right-click or shift+click = exclude (red) · click a point to remove it")]),
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
        hint(`frame ${ms.start + modal.key} (${modal.key + 1}/${ms.len})`),
        h("input", { type: "range", min: 0, max: ms.len - 1, step: 1, value: modal.key, style: "flex:1",
          onInput: e => { modal.key = parseInt(e.target.value); modal.prev = null; } }),
      ]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("input", { type: "text", value: modal.text, style: "flex:1", placeholder: "text prompt (used when there are no points)",
          onChange: e => { modal.text = e.target.value; } }),
        h("button", { onClick: () => { modal.points = []; modal.prev = null; } }, "Clear points"),
        h("button", { class: "pri", disabled: !!modal.busy || !(modal.points.length || modal.text),
          onClick: () => c.previewMask(modal.idx, { points: modal.points, key: modal.key, text: modal.text }, true) }, "👁 Segment"),
      ]),
      modal.busy ? busyBar(c, modal.busy) : null,
      modal.prev ? h("div", { class: "mstrip" }, [...modal.prev.frames.map(f => h("img", { src: f.src, title: `frame ${f.f}` })),
        hint(modal.prev.empty ? "nothing found" : `covers ${(modal.prev.coverage * 100).toFixed(1)}%`)]) : null,
      h("div", { class: "row", style: "margin-top:8px;justify-content:flex-end" }, [
        h("button", { onClick: () => { modal.open = false; } }, "Cancel"),
        h("button", { title: "Apply these points to EVERY shot, each on its frame at the same relative position (a subject that stays in place)",
          onClick: c.savePointsAll }, "Save → all shots"),
        h("button", { class: "pri", onClick: c.savePoints }, "Save (this shot)"),
      ]),
    ]),
  ])));
}
