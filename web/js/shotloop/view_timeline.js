// Timeline: thumbnails, cut marks, cut score, shot bars (click = select, drag the white handles, double-click = split).
import { h } from "../vendor/vue.esm-browser.prod.mjs";
import { hue, fmtT, pill } from "./common.js";

export function timelineCard(c) {
  const { plan, an, segs, sel, n, pxPerFrame, tlWidth, maxLen, cuts, hover, tlEl, zoom, play, vid, active } = c;
  const S = segs.value, N = n.value, ppf = pxPerFrame.value, fps = plan.fps;
  const thumbs = an.value?.thumbs || [];
  const thumbW = Math.max(8, thumbs.length ? tlWidth.value / thumbs.length : 96);
  const score = an.value?.score || [];
  const maxS = Math.max(4, ...score.slice(0, N));
  const sparkPts = score.slice(0, N).map((v, i) => `${(i * ppf).toFixed(1)},${(24 - Math.min(1, v / maxS) * 22).toFixed(1)}`).join(" ");
  const tooLong = S.filter(s => s.len > maxLen.value).length;
  const totalGen = active.value.reduce((a, s) => a + s.gen, 0);
  return h("div", { class: "card" }, [
    h("div", { class: "row", style: "margin-bottom:6px" }, [
      h("span", { class: "sect" }, "Timeline"),
      pill(`${S.length} shots`), pill(`${cuts.value.length} cuts`), pill(`generate ${totalGen} frames`),
      tooLong ? pill(`${tooLong} too long (> ${maxLen.value} f)`, "warn") : null,
      h("span", { class: "grow" }),
      h("span", { class: "hint" }, "zoom"),
      h("input", { type: "range", min: 1, max: 8, step: 0.5, value: zoom.value, style: "width:100px", onInput: e => { zoom.value = parseFloat(e.target.value); } }),
    ]),
    h("div", { class: "tl", ref: tlEl,
      onPointermove: e => { if (!e.buttons) hover.value = c.frameAt(e); }, onPointerleave: () => { hover.value = null; },
      onClick: e => { if (e.target.classList.contains("hdl")) return; c.seek(c.frameAt(e).frame); } }, [
      h("div", { class: "tlin", style: `width:${tlWidth.value}px` }, [
        h("div", { class: "strip" }, thumbs.map(t => h("img", { src: t.src, style: `width:${thumbW}px` }))),
        ...cuts.value.filter(x => x < N).map(x => h("div", { class: "cut", style: `left:${x * ppf}px`, title: `cut @ ${x}` })),
        h("svg", { class: "spark", viewBox: `0 0 ${tlWidth.value} 26`, preserveAspectRatio: "none" },
          [h("polyline", { points: sparkPts, fill: "none", stroke: "#ff7a90", "stroke-width": 1, "vector-effect": "non-scaling-stroke" })]),
        h("div", { class: "segs", onDblclick: e => c.splitAt(c.frameAt(e).frame) }, [
          ...S.map((s, i) => h("div", {
            class: ["seg", i === sel.value && "sel", !!c.skipWhy(s) && "off", s.len > maxLen.value && "long", play.mode && play.idx === i && "playing"],
            style: `left:${s.start * ppf}px;width:${Math.max(2, s.len * ppf - 1)}px;background:${hue(i)}`,
            title: `#${i + 1} · frames ${s.start}-${s.end - 1} · ${s.len} → ${s.gen}${c.skipWhy(s) ? " · skip: " + c.skipWhy(s) : ""}`,
            onClick: () => { sel.value = i; },
          }, s.len * ppf > 26 ? `${i + 1}` : "")),
          ...plan.bounds.filter(b => b < N).map((b, k) => h("div", { class: "hdl", style: `left:${b * ppf}px`, title: `boundary @ ${b}`, onPointerdown: e => c.drag(k, e) })),
        ]),
        hover.value ? h("div", { class: "ph", style: `left:${hover.value.x}px` }) : null,
        vid.value && plan.video ? h("div", { class: "playhead", style: `left:${Math.min(N, play.frame) * ppf}px` }) : null,
      ]),
      hover.value ? h("div", { class: "tip", style: `left:${Math.min(tlWidth.value - 160, Math.max(0, hover.value.x - (tlEl.value?.scrollLeft || 0) - 70))}px;top:2px` },
        [h("img", { src: c.thumbFor(hover.value.frame) }), h("div", `frame ${hover.value.frame} · ${fmtT(hover.value.frame, fps)}`)]) : null,
    ]),
    h("div", { class: "hint", style: "margin-top:4px" },
      "Click a shot to select it (← → move between shots) · drag the white handles to move a boundary · double-click the shot bar to split · Delete removes the selected shot's cut"),
  ]);
}
