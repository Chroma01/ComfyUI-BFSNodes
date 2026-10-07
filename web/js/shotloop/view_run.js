// ▶ Run tab: checks before running, how the loop runs (auto / queue), which shots run (filters) and test limits.
import { h } from "../vendor/vue.esm-browser.prod.mjs";
import { fld, check, section, pill, hint, select } from "./common.js";

function checksSection(c) {
  const { checks, tab } = c;
  const list = checks.value;
  const warns = list.filter(x => x.lvl === "warn").length;
  // one row per kind of issue, with the shots it concerns
  const groups = new Map();
  for (const x of list) { const k = `${x.lvl}|${x.text}`; if (!groups.has(k)) groups.set(k, { ...x, shots: [] }); groups.get(k).shots.push(x.shot); }
  const rows = [...groups.values()].sort((a, b) => (a.lvl === "warn" ? 0 : 1) - (b.lvl === "warn" ? 0 : 1));
  const go = i => { c.select(i); tab.value = "shots"; };
  return section("Checks", rows.length ? h("div", { class: "checks" }, rows.map(x => h("div", { class: "ck" }, [
    h("span", { class: "lvl" }, x.lvl === "warn" ? "⚠" : "ℹ"),
    h("span", { style: "flex:1;min-width:0" }, [h("div", x.text), h("div", { class: "row", style: "gap:3px;margin-top:3px" }, [
      ...x.shots.slice(0, 24).map(i => h("button", { class: "chip", title: "go to this shot", onClick: () => go(i) }, `#${i + 1}`)),
      x.shots.length > 24 ? hint(`+${x.shots.length - 24} more`) : null])]),
    pill(`${x.shots.length} shot(s)`, x.lvl === "warn" ? "warn" : ""),
  ]))) : h("div", { class: "hint" }, "✓ Nothing to fix in the shots that run."),
  { sub: "only shots that run", right: list.length ? pill(`${warns} warning(s) · ${list.length - warns} note(s)`, warns ? "warn" : "") : pill("all good", "ok") });
}

function loopSection(c) {
  const { plan, prog, active } = c;
  const num = (k, step, min) => h("input", { type: "number", step, min, value: plan[k], onChange: e => c.setPlan(k, parseFloat(e.target.value) || 0) });
  return section("Run", [
    h("div", { class: "grid" }, [
      fld("Run", select(plan.run, [["auto", "Auto loop (one run)"], ["queue", "Queue loop (one shot per run)"]], v => c.setPlan("run", v, c.progress))),
      fld("Skipped shots in the output", select(plan.skip_fill, [["original", "Keep original video"], ["drop", "Remove them"]], v => c.setPlan("skip_fill", v))),
      fld("Max shots (0 = all)", num("max_parts", 1, 0), "quick tests"),
      fld("Max total seconds (0 = all)", num("max_total_s", 0.5, 0), "quick tests"),
    ]),
    plan.run === "queue" ? h("div", { style: "margin-top:8px" }, [
      h("div", { class: "row", style: "margin-bottom:4px" }, [h("b", "Queue loop"), h("span", { class: "grow" }),
        hint(`${prog.done}/${prog.count || active.value.length} shots done`)]),
      h("div", { class: "prog" }, [h("div", { style: `width:${(100 * prog.done / Math.max(1, prog.count || active.value.length)).toFixed(1)}%` })]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        check(plan.auto_continue, v => c.setPlan("auto_continue", v), "Auto-queue the next shot"),
        h("button", { onClick: () => c.progress(false) }, "↻ Status"), h("button", { class: "dng", onClick: () => c.progress(true) }, "⟲ Reset loop"),
      ]),
      hint("Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video.", "display:block;margin-top:4px"),
    ]) : null,
  ], { sub: `${active.value.length} shot(s) will run` });
}

function filtersSection(c) {
  const { plan, stats, segs, busy, filtersOn } = c;
  const F = plan.filters;
  const chk = (k, label) => check(F[k], v => c.setFilter(k, v), label);
  const fnum = (k, step, label, help) => fld(label, h("input", { type: "number", step, min: 0, value: F[k], onChange: e => c.setFilter(k, parseFloat(e.target.value) || 0) }), help);
  const skippedN = segs.value.filter(s => c.skipWhy(s)).length;
  return section("Filters", [
    h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
      chk("person", "Needs a person"), chk("face", "Needs a face"), chk("skip_dark", "Skip dark / fades"), chk("skip_static", "Skip static shots"),
    ]),
    h("div", { class: "grid" }, [
      fnum("min_person_area", 0.01, "Min person size (0-1 of frame)", "e.g. 0.03 skips wide shots"),
      fnum("max_persons", 1, "Max people (0 = any)", "skip crowds"),
      fnum("min_frames", 1, "Min frames (0 = off)"),
      fnum("samples", 1, "Frames sampled / shot"),
      fnum("dark_level", 0.01, "Dark below (0-1)"),
      fnum("static_level", 0.001, "Static below"),
    ]),
    h("div", { class: "row", style: "margin-top:8px" }, [
      h("button", { class: "pri", disabled: !!busy.value, onClick: c.analyzeContent }, "👤 Analyse people & faces"),
      hint(stats.value.length ? `stats for ${stats.value.length} shots · YOLO person/face from models/ultralytics` : "runs the detectors on a few frames of every shot"),
    ]),
  ], { open: filtersOn.value || stats.value.length > 0, sub: "skipped shots do not run",
       right: filtersOn.value ? pill(`${skippedN} skipped`, "warn") : pill("off") });
}

export function runTab(c) {
  return h("div", { class: "card" }, [checksSection(c), loopSection(c), filtersSection(c)]);
}
