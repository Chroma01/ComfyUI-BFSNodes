// ▶ Run tab: checks before running and how the loop runs (auto / queue).
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
  return section("Run", [
    h("div", { class: "grid" }, [
      fld("Run", select(plan.run, [["auto", "Auto loop (one run)"], ["queue", "Queue loop (one shot per run)"]], v => c.setPlan("run", v, c.progress))),
      fld("Skipped shots in the output", select(plan.skip_fill, [["original", "Keep original video"], ["drop", "Remove them"]], v => c.setPlan("skip_fill", v))),
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

export function runTab(c) {
  return h("div", { class: "card" }, [checksSection(c), loopSection(c)]);
}
