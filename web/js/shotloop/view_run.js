// ▶ Run tab: checks before running and how the loop runs (auto / queue).
import { h, fld, check, section, pill, hint, select } from "./common.js";

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
      fld("Run", select(plan.run, [["recursive", "One run, shot by shot (recommended)"], ["queue", "Queue loop (one shot per run)"],
        ["auto", "Auto loop (all shots at once)"]], v => c.setPlan("run", v, c.progress))),
      plan.run !== "auto" ? fld("Between shots", select(plan.between_shots || "keep", [["keep", "Keep models loaded (fastest)"],
        ["free", "Free memory: unload models (low VRAM)"]], v => c.setPlan("between_shots", v))) : null,
      fld("Skipped shots in the output", select(plan.skip_fill, [["original", "Keep original video"], ["drop", "Remove them"]], v => c.setPlan("skip_fill", v))),
    ]),
    plan.run !== "auto" ? h("div", { style: "margin-top:8px" }, [
      h("div", { class: "row", style: "margin-bottom:4px" }, [h("b", plan.run === "queue" ? "Queue loop" : "Shot by shot"), h("span", { class: "grow" }),
        hint(`${prog.done}/${prog.count || active.value.length} shots done`)]),
      h("div", { class: "prog" }, [h("div", { style: `width:${(100 * prog.done / Math.max(1, prog.count || active.value.length)).toFixed(1)}%` })]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        plan.run === "queue" ? check(plan.auto_continue, v => c.setPlan("auto_continue", v), "Auto-queue the next shot") : null,
        h("button", { onClick: () => c.progress(false) }, "↻ Status"), h("button", { class: "dng", onClick: () => c.progress(true) }, "⟲ Reset loop"),
      ]),
      hint(plan.run === "queue"
        ? "Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video."
        : "One Queue renders every shot in order inside the same run: each shot is stored, then BFS Shot Join starts the next one, so continuity can read the shot before. After a failure, Queue again to resume from the stored shots.",
        "display:block;margin-top:4px"),
    ]) : null,
  ], { sub: `${active.value.length} shot(s) will run` });
}

function memorySection(c) {
  const { memory, vram, size } = c;
  const m = memory.value;
  if (!m.level) return null;
  const k = n => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);
  const pct = vram.max_tokens ? Math.min(100, 100 * m.top.tok / (1.3 * vram.max_tokens)) : 0;
  return section("Memory", [
    h("div", { class: "grid" }, [
      fld("GPU", vram.gpu ? `${vram.gpu} · ${vram.total_gb} GB` : "—"),
      fld("Biggest shot in this plan", m.top.i >= 0 ? `#${m.top.i + 1} · ${m.top.gen} frames · ${size.w}×${size.h} · ${k(m.top.tok)} tokens` : "—"),
      fld("Biggest that ran here", vram.max_tokens ? `${k(vram.max_tokens)} tokens · peak ${vram.max_peak_gb} GB` : "nothing recorded yet"),
    ]),
    vram.max_tokens ? h("div", { class: "prog", style: "margin-top:6px", title: "100% = 1.3× the biggest shot that ran on this GPU" },
      [h("div", { style: `width:${pct.toFixed(1)}%;${m.level === "warn" ? "background:var(--warn,#d97706)" : ""}` })]) : null,
    hint({ ok: "✓ Every shot is within what already ran on this GPU.",
      note: "A bit bigger than anything that ran here: it will probably fit, maybe slower.",
      warn: "⚠ Much bigger than anything that ran here: it may run out of memory. Lower Megapixels or Max seconds / shot, use head 5, or set Between shots = free memory.",
      info: "No shot has run on this GPU yet: after the first one, every shot is compared with what already ran here." }[m.level],
      "display:block;margin-top:4px"),
  ], { sub: { ok: "within what ran here", note: "a bit bigger", warn: "may run out of memory", info: "no data yet" }[m.level] });
}

export function runTab(c) {
  return h("div", { class: "card" }, [checksSection(c), memorySection(c), loopSection(c)]);
}
