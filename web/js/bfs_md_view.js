// Shows a node guide (Markdown in web/js/docs) as a formatted window inside ComfyUI.
// A small renderer for what the guides use: headings, paragraphs, lists, tables, code, bold / italic, links.

const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function inline(s) {
  s = esc(s);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>");
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, t, u) => `<a href="${u}" target="_blank" rel="noopener">${t}</a>`);
  return s;
}

export function renderMarkdown(md) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {                                    // code block
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(esc(lines[i++]));
      i++; out.push(`<pre>${buf.join("\n")}</pre>`); continue;
    }
    const h = l.match(/^(#{1,4})\s+(.*)/);
    if (h) { out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
    if (/^\s*\|/.test(l)) {                                  // table
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = r => r.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map(c => inline(c.trim().replace(/\\\|/g, "|")));
      const body = rows.filter(r => !/^\s*\|[\s:|-]+\|\s*$/.test(r));
      out.push("<table>" + body.map((r, k) => `<tr>${cells(r).map(c => k ? `<td>${c}</td>` : `<th>${c}</th>`).join("")}</tr>`).join("") + "</table>");
      continue;
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(l)) {                     // list (one level, continuation lines joined)
      const ordered = /^\s*\d+\./.test(l); const items = [];
      while (i < lines.length && (/^\s*([-*]|\d+\.)\s+/.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        const m = lines[i].match(/^\s*(?:[-*]|\d+\.)\s+(.*)/);
        if (m) items.push(m[1]); else items[items.length - 1] += " " + lines[i].trim();
        i++;
      }
      out.push(`<${ordered ? "ol" : "ul"}>${items.map(x => `<li>${inline(x)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    if (!l.trim()) { i++; continue; }
    const para = [];                                         // paragraph
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*\||\s*([-*]|\d+\.)\s)/.test(lines[i])) para.push(lines[i++]);
    out.push(`<p>${inline(para.join(" "))}</p>`);
  }
  return out.join("\n");
}

function styles() {
  if (document.getElementById("bfs-md-css")) return;
  const st = document.createElement("style"); st.id = "bfs-md-css";
  st.textContent = `
  .bmd-back{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:10000;display:flex;align-items:center;justify-content:center}
  .bmd{background:#1b1b21;color:#e8e8ee;border:1px solid #3a3a45;border-radius:12px;width:min(900px,94vw);max-height:92vh;overflow:auto;
       padding:18px 24px;font:13px/1.6 system-ui,sans-serif}
  .bmd h1{font-size:20px;margin:4px 0 10px}.bmd h2{font-size:16px;margin:18px 0 6px;color:#ffd27a}.bmd h3{font-size:14px;margin:14px 0 4px}
  .bmd code{background:#2a2a33;padding:1px 4px;border-radius:4px}
  .bmd pre{background:#121217;border:1px solid #2c2c35;border-radius:8px;padding:10px;white-space:pre-wrap;font-size:12px}
  .bmd table{border-collapse:collapse;margin:8px 0;width:100%}.bmd td,.bmd th{border:1px solid #33333d;padding:4px 8px;text-align:left;vertical-align:top}
  .bmd th{background:#24242c}.bmd a{color:#8fb3ff}
  .bmd-close{float:right;cursor:pointer;border:1px solid #444;background:#26262e;color:#ddd;border-radius:6px;padding:2px 10px}`;
  document.head.appendChild(st);
}

export async function showGuide(url) {
  styles();
  const back = document.createElement("div"); back.className = "bmd-back";
  back.innerHTML = `<div class="bmd"><button class="bmd-close">close</button><div class="bmd-body">Loading…</div></div>`;
  back.addEventListener("pointerdown", e => { if (e.target === back) back.remove(); });
  back.querySelector(".bmd-close").addEventListener("click", () => back.remove());
  back.addEventListener("keydown", e => e.stopPropagation());
  document.body.appendChild(back);
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    back.querySelector(".bmd-body").innerHTML = renderMarkdown(await r.text());
  } catch (e) {
    back.querySelector(".bmd-body").textContent = `Could not load the guide (${e.message}).`;
  }
}
