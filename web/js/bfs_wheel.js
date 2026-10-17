/**
 * The mouse wheel over a node's HTML panel zooms / pans the canvas like anywhere else on the graph, unless the pointer
 * is on something that scrolls that way and still can (a list, a text box, a tall panel): that one scrolls, and once
 * it reaches its end the wheel goes back to the canvas. Ctrl / Cmd + wheel always goes to the canvas.
 */
import { app } from "../../scripts/app.js";

function scrolls(start, stop, dx, dy) {
  for (let n = start; n && n.nodeType === 1; n = n.parentElement) {
    if (n.tagName === "SELECT" && (n.size > 1 || n.multiple)) return true;
    const cs = getComputedStyle(n);
    if (dy && /(auto|scroll)/.test(cs.overflowY) && n.scrollHeight > n.clientHeight + 1
      && (dy < 0 ? n.scrollTop > 0 : n.scrollTop + n.clientHeight < n.scrollHeight - 1)) return true;
    if (dx && /(auto|scroll)/.test(cs.overflowX) && n.scrollWidth > n.clientWidth + 1
      && (dx < 0 ? n.scrollLeft > 0 : n.scrollLeft + n.clientWidth < n.scrollWidth - 1)) return true;
    if (n === stop) break;
  }
  return false;
}

export function wheelToCanvas(el) {
  el.addEventListener("wheel", e => {
    const canvas = app.canvas?.canvas;
    if (!canvas || e.defaultPrevented || (!(e.ctrlKey || e.metaKey) && scrolls(e.target, el, e.deltaX, e.deltaY))) return;
    e.preventDefault();
    e.stopPropagation();
    canvas.dispatchEvent(new WheelEvent("wheel", {
      deltaX: e.deltaX, deltaY: e.deltaY, deltaZ: e.deltaZ, deltaMode: e.deltaMode, clientX: e.clientX, clientY: e.clientY,
      screenX: e.screenX, screenY: e.screenY, ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey, metaKey: e.metaKey,
      bubbles: true, cancelable: true,
    }));
  }, { passive: false });
}
