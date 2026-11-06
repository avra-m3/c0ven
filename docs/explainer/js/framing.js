// Framing for the floors seen from above: a fixed pitch, at whatever distance and screen offset fit an outline into the
// part of the screen the HUD leaves clear, beside the side panel or under it, whichever shows it larger. Also keeps a
// floor's name tags and reveal labels from covering one another.
import { camera, reduced, scene, toScreen } from "./world.js";

export function freeRects() {
  const W = innerWidth, H = innerHeight, narrow = W <= 720, g = 12;
  const top = document.querySelector(".topbar").getBoundingClientRect().bottom + 6;
  const sd = document.querySelector(".side").getBoundingClientRect();
  // A phone's reveal has its buttons in the panel, so nothing else sits at the bottom once the narrator is done.
  const revealing = narrow && !document.getElementById("reveal").hidden;
  const bottom = H - (revealing ? 40 : narrow ? 172 : 104);
  const under = { x: g, y: sd.bottom + 6, w: W - 2 * g, h: bottom - sd.bottom - 6 };
  return narrow ? [under] : [{ x: g, y: top, w: sd.left - 2 * g, h: bottom - top }, under];
}
const probe = new THREE.PerspectiveCamera(), pv = new THREE.Vector3();
function fitPoints({ pts, pitch }, fov, rect) {
  const W = innerWidth, H = innerHeight;
  probe.fov = fov; probe.aspect = W / H; probe.near = 0.1; probe.far = 300; probe.updateProjectionMatrix();
  const lo3 = pts[0].clone(), hi3 = pts[0].clone();
  for (const p of pts) { lo3.min(p); hi3.max(p); }
  const c = lo3.add(hi3).multiplyScalar(0.5), dir = new THREE.Vector3(0, Math.sin(pitch), Math.cos(pitch));
  const ext = (d) => {
    probe.position.copy(c).addScaledVector(dir, d); probe.lookAt(c); probe.updateMatrixWorld();
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) {
      pv.copy(p).project(probe);
      const x = (pv.x + 1) * W / 2, y = (1 - pv.y) * H / 2;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    return [x0, x1, y0, y1];
  };
  let lo = 4, hi = 200;
  const w = Math.max(rect.w, 40), h = Math.max(rect.h, 40);
  for (let k = 0; k < 26; k++) { const d = (lo + hi) / 2, e = ext(d); if (e[1] - e[0] <= w && e[3] - e[2] <= h) hi = d; else lo = d; }
  const e = ext(hi);
  return { c, dir, d: hi, off: [(e[0] + e[1]) / 2 - (rect.x + rect.w / 2), (e[2] + e[3]) / 2 - (rect.y + rect.h / 2)], scale: 1 / hi };
}
// The best of the clear rects for an outline {pts, pitch}. A larger scale shows it larger.
export const bestFit = (outline, fov, rects = freeRects()) => rects.map((r) => fitPoints(outline, fov, r)).reduce((a, b) => (b.scale > a.scale ? b : a));

// A floor's camera that follows outline(), easing to a new framing. Set snap to arrive at once, and key to "" to refit.
export function makeFramer(cam, fov, outline) {
  const F = { key: "", at: 0, d: 20, fog: 20, snap: true, pts: null, pos: new THREE.Vector3(), look: new THREE.Vector3(), off: [0, 0], tPos: new THREE.Vector3(), tLook: new THREE.Vector3(), tOff: [0, 0] };
  F.place = (dt) => {
    const now = performance.now();
    if (!F.key || now - F.at > 100) {
      F.at = now;
      const o = outline(), rects = freeRects(), key = innerWidth + "x" + innerHeight + rects.map((r) => [r.x, r.y, r.w, r.h].map(Math.round).join(",")).join("|");
      if (key !== F.key || o.pts !== F.pts) {
        F.key = key; F.pts = o.pts;
        const b = bestFit(o, fov, rects);
        F.tPos.copy(b.c).addScaledVector(b.dir, b.d); F.tLook.copy(b.c); F.tOff = b.off; F.d = b.d;
      }
    }
    const k = F.snap || reduced ? 1 : 1 - Math.exp(-dt * 4);
    F.snap = false;
    F.pos.lerp(F.tPos, k); F.look.lerp(F.tLook, k);
    F.off[0] += (F.tOff[0] - F.off[0]) * k; F.off[1] += (F.tOff[1] - F.off[1]) * k;
    camera.setViewOffset(innerWidth, innerHeight, F.off[0], F.off[1], innerWidth, innerHeight);
    cam.look.copy(F.look);
    F.fog += (F.d - F.fog) * k;
    scene.fog.near = F.fog * 1.05; scene.fog.far = F.fog * 4.2;
    return F.pos;
  };
  return F;
}

// Screen boxes {x (centre), y (top), w, h} nudged apart vertically; a few passes settle small clusters.
export function declutter(items, gap = 4) {
  for (let pass = 0; pass < 4; pass++) for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
    const a = items[i], b = items[j];
    if (Math.abs(a.x - b.x) * 2 >= a.w + b.w + gap) continue;
    const [t, u] = a.y <= b.y ? [a, b] : [b, a], over = t.y + t.h + gap - u.y;
    if (over > 0) { t.y -= over / 2; u.y += over / 2; }
  }
}
// Element sizes, read once after a change rather than every frame.
export const tagSize = new Map();
export const sizeOf = (el) => { let s = tagSize.get(el); if (!s) { s = [el.offsetWidth, el.offsetHeight]; tagSize.set(el, s); } return s; };

// Places name tags: items {el, x, w, h, y} kept apart and on the screen.
export function placeTags(items) {
  declutter(items);
  for (const it of items) it.el.style.transform = `translate(${Math.round(Math.min(innerWidth - it.w - 4, Math.max(4, it.x - it.w / 2)))}px, ${Math.round(it.y)}px)`;
}
// Places a floor's reveal labels, [element, point()], kept apart.
export function placeLabelsApart(anchors) {
  const labs = [];
  for (const [el, at] of anchors) {
    if (!el.classList.contains("on")) continue;
    const [x, y] = toScreen(at()), [w, h] = sizeOf(el);
    labs.push({ el, x, w, h: h + 6, y: y - h - 6 });
  }
  declutter(labs, 6);
  for (const it of labs) it.el.style.transform = `translate(${Math.round(Math.min(innerWidth - it.w / 2 - 6, Math.max(it.w / 2 + 6, it.x)))}px, ${Math.round(it.y + it.h)}px) translate(-50%, -100%)`;
}
