// Meridian Words: a narrated three.js game, drawn entirely by code, that explains why c0ven is fast. Each floor of an
// office is one of the engine's tricks. To add a floor: write js/floors/floorN.js (config, camera, enter, start, frame,
// as the others do), add narration/floorN.json and its audio, list it in FLOORS below, and in index.html add its label
// elements and update the count of open floors. Serve this folder over HTTP to run it; modules do not load from a file.
import floor1 from "./floors/floor1.js";
import floor2 from "./floors/floor2.js";
import floor3 from "./floors/floor3.js";
import { $, camera, reduced, renderer, resize, scene, tubes, view } from "./world.js";
import { loadNarration, skip, speaking, unlockAudio } from "./narration.js";
import { addFloor, floors, goFloor, placeLabels, setFloorHud } from "./hud.js";

const FLOORS = [floor1, floor2, floor3];
FLOORS.forEach(addFloor);
const narrationRead = loadNarration(FLOORS.map((F) => F.number));

// ---------- Frame loop ----------
const pointer = { x: 0, y: 0 };
addEventListener("pointermove", (e) => { pointer.x = e.clientX / innerWidth - 0.5; pointer.y = e.clientY / innerHeight - 0.5; });

function onResize() {
  resize();
  for (const F of FLOORS) if (F.resized) F.resized();
}
addEventListener("resize", onResize); onResize();

let lastNow = 0;
function frame(now) {
  const t = now / 1000, dt = Math.min(0.1, (now - lastNow) / 1000); lastNow = now;
  const F = floors[view.floor], C = F.camera;
  if (!C.offset && camera.view && camera.view.enabled) camera.clearViewOffset();
  const fov = C.fov || view.baseFov;
  if (camera.fov !== fov) { camera.fov = fov; camera.updateProjectionMatrix(); }
  const base = C.place(dt);
  const px = reduced ? 0 : pointer.x * 0.6 * C.par, py = reduced ? 0 : pointer.y * 0.3 * C.par;
  camera.position.set(base.x + px + (Math.random() - 0.5) * view.shake, base.y - py + (Math.random() - 0.5) * view.shake, base.z);
  camera.lookAt(C.look);
  view.shake *= 0.86;
  F.frame(t, dt);
  tubes.forEach((m, i) => { m.material.color.setScalar(((t * 0.37 + i * 0.31) % 7) < 0.06 ? 0.55 : 0.92); });
  renderer.render(scene, camera);
  if (F.afterRender) F.afterRender();
  placeLabels(F);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- Keys and start ----------
addEventListener("keydown", (e) => {
  if (!$("start").hidden) {
    if (e.key === "Enter") begin(1);
    else if (floors[e.key]) begin(+e.key);
    return;
  }
  if (e.key === " " && speaking()) { e.preventDefault(); skip(); return; }
  const n = parseInt(e.key, 10);
  if (n >= 1) {
    const b = $("choices").querySelectorAll("button")[n - 1];
    if (b) b.click();
  }
});
function begin(n) {
  $("start").hidden = true;
  unlockAudio();
  narrationRead.then(() => goFloor(n));
}
const linked = FLOORS.find((F) => location.hash === `#floor${F.number}`) || floor1;
for (const F of FLOORS) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "opt" + (F === linked ? " primary" : "");
  b.innerHTML = `Floor ${F.number} · ${F.name} <small>${F.minutes} min</small>`;
  b.addEventListener("click", () => begin(F.number));
  $("floors").appendChild(b);
}
setFloorHud(1);
