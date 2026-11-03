// The lift: one car, moved behind a floor's doors while they are shut, and the ride that carries the camera from one
// floor to another with no cut. While view.rig is set, main.js uses its position, look, field of view and off (how
// much of the floor's screen offset to keep, 0 to 1) instead of the floor's camera.
import { canvasTex, scene, view } from "./world.js";
import { lerp } from "./motion.js";
import { CAR, dialAngle, dialMat, leafIn, shaftTex } from "./tower.js";
import { ding, doorSound, setHum, setMusic } from "./sound.js";

const { w: W, d: D, h: H, doorW, doorH, eye } = CAR;
const car = new THREE.Group(); scene.add(car);
const FRONT = -0.23;   // the car's door wall, behind the thickest wall a floor's doors stand in

// All of the car is unlit and painted with its light, so it looks the same on every floor.
const basic = (map, o = {}) => new THREE.MeshBasicMaterial(Object.assign({ map, fog: false, side: THREE.DoubleSide }, o));
function panel(w, h, m, x, y, z, ry = 0, rx = 0) {
  const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), m);
  p.position.set(x, y, z); p.rotation.set(rx, ry, 0); car.add(p); return p;
}
const walnut = canvasTex(512, 512, (g, w, h) => {
  g.fillStyle = "#4b2f1e"; g.fillRect(0, 0, w, h);
  for (let c = 0; c < 4; c++) {
    g.fillStyle = ["#533522", "#4a2e1d", "#573824", "#4e3120"][c]; g.fillRect(c * 128, 0, 128, h);
    g.strokeStyle = "rgba(25,12,5,0.3)"; g.lineWidth = 1;
    for (let k = 0; k < 16; k++) { const x = c * 128 + 6 + k * 8; g.beginPath(); g.moveTo(x, 0); g.bezierCurveTo(x + 7, h * 0.3, x - 7, h * 0.65, x + 2, h); g.stroke(); }
    g.fillStyle = "rgba(15,8,3,0.7)"; g.fillRect(c * 128, 0, 2, h);
  }
  const l = g.createLinearGradient(0, 0, 0, h);
  l.addColorStop(0, "rgba(255,220,170,0.34)"); l.addColorStop(0.5, "rgba(255,220,170,0.06)"); l.addColorStop(1, "rgba(0,0,0,0.5)");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
});
const rugTex = canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = "#3b1d20"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#a9833f"; g.lineWidth = 5; g.strokeRect(12, 12, w - 24, h - 24);
  g.fillStyle = "#59302d";
  for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) { g.save(); g.translate(40 + x * 44, 40 + y * 44); g.rotate(Math.PI / 4); g.fillRect(-9, -9, 18, 18); g.restore(); }
  const r = g.createRadialGradient(w / 2, h / 2, 10, w / 2, h / 2, 180);
  r.addColorStop(0, "rgba(255,215,160,0.2)"); r.addColorStop(1, "rgba(0,0,0,0.42)");
  g.fillStyle = r; g.fillRect(0, 0, w, h);
});
const ceilTex = canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = "#6c6352"; g.fillRect(0, 0, w, h);
  g.fillStyle = "#fff0d0"; g.fillRect(30, 30, w - 60, h - 60);
  g.fillStyle = "rgba(90,74,50,0.85)";
  for (let k = 1; k < 5; k++) { g.fillRect(30 + k * (w - 60) / 5, 30, 3, h - 60); g.fillRect(30, 30 + k * (h - 60) / 5, w - 60, 3); }
});
// The back wall: polished steel, a brass rail, and the month's best employee.
const backTex = canvasTex(512, 668, (g, w, h) => {
  const l = g.createLinearGradient(0, 0, w, 0);
  l.addColorStop(0, "#54575c"); l.addColorStop(0.5, "#9a9b98"); l.addColorStop(1, "#4d5055");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
  for (let k = 0; k < 600; k++) { g.fillStyle = "rgba(255,255,255,0.03)"; g.fillRect(0, (k * 37) % h, w, 1); }
  const fx = 166, fy = 110, fw = 180, fh = 240;
  g.fillStyle = "#2b1a10"; g.fillRect(fx - 10, fy - 10, fw + 20, fh + 20);
  g.fillStyle = "#e7e2d2"; g.fillRect(fx, fy, fw, fh);
  g.fillStyle = "#1a1f27"; g.textAlign = "center";
  g.font = "700 18px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("EMPLOYEE OF THE MONTH", fx + fw / 2, fy + 28);
  // A portrait of the shelf, with its legs out.
  g.fillStyle = "#4a3d31"; g.fillRect(fx + 54, fy + 46, 72, 116);
  const cols = ["#5d6b7f", "#7a6a55", "#4f6a63", "#8a5a48"];
  for (let r = 0; r < 4; r++) { g.fillStyle = "#2b2620"; g.fillRect(fx + 60, fy + 52 + r * 27, 60, 22); for (let b = 0; b < 6; b++) { g.fillStyle = cols[(r + b) % 4]; g.fillRect(fx + 62 + b * 9.6, fy + 55 + r * 27, 8, 19); } }
  g.fillStyle = "#2b2620"; g.fillRect(fx + 62, fy + 162, 4, 28); g.fillRect(fx + 114, fy + 162, 4, 28);
  g.fillStyle = "#b8893a"; g.fillRect(fx + 56, fy + 188, 16, 6); g.fillRect(fx + 108, fy + 188, 16, 6);
  g.fillStyle = "#1a1f27"; g.font = "600 17px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("(NO REASON GIVEN)", fx + fw / 2, fy + 222);
});
const jambTex = canvasTex(128, 512, (g, w, h) => {
  g.fillStyle = "#7f7b70"; g.fillRect(0, 0, w, h);
  for (let k = 0; k < 200; k++) { g.fillStyle = k % 2 ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)"; g.fillRect((k * 37) % w, 0, 1, h); }
  const l = g.createLinearGradient(0, 0, 0, h); l.addColorStop(0, "rgba(255,226,180,0.3)"); l.addColorStop(1, "rgba(0,0,0,0.35)");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
});
for (const sx of [-1, 1]) {
  panel(D, H, basic(walnut), sx * W / 2, H / 2, FRONT - D / 2, sx * Math.PI / 2);
  // A brass rail at hand height on each side.
  const rail = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, D - 0.2), basic(null, { color: 0xc9a25e })); rail.position.set(sx * (W / 2 - 0.05), 0.95, FRONT - D / 2); car.add(rail);
  // The door wall's jamb, either side of the opening.
  panel((W - doorW) / 2, H, basic(jambTex), sx * (doorW / 2 + (W - doorW) / 4), H / 2, FRONT, Math.PI);
}
panel(W, D, basic(rugTex), 0, 0.002, FRONT - D / 2, 0, -Math.PI / 2);
panel(W, D, basic(ceilTex), 0, H, FRONT - D / 2, 0, Math.PI / 2);
panel(W, H, basic(backTex), 0, H / 2, FRONT - D, 0, 0);
const backRail = new THREE.Mesh(new THREE.BoxGeometry(W - 0.2, 0.03, 0.03), basic(null, { color: 0xc9a25e })); backRail.position.set(0, 0.95, FRONT - D + 0.05); car.add(backRail);
panel(doorW + 0.02, H - doorH, leafIn, 0, (H + doorH) / 2, FRONT, Math.PI);
// The opening's lining, from the car's door wall out to the floor's doors, so the floor's wall does not show around it.
const LINING = -FRONT - 0.09;
for (const sx of [-1, 1]) panel(LINING, doorH, basic(jambTex), sx * doorW / 2, doorH / 2, FRONT + LINING / 2, sx * Math.PI / 2);
panel(doorW, LINING, basic(jambTex), 0, doorH, FRONT + LINING / 2, 0, Math.PI / 2);
panel(doorW, LINING, basic(null, { color: 0xb08a4a }), 0, 0.003, FRONT + LINING / 2, 0, -Math.PI / 2);

// Above the doors, inside: the dial, facing the rider.
const dial = panel(0.56, 0.28, dialMat(), 0, doorH + 0.17, FRONT - 0.005, Math.PI);
const needlePivot = new THREE.Group(); needlePivot.position.set(0, doorH + 0.035, FRONT - 0.01); car.add(needlePivot);
const needle = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.22, 0.004), basic(null, { color: 0xe8a33d })); needle.position.y = 0.11; needlePivot.add(needle);
dial.renderOrder = 1;

// The button plate, on the rider's right. Floors above the open ones are taped over. Redrawn for each ride.
const plateC = document.createElement("canvas"); plateC.width = 180; plateC.height = 640;
const plateTex = new THREE.CanvasTexture(plateC); plateTex.encoding = THREE.sRGBEncoding;
function drawPlate(level, dest) {
  const g = plateC.getContext("2d"), w = plateC.width, h = plateC.height;
  g.fillStyle = "#a8864c"; g.fillRect(0, 0, w, h);
  g.fillStyle = "rgba(255,255,255,0.08)"; g.fillRect(0, 0, 6, h);
  g.fillStyle = "#0e1014"; g.fillRect(24, 22, w - 48, 84);
  g.fillStyle = "#ffb347"; g.textAlign = "center"; g.textBaseline = "middle"; g.font = "700 70px 'IBM Plex Mono', Menlo, monospace";
  g.fillText(level === 0 ? "G" : String(level).padStart(2, "0"), w / 2, 66);
  g.fillStyle = "#1c1810"; g.font = "600 17px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("MERIDIAN WORDS LTD.", w / 2, 128);
  for (let r = 0; r < 7; r++) for (let c = 0; c < 4; c++) {
    g.fillStyle = "#3a3226"; g.beginPath(); g.arc(36 + c * 36, 160 + r * 32, 11, 0, 7); g.fill();
  }
  g.save(); g.translate(w / 2, 256); g.rotate(-0.1); g.fillStyle = "#e0c94f"; g.fillRect(-92, -24, 184, 48);
  g.fillStyle = "#1a1a12"; g.font = "700 20px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("4 TO 48: OUT OF ORDER", 0, 2); g.restore();
  ["3", "2", "1", "G"].forEach((s, i) => {
    const y = 412 + i * 54, lv = s === "G" ? 0 : +s, lit = lv === dest;
    g.fillStyle = lit ? "#ffb347" : "#2a2519"; g.beginPath(); g.arc(w / 2, y, 21, 0, 7); g.fill();
    g.fillStyle = lit ? "#2a1a05" : "#d9cfb2"; g.font = "700 26px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText(s, w / 2, y + 1);
  });
  plateTex.needsUpdate = true;
}
panel(0.2, 0.71, basic(plateTex), -(doorW / 2 + 0.17), 1.28, FRONT - 0.006, Math.PI);
// The notices on the rider's left.
const noticeTex = canvasTex(180, 256, (g, w, h) => {
  g.fillStyle = "#e4dfcf"; g.fillRect(0, 0, w, h);
  g.fillStyle = "#1a1f27"; g.textAlign = "center"; g.font = "700 24px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("LIFT 1", w / 2, 34);
  g.fillRect(24, 44, w - 48, 2);
  g.font = "500 19px 'Barlow Condensed', Arial Narrow, sans-serif";
  ["CAPACITY: ONE COURIER", "", "IN CASE OF FIRE,", "KEEP WRITING", "", "LAST INSPECTED:", "BY MANAGEMENT", "(NOTED)"].forEach((s, i) => g.fillText(s, w / 2, 76 + i * 22));
});
panel(0.2, 0.28, basic(noticeTex), doorW / 2 + 0.17, 1.52, FRONT - 0.006, Math.PI);

// A passing landing's light, sweeping down the doors and side walls.
const bandTex = canvasTex(8, 128, (g, w, h) => {
  const l = g.createLinearGradient(0, 0, 0, h);
  l.addColorStop(0, "rgba(255,226,170,0)"); l.addColorStop(0.5, "rgba(255,226,170,0.75)"); l.addColorStop(1, "rgba(255,226,170,0)");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
});
const bandM = basic(bandTex, { transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
const bands = [panel(doorW, 0.7, bandM, 0, 0, -0.08, Math.PI), panel(D, 0.7, bandM, W / 2 - 0.005, 0, FRONT - D / 2, -Math.PI / 2), panel(D, 0.7, bandM, -W / 2 + 0.005, 0, FRONT - D / 2, Math.PI / 2)];

const setLevel = (lv) => { needlePivot.rotation.z = dialAngle(lv); };
const smooth = (u) => u * u * (3 - 2 * u);
// A wide enough view from the back of the car to show the doors whole, at any window shape.
const carFov = () => Math.max(64, 2 * Math.atan(0.86 / (D - 0.28 + 0.1) / (innerWidth / innerHeight)) * 180 / Math.PI);

let speed = 1, riding = false;
export const isRiding = () => riding;
export function skipRide() { if (riding) speed = 6; }
// A tween that a skip speeds up.
function rt(seconds, fn) {
  return new Promise((done) => {
    let u = 0, last = performance.now();
    const tick = (now) => { u = Math.min(1, u + (now - last) / 1000 * speed / seconds); last = now; fn(u); if (u < 1) requestAnimationFrame(tick); else done(); };
    requestAnimationFrame(tick);
  });
}
const wait = (s) => rt(s, () => {});
const doors = (L, from, to, seconds) => { doorSound(seconds / speed, to > from); return rt(seconds, (u) => L.open(lerp(from, to, smooth(u)))); };
const V = (x, y, z) => new THREE.Vector3(x, y, z);

// Takes the camera from floor `from` to floor `to`, on foot through the doors. enter() switches the game to the new
// floor; it runs while the car's doors are shut. Each floor's `lift` is its doors; it may add `route` (points to pass
// on the way to them), `walk` (seconds of that walk), `doorAt` (seconds before they open) and `onWalk(u)`.
export async function ride(from, to, enter) {
  riding = true; speed = 1;
  document.body.classList.add("riding");
  const A = from.lift, B = to.lift, r0 = from.number, r1 = to.number;
  let door = A.world();
  car.position.copy(door);
  drawPlate(r0, r1); setLevel(r0);
  const rig = view.rig = { pos: from.camera.place(0).clone(), look: from.camera.look.clone(), fov: from.camera.fov || view.baseFov, off: 1, base: null, baseFov: 42 };

  // Walk in. The doors open as the walk begins, with a bell, and the car's music grows clearer on the way in.
  const down = r1 < r0, walk = A.walk || 2.0, doorAt = A.doorAt === undefined ? 0.1 : A.doorAt;
  let heard = false;
  const route = new THREE.CatmullRomCurve3([rig.pos.clone(), ...(A.route || []), V(door.x, door.y + 1.72, door.z + 2.9), V(door.x, door.y + eye, door.z + 0.1), V(door.x, door.y + eye, door.z - 0.95)], false, "centripetal");
  const look0 = rig.look.clone(), lookIn = V(door.x, door.y + eye + 0.02, door.z - 5), fov0 = rig.fov, fov1 = carFov();
  A.call(true);
  const opening = wait(doorAt).then(() => { ding(down); heard = true; return doors(A, 0, 1, 0.75); });
  await rt(walk, (u) => {
    if (heard) setMusic(0.25 + 0.75 * smooth(u));
    rig.pos.copy(route.getPoint(smooth(u)));
    rig.look.lerpVectors(look0, lookIn, smooth(Math.min(1, u * 1.5)));
    rig.fov = lerp(fov0, fov1, smooth(u));
    rig.off = 1 - smooth(Math.min(1, u * 1.4));
    if (A.onWalk) A.onWalk(u);
  });
  await opening;
  setMusic(1);
  A.call(false);

  // Turn to face the doors, as anyone does, stepping to the back as they shut.
  const inside = rig.pos.clone(), rear = V(door.x, door.y + eye, door.z + FRONT - D + 0.28);
  const turning = rt(0.85, (u) => {
    const e = smooth(u), a = Math.PI * e;
    rig.pos.lerpVectors(inside, rear, e);
    rig.look.set(rig.pos.x - Math.sin(a) * 3, door.y + eye + 0.04 * e, rig.pos.z - Math.cos(a) * 3);
  });
  await wait(0.25);
  await doors(A, 1, 0, 0.7);
  await turning;

  // The shut doors hide the switch: the car and camera move to the new floor's doors.
  enter();
  const next = B.world(), shift = next.clone().sub(door);
  car.position.copy(next); rig.pos.add(shift); rig.look.add(shift); door = next;
  B.open(0);
  await wait(0.1);

  // The ride. The dial, the display, the shaft in the door windows and the light of passing landings all follow one level.
  const floors = Math.abs(r1 - r0), dir = Math.sign(r1 - r0), pos0 = rig.pos.clone(), rideTime = Math.min(5, 2.3 + floors * 0.65);
  const startShaft = shaftTex.offset.y;
  let shown = r0;
  await rt(rideTime, (u) => {
    const p = smooth(u), lv = lerp(r0, r1, p), pace = Math.sin(u * Math.PI);
    setLevel(lv); B.show(lv);
    setHum(pace);
    const n = Math.round(lv);
    if (n !== shown) { shown = n; drawPlate(n, r1); }
    shaftTex.offset.y = startShaft + dir * p * Math.max(1, floors) * 1.5;
    // Two sweeps a storey; each band runs down the walls as the car rises and up as it falls.
    const f = (lv * dir * 2 + 100) % 1;
    for (const b of bands) b.position.y = lerp(H + 0.4, -0.4, f);
    bandM.opacity = pace * Math.sin(f * Math.PI) * 0.8;
    rig.pos.y = pos0.y + (Math.sin(performance.now() * 0.05) * 0.0035 + Math.sin(performance.now() * 0.031) * 0.0025) * pace - dir * 0.03 * Math.sin(u * Math.PI * 2) * (1 - pace);
    rig.look.y = door.y + eye + 0.04;
  });
  bandM.opacity = 0; setHum(0); setLevel(r1); B.show(r1);
  // The car settles with a small dip, then the bell, which is let ring before the doors move.
  await rt(0.28, (u) => { rig.pos.y = pos0.y - 0.018 * dir * Math.sin(u * Math.PI); });
  ding(down);
  await wait(down ? 0.8 : 0.35);

  // Doors open; walk out to where the floor's camera stands, leaving the music behind.
  const opened = doors(B, 0, 1, 0.75);
  await wait(0.3);
  const start = rig.pos.clone(), lookOut = V(door.x, door.y + eye, door.z + 6), end0 = rig.base.clone();
  const out = new THREE.CatmullRomCurve3([start, V(door.x, door.y + eye, door.z + 0.15), V(door.x, door.y + 1.75, door.z + 2.7), end0], false, "centripetal");
  const seen = { hud: false };
  const shutting = (async () => { await opened; await wait(0.9); await doors(B, 1, 0, 0.7); })();
  await rt(A === B ? 0.1 : 1.75, (u) => {
    const e = smooth(u);
    setMusic(1 - e * 0.98);
    rig.pos.copy(out.getPoint(e)).addScaledVector(rig.base.clone().sub(end0), e);
    rig.look.lerpVectors(lookOut, to.camera.look, smooth(Math.min(1, u * 1.15)));
    rig.fov = lerp(fov1, rig.baseFov, e);
    rig.off = e;
    if (u > 0.55 && !seen.hud) { seen.hud = true; document.body.classList.remove("riding"); }
  });
  setMusic(0);
  view.rig = null;
  riding = false;
  shutting.then(() => {});
}

// Puts the car behind a floor's doors with no ride, for the start of the game and for reduced motion.
export function parkCar(F) { car.position.copy(F.lift.world()); setLevel(F.number); drawPlate(F.number, F.number); F.lift.open(0); }
