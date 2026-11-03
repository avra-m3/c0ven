// Floor 1, the shelf room, around x = 0. Reading the model's weights is the slow part, so move them closer.
import { $, box, camera, canvasTex, closeFloor, halo, haloTex, keyLight, mat, openFloor, pool, reduced, rep, scene, tube, tubes, up, view } from "../world.js";
import { backOut, ease, sleep, tween } from "../motion.js";
import { makeAdviser, makeCabinet, makeChair, makeClock, makeCoatStand, makeCooler, makeDesk, makeEnvelope, makePhone, makePlant, makeTypewriter, paperM, poseAdviser, sitAndThink, woodDark } from "../props.js";
import { carpetTex, ceilingTex, city, makeLift, plaster, skirtM, trimM, wallWithDoor } from "../tower.js";
import { speak } from "../narration.js";
import { addHistory, choose, clearStage, endings, findEnding, goFloor, labels, revealButtons, setClock } from "../hud.js";

const config = {
  number: 1, name: "The Shelf Room", minutes: 3, scale: 4.2, keys: ["Fetching", "Thinking"],
  labels: ["aAdv", "aShelf", "aBook", "aPath"],
  map: [
    ["Adviser", "the graphics card's maths units"],
    ["Shelf", "memory"],
    ["Manual", "the model's weights: billions of learned numbers"],
    ["The walk", "reading those numbers again, for every word"],
  ],
  lesson: "The maths is quick. The reading is slow. A faster AI engine mostly reads less, or reads from somewhere closer. c0ven is built around that problem, and each floor above is one of its tricks.",
  next: "Floor 2 · The Annex · open now.",
};

const RUNS = {
  built: { label: "As built", fetch: 3.4, think: 0.6 },
  faster: { label: "Faster thinker", fetch: 3.4, think: 0.3 },
  closer: { label: "Shelf moved", fetch: 0.8, think: 0.6 },
};

const opened = openFloor();
const tube2 = (x, y, z) => { tube(x, y, z); return tubes[tubes.length - 1]; };
// The room: carpet, a suspended ceiling, the lift at the left of the back wall and windows along the rest of it.
const ROOM = { x0: -5.0, x1: 9.4, back: -2.6, front: 17, h: 3.7 }, LIFT_X = -3.45, WIN = { x0: -1.5, x1: 7.5, y0: 1.0, y1: 3.2 };
const cityUpdate = city(scene, 2.2, 1, { near: -42 });
const lift = makeLift(scene, { x: LIFT_X, z: ROOM.back + 0.005, number: 1, name: "The Shelf Room" });
let clockHands = null, certificate = null, cooler = null;
const clock = { run: false, a: 0 };
(() => {
  const { x0, x1, back, front, h } = ROOM, w = x1 - x0, d = front - back, cx = (x0 + x1) / 2, cz = (back + front) / 2;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat(0xffffff, { map: rep(carpetTex, w, d), roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.position.set(cx, 0, cz); floor.receiveShadow = true; scene.add(floor);
  const ceil = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat(0xffffff, { map: rep(ceilingTex, w / 1.2, d / 1.2), emissive: 0xffffff, emissiveMap: rep(ceilingTex, w / 1.2, d / 1.2), emissiveIntensity: 0.3, roughness: 0.95 }));
  ceil.rotation.x = Math.PI / 2; ceil.position.set(cx, h, cz); scene.add(ceil);
  for (const [x, ry] of [[x0, Math.PI / 2], [x1, -Math.PI / 2]]) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(d, h), plaster); wall.rotation.y = ry; wall.position.set(x, h / 2, cz); wall.receiveShadow = true; scene.add(wall);
    box(0.04, 0.14, d, skirtM, x + Math.sign(cx - x) * 0.02, 0.07, cz);
    box(0.03, 0.05, d, trimM, x + Math.sign(cx - x) * 0.015, 1.05, cz);
  }
  // The back wall, around the lift and the window opening.
  wallWithDoor(scene, plaster, x0, WIN.x0, h, back, LIFT_X, 0.2);
  box(x1 - WIN.x1, h, 0.2, plaster, (x1 + WIN.x1) / 2, h / 2, back - 0.1);
  const wx = (WIN.x0 + WIN.x1) / 2, ww = WIN.x1 - WIN.x0;
  box(ww, WIN.y0, 0.2, plaster, wx, WIN.y0 / 2, back - 0.1);
  box(ww, h - WIN.y1, 0.2, plaster, wx, (h + WIN.y1) / 2, back - 0.1);
  for (const [a, b] of [[x0, LIFT_X - 0.77], [LIFT_X + 0.77, x1]]) {
    box(b - a, 0.14, 0.04, skirtM, (a + b) / 2, 0.07, back + 0.02);
    box(b - a, 0.05, 0.03, trimM, (a + b) / 2, 1.05, back + 0.015);
  }
  // Glass, piers, a mullion in each bay and a deep sill; radiators under it.
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(ww, WIN.y1 - WIN.y0), new THREE.MeshBasicMaterial({ color: 0x9fb4c8, transparent: true, opacity: 0.05, blending: THREE.AdditiveBlending, depthWrite: false }));
  glass.position.set(wx, (WIN.y0 + WIN.y1) / 2, back - 0.12); scene.add(glass);
  const bays = 4, bay = ww / bays;
  for (let i = 0; i <= bays; i++) box(0.24, WIN.y1 - WIN.y0, 0.26, mat(0x353f4f), WIN.x0 + i * bay, (WIN.y0 + WIN.y1) / 2, back - 0.08);
  for (let i = 0; i < bays; i++) {
    box(0.035, WIN.y1 - WIN.y0, 0.05, trimM, WIN.x0 + (i + 0.5) * bay, (WIN.y0 + WIN.y1) / 2, back - 0.13);
    box(bay - 0.5, 0.42, 0.14, mat(0x6c7078, { roughness: 0.5 }), WIN.x0 + (i + 0.5) * bay, 0.42, back + 0.09);
    for (let k = 0; k < 16; k++) box(0.015, 0.36, 0.01, mat(0x3c4047), WIN.x0 + (i + 0.5) * bay - (bay - 0.6) / 2 + k * (bay - 0.6) / 15, 0.42, back + 0.165);
  }
  box(ww + 0.2, 0.05, 0.3, mat(0x56606f, { roughness: 0.6 }), wx, WIN.y0 + 0.015, back + 0.02);
  // Blinds, half down over two bays: slats with gaps, and a cord.
  const slats = canvasTex(16, 256, (g, cw, ch) => {
    g.clearRect(0, 0, cw, ch);
    for (let y = 0; y < ch; y += 16) { g.fillStyle = "#a5a99f"; g.fillRect(0, y, cw, 10); g.fillStyle = "#6c7068"; g.fillRect(0, y + 10, cw, 2); }
  });
  slats.wrapS = slats.wrapT = THREE.RepeatWrapping;
  for (const [i, drop] of [[0, 1.05], [2, 0.75], [3, 1.45]]) {
    const t = slats.clone(); t.needsUpdate = true; t.repeat.set(1, drop * 3.4);
    const b = new THREE.Mesh(new THREE.PlaneGeometry(bay - 0.24, drop), new THREE.MeshStandardMaterial({ map: t, transparent: true, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.6 }));
    b.position.set(WIN.x0 + (i + 0.5) * bay, WIN.y1 - drop / 2, back + 0.0); scene.add(b);
    box(bay - 0.2, 0.05, 0.05, mat(0x8a8d86), WIN.x0 + (i + 0.5) * bay, WIN.y1 - 0.01, back + 0.01);
    box(0.008, drop * 0.9, 0.008, mat(0xd8d2b8), WIN.x0 + (i + 0.5) * bay + (bay - 0.3) / 2, WIN.y1 - drop * 0.5, back + 0.03);
  }
  // Fluorescent panels set in the ceiling; each is a real light and a pool on the carpet.
  for (const x of [-0.6, 3.2, 7]) {
    tube(x, h - 0.02, 0.4); tubes[tubes.length - 1].scale.set(1.25, 0.6, 3.2);
    halo(0xdfe9e4, 1.5, x, h - 0.1, 0.4, scene, 0.14);
    pool(scene, 0xc8daf0, 2.2, x, 0.012, 0.4, 0.075);
    box(2.44, 0.03, 0.74, trimM, x, h - 0.012, 0.4);
  }
  for (const x of [1.3, 5.4]) { const panel = tube2(x, h - 0.02, 4.4); pool(scene, 0xc8daf0, 2.0, x, 0.012, 4.4, 0.06); box(2.44, 0.03, 0.74, trimM, x, h - 0.012, 4.4); panel.scale.set(1.25, 0.6, 3.2); }
  keyLight(2.5, h - 0.2, 5, 2.5, Math.PI / 3.2, 18);

  // The night's empty desks at the back, the lamps off, the chairs in.
  for (const [x, z] of [[4.2, -1.45], [8.1, -1.0]]) {
    makeDesk(x, z, false);
    makeChair(x, z - 1.0);
    box(0.34, 0.14, 0.26, mat(0x3a3f48), x + 0.05, 1.09, z - 0.02);
  }
  // Cooler, plant, cabinets, a coat left behind, notices on the cork board, the clock over the window.
  cooler = makeCooler(scene, -1.2, -2.15);
  makePlant(scene, -4.45, -2.15, 1.15);
  makeCabinet(scene, -0.45, -2.25, 3); makeCabinet(scene, 8.95, -2.25, 4);
  makeCoatStand(scene, -4.5, -0.3);
  const cork = canvasTex(512, 340, (g, cw, ch) => {
    g.fillStyle = "#8a6a45"; g.fillRect(0, 0, cw, ch);
    for (let n = 0; n < 700; n++) { g.fillStyle = "rgba(60,42,24,0.35)"; g.fillRect((n * 83) % cw, (n * 131) % ch, 3, 3); }
    const note = (x, y, nw, nh, bg, rot, lines) => {
      g.save(); g.translate(x, y); g.rotate(rot);
      g.fillStyle = "rgba(0,0,0,0.25)"; g.fillRect(4, 5, nw, nh);
      g.fillStyle = bg; g.fillRect(0, 0, nw, nh);
      g.fillStyle = "#23272f"; g.font = "700 24px 'Barlow Condensed', Arial Narrow, sans-serif"; g.textAlign = "center";
      lines.forEach((l, i) => g.fillText(l, nw / 2, 42 + i * 26));
      g.fillStyle = "#c2453d"; g.beginPath(); g.arc(nw / 2, 10, 7, 0, 7); g.fill();
      g.restore();
    };
    note(26, 24, 190, 118, "#f2f2ec", 0.04, ["QUIET PLEASE", "ADVISERS", "THINKING"]);
    note(248, 18, 220, 100, "#f2e6c0", -0.05, ["WHO MOVED", "THE SHELF?"]);
    note(120, 180, 250, 130, "#e8eef2", 0.02, ["FIRE DRILL", "CANCELLED.", "AGAIN."]);
  });
  const board = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 0.73), new THREE.MeshStandardMaterial({ map: cork, roughness: 0.9 }));
  board.position.set(8.45, 1.95, ROOM.back + 0.01); scene.add(board);
  for (const [bw, bh, x, y] of [[1.2, 0.05, 8.45, 2.34], [1.2, 0.05, 8.45, 1.56], [0.05, 0.8, 7.87, 1.95], [0.05, 0.8, 9.03, 1.95]]) box(bw, bh, 0.05, woodDark, x, y, ROOM.back + 0.02);
  clockHands = makeClock(scene, 0.35, 3.45, ROOM.back + 0.01, 0.21);
  // Awarded to the adviser after the weekend course; the wall has room for it.
  const certTex = canvasTex(256, 180, (g, cw, ch) => {
    g.fillStyle = "#efe9d6"; g.fillRect(0, 0, cw, ch); g.strokeStyle = "#a9833f"; g.lineWidth = 6; g.strokeRect(8, 8, cw - 16, ch - 16);
    g.fillStyle = "#23272f"; g.textAlign = "center"; g.font = "700 24px 'Barlow Condensed', Arial Narrow, sans-serif";
    g.fillText("CERTIFICATE OF", cw / 2, 52); g.fillText("FASTER THINKING", cw / 2, 82); g.font = "500 20px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("(WEEKEND COURSE)", cw / 2, 120);
    g.fillStyle = "#c2453d"; g.beginPath(); g.arc(cw / 2, 148, 11, 0, 7); g.fill();
  });
  certificate = new THREE.Group(); certificate.position.set(-2.05, 1.72, ROOM.back + 0.01); scene.add(certificate);
  certificate.add(new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.44), new THREE.MeshStandardMaterial({ map: certTex, roughness: 0.5 })));
  box(0.68, 0.5, 0.012, woodDark, 0, 0, -0.008, certificate);
  certificate.visible = false;
})();
// A paler track worn in the carpet, desk to shelf: forty years of the same walk.
const trailTex = canvasTex(1024, 128, (g, w, h) => {
  g.clearRect(0, 0, w, h);
  const l = g.createLinearGradient(0, 0, 0, h); l.addColorStop(0, "rgba(173,186,205,0)"); l.addColorStop(0.5, "rgba(173,186,205,0.3)"); l.addColorStop(1, "rgba(173,186,205,0)");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
  for (let k = 0; k < 900; k++) { g.fillStyle = k % 2 ? "rgba(206,216,230,0.14)" : "rgba(20,24,30,0.1)"; g.fillRect((k * 89) % w, 34 + (k * 53) % 60, 5, 2); }
  g.fillStyle = "rgba(14,17,22,0.34)";
  for (let x = 30; x < w - 20; x += 64) { g.beginPath(); g.ellipse(x, x % 128 ? 50 : 78, 15, 7, 0, 0, 7); g.fill(); }
});
const trail = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 0.95), new THREE.MeshBasicMaterial({ map: trailTex, transparent: true, depthWrite: false }));
trail.rotation.x = -Math.PI / 2; trail.position.set(3.25, 0.012, 0.56); scene.add(trail);

makeDesk(0.35, 1.35);
// Just behind where the adviser stands, so sitting down lands on the seat.
makeChair(0.35, 0.15);
const typer = makeTypewriter(scene, 0.25, 1.02, 1.08, 0);
makePhone(1.06, 1.03, -0.5);
box(0.32, 0.04, 0.24, mat(0x3a3f48), -0.45, 1.04, 1.48).rotation.y = 0.12;
box(0.28, 0.018, 0.2, paperM, -0.45, 1.07, 1.48).rotation.y = 0.2;
// A name block facing the camera.
const nameTex = canvasTex(256, 64, (g, w, h) => { g.fillStyle = "#22262d"; g.fillRect(0, 0, w, h); g.fillStyle = "#d2ab62"; g.textAlign = "center"; g.textBaseline = "middle"; g.font = "600 36px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("SENIOR ADVISER", w / 2, h / 2 + 2); });
const nameBlock = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 0.105), new THREE.MeshStandardMaterial({ map: nameTex, roughness: 0.5 })); nameBlock.position.set(0.3, 1.1, 1.82); nameBlock.rotation.x = -0.25; scene.add(nameBlock);
pool(scene, 0xffb060, 0.5, -0.28, 1.024, 1.12, 0.32);
pool(scene, 0xffa040, 1.15, 0.1, 0.014, 1.6, 0.1);
// Steam off the mug.
const steam = reduced ? [] : [0, 1, 2].map((i) => {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color: 0xdfe3e8, transparent: true, opacity: 0, depthWrite: false }));
  s.position.set(1.13, 1.15, 1.05); s.userData.ph = i / 3; scene.add(s); return s;
});
// Dust hanging in the light by the shelf.
const motes = reduced ? [] : Array.from({ length: 9 }, (_, i) => {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color: 0xe4e4d8, transparent: true, opacity: 0.05, depthWrite: false }));
  s.scale.setScalar(0.05 + (i % 3) * 0.018); s.position.set(3.6 + (i * 0.71) % 3.6, 0.5 + (i * 0.53) % 2.6, -0.5 + (i * 0.37) % 1.2);
  s.userData.ph = i * 1.3; scene.add(s); return s;
});

// The shelf. shelf moves across the floor; shelfBody lifts and waddles on the legs that fold out from under it.
const FAR = 6.2, NEAR = 1.9;
const shelf = new THREE.Group(); scene.add(shelf);
const shelfBody = new THREE.Group(); shelf.add(shelfBody);
const shelfWood = mat(0x4a3d31);
box(1.3, 2.4, 0.08, shelfWood, 0, 1.2, -0.25, shelfBody);
box(0.08, 2.4, 0.55, shelfWood, -0.61, 1.2, 0, shelfBody);
box(0.08, 2.4, 0.55, shelfWood, 0.61, 1.2, 0, shelfBody);
box(1.4, 0.08, 0.64, shelfWood, 0, 2.44, 0, shelfBody);
box(1.36, 0.1, 0.6, mat(0x3a2f26), 0, 0.05, 0, shelfBody);
// Legs: a pivot under each corner whose y scale is the leg's length; the boot undoes that scale.
const shelfLegs = [];
for (const [x, z, phase] of [[-0.5, 0.17, 0], [0.5, -0.17, 0], [0.5, 0.17, Math.PI], [-0.5, -0.17, Math.PI]]) {
  const pivot = new THREE.Group(); pivot.position.set(x, 0.02, z); shelfBody.add(pivot);
  const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.03, 1, 10), mat(0x2b2620, { roughness: 0.7 }));
  leg.position.y = -0.5; leg.castShadow = true; pivot.add(leg);
  const foot = new THREE.Group(); foot.position.y = -1; pivot.add(foot);
  box(0.17, 0.07, 0.1, mat(0xb8893a, { metalness: 0.4, roughness: 0.4 }), 0.03, 0.01, 0, foot);
  pivot.visible = false;
  shelfLegs.push({ pivot, foot, phase });
}
// The puff of dust where the shelf lands.
const dust = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.7, 24), new THREE.MeshBasicMaterial({ color: 0xaab0b8, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
dust.rotation.x = -Math.PI / 2; dust.position.y = 0.03; scene.add(dust);
function setLegs(len) {
  const L = Math.max(len, 0.001);
  for (const l of shelfLegs) { l.pivot.visible = len > 0.01; l.pivot.scale.y = L; l.foot.scale.y = 1 / L; }
}
const spineCols = [0x5d6b7f, 0x7a6a55, 0x4f6a63, 0x6d7486, 0x8a5a48, 0x3f4e66];
let k = 0, shelfManual = null;
for (let s = 0; s < 5; s++) {
  const y = 0.06 + s * 0.56;
  box(1.2, 0.05, 0.52, shelfWood, 0, y, 0, shelfBody);
  if (s === 4) break;
  let x = -0.54;
  while (x < 0.5) {
    const w = 0.05 + ((k * 7) % 5) * 0.012, h = 0.34 + ((k * 5) % 7) * 0.025;
    if (x + w > 0.56) break;
    // The one manual the adviser fetches, in the desk book's colour; its gap shows while it is out.
    const manual = s === 1 && !shelfManual && x > -0.1;
    const b = box(w, h, 0.36, mat(manual ? 0x4f6a63 : spineCols[k % spineCols.length]), x + w / 2, y + 0.025 + h / 2, 0.02, shelfBody);
    if (manual) shelfManual = b;
    if (k % 9 === 4) b.rotation.z = 0.1;
    x += w + 0.008; k++;
  }
}
// The label, forty years of dust along the top, and a cobweb at the corner.
const shelfLabel = canvasTex(256, 64, (g, w, h) => { g.fillStyle = "#d9c9a8"; g.fillRect(0, 0, w, h); g.fillStyle = "#3a2a22"; g.textAlign = "center"; g.textBaseline = "middle"; g.font = "700 44px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("MANUALS", w / 2, h / 2 + 2); });
const shelfSign = new THREE.Mesh(new THREE.PlaneGeometry(0.72, 0.18), new THREE.MeshStandardMaterial({ map: shelfLabel, roughness: 0.7 })); shelfSign.position.set(0, 2.28, 0.32); shelfBody.add(shelfSign);
box(1.38, 0.012, 0.62, mat(0xa8a69c, { roughness: 1 }), 0, 2.487, 0, shelfBody);
const webTex = canvasTex(128, 128, (g, w) => {
  g.strokeStyle = "rgba(226,226,216,0.55)"; g.lineWidth = 1;
  for (let a = 0; a <= 6; a++) { g.beginPath(); g.moveTo(0, 0); g.lineTo(Math.cos(a * Math.PI / 12) * w, Math.sin(a * Math.PI / 12) * w); g.stroke(); }
  for (let r = 24; r < w; r += 24) { g.beginPath(); g.arc(0, 0, r, 0, Math.PI / 2); g.stroke(); }
});
const web = new THREE.Mesh(new THREE.PlaneGeometry(0.44, 0.44), new THREE.MeshBasicMaterial({ map: webTex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
web.position.set(-0.36, 2.12, 0.29); web.rotation.z = Math.PI / 2 * 3 + Math.PI / 2; shelfBody.add(web);
shelf.position.set(FAR, 0, 0);
shelf.rotation.y = -0.18;

const HOME = 0.35;
const A1 = makeAdviser({ x: HOME, z: 0.45 });
const adv = A1.group;
// A clerk's green eyeshade and sleeve garters, so this adviser is someone.
const visorM = new THREE.MeshStandardMaterial({ color: 0x2f7a4c, emissive: 0x1f5a36, emissiveIntensity: 0.6, transparent: true, opacity: 0.86, side: THREE.DoubleSide, roughness: 0.4 });
const visor = new THREE.Mesh(new THREE.CylinderGeometry(0.175, 0.2, 0.035, 22, 1, true, -1.25, 2.5), visorM); visor.position.set(0, 0.24, 0.02); A1.head.add(visor);
const peak = new THREE.Mesh(new THREE.CircleGeometry(0.2, 22, -0.3 + Math.PI / 2 - 1.0, 2.0), visorM); peak.rotation.x = -Math.PI / 2 + 0.18; peak.position.set(0, 0.225, 0.04); A1.head.add(peak);
for (const arm of [A1.armL, A1.armR]) { const g = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.062, 0.03, 12), mat(0xc2453d)); g.position.y = -0.2; arm.add(g); }
// The mortarboard from the weekend course.
const mortar = new THREE.Group(); mortar.position.y = 0.31; mortar.visible = false; A1.head.add(mortar);
const mortarBlack = mat(0x14161a, { roughness: 0.7 });
const skull = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.15, 0.08, 16), mortarBlack); mortar.add(skull);
const plate = box(0.42, 0.012, 0.42, mortarBlack, 0, 0.05, 0, mortar); plate.rotation.y = Math.PI / 4;
box(0.01, 0.14, 0.01, mat(0xe8a33d), 0.15, -0.02, 0.15, mortar);
// The manual, carried flat across both hands; rotated back out of the forearm's tilt in the carry pose.
const handBook = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.07, 0.24), mat(0x4f6a63));
handBook.position.set(-0.25, 0.02, -0.04); handBook.rotation.x = 1.95; handBook.visible = false; A1.armR.hand.add(handBook);

const E1 = makeEnvelope();
// The courier hands it over: it comes from the camera's side of the desk.
const ENV_IN = new THREE.Vector3(-0.42, 1.09, 1.48), ENV_OUT = new THREE.Vector3(0.9, 2.1, 10.5);
E1.group.position.copy(ENV_OUT); E1.group.visible = false;
const deskBook = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, 0.22), mat(0x4f6a63));
deskBook.position.set(0.75, 1.05, 1.05); deskBook.rotation.y = 0.3; deskBook.castShadow = true; deskBook.visible = false; scene.add(deskBook);

// The walk, drawn on the floor in the reveal
const pathGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(HOME, 0.02, 0.6), new THREE.Vector3(FAR - 0.8, 0.02, 0.6)]);
const pathLine = new THREE.Line(pathGeo, new THREE.LineDashedMaterial({ color: 0xe8a33d, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0 }));
pathLine.computeLineDistances(); scene.add(pathLine);
function setPath() {
  const p = pathGeo.attributes.position;
  p.setX(1, Math.max(HOME + 0.3, shelf.position.x - 0.8)); p.needsUpdate = true;
  pathLine.computeLineDistances();
}

const camera1 = { look: new THREE.Vector3(1.7, 1.42, 0), wide: new THREE.Vector3(1.7, 2.45, 10.2), narrow: new THREE.Vector3(1.1, 2.7, 13.5), fog: [24, 110], par: 1, offset: true };
// Off centre on wide screens, to clear the stopwatch.
camera1.place = () => {
  const tall = innerWidth / innerHeight < 0.9;
  camera1.fov = tall ? 72 : 44;
  if (tall) { if (camera.view && camera.view.enabled) camera.clearViewOffset(); }
  else camera.setViewOffset(innerWidth, innerHeight, innerWidth > 720 ? 70 : 0, 0, innerWidth, innerHeight);
  return tall ? camera1.narrow : camera1.wide;
};
const anchors = [
  [$("aAdv"), () => up(adv, 2.2)],
  [$("aShelf"), () => up(shelf, 2.65)],
  [$("aBook"), () => up(deskBook, 0.2)],
  [$("aPath"), () => new THREE.Vector3((HOME + shelf.position.x - 0.8) / 2, 0.1, 0.6)],
];

const puffs = [0, 1, 2, 3].map(() => { const p = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color: 0xc4c2b6, transparent: true, opacity: 0, depthWrite: false })); scene.add(p); return p; });
const squash = (u, kk) => shelfBody.scale.set(1 + 0.6 * kk * Math.sin(u * Math.PI), 1 - kk * Math.sin(u * Math.PI), 1 + 0.6 * kk * Math.sin(u * Math.PI));

// The shelf crouches, pops its legs out, looks both ways, waddles to `to` in `seconds`, and sets itself down.
async function moveShelf(to, seconds) {
  shelf.userData.moving = true;
  const LEG = 0.42, from = shelf.position.x, dir = Math.sign(to - from);
  const steps = Math.max(3, Math.round(Math.abs(to - from) / 0.55));
  // The boots are built with their toes toward +x; mirror them to point the way it walks.
  for (const l of shelfLegs) l.foot.scale.x = dir < 0 ? -1 : 1;
  await tween(0.25, (u) => squash(u, 0.06));
  await tween(0.45, (u) => { const e = backOut(u); setLegs(LEG * e); shelfBody.position.y = LEG * e; shelfBody.rotation.z = Math.sin(u * Math.PI * 3) * 0.03 * (1 - u); });
  await tween(0.3, (u) => { shelfBody.rotation.y = Math.sin(u * Math.PI * 2) * 0.14; shelfBody.rotation.z = -dir * 0.07 * Math.sin(u * Math.PI); });
  shelfBody.rotation.y = 0;
  await tween(seconds, (u) => {
    shelf.position.x = from + (to - from) * ease(u);
    const amp = Math.min(1, u * 6, (1 - u) * 6), ph = u * steps * Math.PI * 2;
    // A foot pushes back while planted and lifts as it swings forward; the body rides highest over upright legs.
    for (const l of shelfLegs) {
      const p = ph + l.phase, lift = Math.max(0, dir * Math.cos(p)) * amp;
      l.pivot.rotation.z = Math.sin(p) * 0.4 * amp;
      l.pivot.scale.y = LEG * (1 - 0.3 * lift); l.foot.scale.y = 1 / l.pivot.scale.y;
    }
    shelfBody.position.y = LEG + Math.abs(Math.cos(ph)) * 0.06 * amp;
    shelfBody.rotation.x = Math.sin(ph) * 0.06 * amp;
    shelfBody.rotation.z = (Math.sin(2 * ph) * 0.02 - dir * 0.03) * amp;
    setPath();
  });
  for (const l of shelfLegs) l.pivot.rotation.z = 0;
  shelfBody.rotation.set(0, 0, 0);
  await tween(0.35, (u) => { const e = u * u; setLegs(LEG * (1 - e)); shelfBody.position.y = LEG * (1 - e); });
  setLegs(0); shelfBody.position.y = 0;
  if (!reduced) view.shake = 0.12;
  dust.position.x = shelf.position.x;
  tween(0.5, (u) => { dust.scale.setScalar(1 + u * 2.2); dust.material.opacity = 0.45 * (1 - u); });
  // Forty years of dust leaves the top of it at once.
  puffs.forEach((p, i) => {
    const a = i * 1.57;
    p.position.set(shelf.position.x + Math.cos(a) * 0.4, 2.45, Math.sin(a) * 0.2);
    tween(0.9, (u) => { p.position.y = 2.45 + u * 0.4; p.position.x += Math.cos(a) * 0.006; p.scale.setScalar(0.5 + u * 1.6); p.material.opacity = 0.5 * (1 - u); });
  });
  await tween(0.3, (u) => squash(u, 0.07));
  shelfBody.scale.set(1, 1, 1);
  shelf.userData.moving = false;
}

// Floor 1: one word delivered and stamped. The clock counts only the adviser's fetching and thinking.
async function deliver(run) {
  setClock(0, 0);
  E1.stamp.visible = false; E1.stamp.scale.set(1, 1, 1); deskBook.visible = false; shelfManual.visible = true;
  E1.group.visible = true;
  await tween(0.7, (u) => {
    E1.group.position.lerpVectors(ENV_OUT, ENV_IN, ease(u));
    E1.group.rotation.z = (1 - ease(u)) * Math.sin(u * 13) * 0.2;
  });
  E1.group.rotation.z = 0;
  // The adviser stops typing and stands; the clock starts as they set off.
  A1.busy = false; A1.watch = E1.group.position;
  await tween(0.4, (u) => { A1.sit = 1 - ease(u); adv.position.z = 0.45 - 0.25 * A1.sit; });
  A1.sit = 0; A1.watch = null; clock.run = true;
  const target = shelf.position.x - 0.85, dist = target - HOME;
  let last = 0;
  A1.walking = 1;
  await tween(run.fetch, (u) => {
    const there = u < 0.5, w = there ? u * 2 : (u - 0.5) * 2;
    adv.position.x = there ? HOME + dist * w : target - dist * w;
    adv.position.z = 0.45 + 0.12 * Math.min(1, (there ? w : 1 - w) * 6);  // step out past the chair
    A1.heading = there ? Math.PI / 2 : -Math.PI / 2;
    A1.walkPhase += (u - last) * dist * 12; last = u;
    A1.reach = u > 0.4 && there;
    if (A1.reach) shelfManual.visible = false;
    handBook.visible = A1.carry = !there;
    setClock(run.fetch * u, 0);
  });
  A1.walking = 0;
  adv.position.x = HOME;
  handBook.visible = A1.carry = false; deskBook.visible = true;
  await sitAndThink(A1, run.fetch, run.think);
  clock.run = false;
  E1.stamp.visible = true;
  if (!reduced) tween(0.25, (u) => { const k2 = 1 + 0.6 * Math.sin(u * Math.PI); E1.stamp.scale.set(k2, 1, k2); });
  addHistory(run);
  await sleep(0.4);
  // Back to the typewriter, the book returned to its shelf between runs.
  await tween(0.45, (u) => { A1.sit = ease(u); adv.position.z = 0.45 - 0.25 * A1.sit; });
  A1.busy = true;
}

function resetFloor1() {
  shelf.position.x = FAR; setPath();
  adv.position.set(HOME, 0, 0.2); adv.rotation.y = A1.heading = 0; A1.sit = 1; A1.reach = false; A1.busy = true; A1.watch = null;
  E1.group.position.copy(ENV_OUT); E1.group.visible = false; E1.group.rotation.z = 0; E1.stamp.visible = false; deskBook.visible = false;
  shelfManual.visible = true; mortar.visible = false; certificate.visible = false; clock.run = false;
}
async function intro() {
  await speak("intro");
  choose([["Thinking about the word", () => watch("think")], ["Fetching the manual", () => watch("fetch")]]);
}
async function watch(b) {
  await speak("watch");
  await deliver(RUNS.built);
  await speak(b === "fetch" ? "right" : "wrong");
  choose([["Train a faster thinker", faster], ["Move the shelf to the desk", closer]]);
}
async function faster() {
  clearStage();
  if (Math.abs(shelf.position.x - FAR) > 0.01) await moveShelf(FAR, 2.2);
  mortar.visible = certificate.visible = true;
  await speak("faster1");
  await deliver(RUNS.faster);
  findEnding("thinking", "The Thinking Ending");
  await speak("faster2");
  if (endings[1].has("shelf")) choose([["See what it means", reveal]]);
  else choose([["Try the other door", closer]]);
}
async function closer() {
  clearStage();
  mortar.visible = false;
  await Promise.all([speak("closer1"), moveShelf(NEAR, 4.4)]);
  await deliver(RUNS.closer);
  findEnding("shelf", "The Short Walk");
  await speak("closer2");
  choose([["What just happened?", reveal]]);
}
async function reveal() {
  labels(true);
  const more = !endings[1].has("thinking");
  $("nextNote").textContent = config.next + (more ? " One ending on this floor is still hidden." : " Both endings found.");
  revealButtons([]);
  await speak("reveal", (line) => { if (line === "reveal2") $("reveal").hidden = false; });
  $("reveal").hidden = false;
  $("caption").hidden = true;
  revealButtons([
    ["Take the lift to floor 2", () => goFloor(2), true],
    ...(more ? [["Try the other door", faster]] : []),
    ["Replay floor 1", () => { endings[1].clear(); goFloor(1); }],
  ]);
}

setPath();
resetFloor1();
closeFloor(1, opened);

// The clock keeps real time, and races only while the stopwatch runs.
let clockA = 0;
export default {
  ...config, anchors, camera: camera1, lift, enter: resetFloor1, start: intro,
  frame(t, dt) {
    poseAdviser(A1, t, dt);
    cityUpdate(t);
    clockA += dt * (clock.run ? 2.2 : 0.01);
    clockHands.minute.rotation.z = -1.05 - clockA; clockHands.hour.rotation.z = -2.15 - clockA / 12;
    typer.carriage.position.x = A1.busy ? 0.1 * ((t * 0.5 % 1) - 0.5) : 0;
    if (reduced) return;
    if (!shelf.userData.moving) shelfBody.position.y = (1 + Math.sin(t * 1.3)) * 0.006;
    for (const s of steam) {
      const u = (t * 0.28 + s.userData.ph) % 1;
      s.position.y = 1.16 + u * 0.4; s.position.x = 1.13 + Math.sin(t * 2 + u * 5) * 0.025; s.scale.setScalar(0.04 + u * 0.1); s.material.opacity = 0.13 * Math.sin(u * Math.PI);
    }
    for (const m of motes) {
      m.position.y += dt * 0.025; m.position.x += Math.sin(t * 0.5 + m.userData.ph) * dt * 0.02;
      if (m.position.y > 3.1) m.position.y = 0.4;
      m.material.opacity = 0.05 + 0.04 * Math.sin(t * 0.8 + m.userData.ph);
    }
    const b = (t * 0.17) % 1;
    cooler.bubble.position.y = 1.02 + Math.min(1, b * 8) * 0.38; cooler.bubble.material.opacity = b < 0.125 ? 0.7 : 0;
  },
  onLabels(on) { pathLine.material.opacity = on ? 0.95 : 0; },
};
