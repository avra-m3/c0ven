// Floor 1, the shelf room, around x = 0. Reading the model's weights is the slow part, so move them closer.
import { $, box, canvasTex, halo, haloTex, keyLight, mat, pool, reduced, scene, sign, tube, up, view, windowsTex } from "../world.js";
import { backOut, ease, sleep, tween } from "../motion.js";
import { makeAdviser, makeChair, makeDesk, makeEnvelope, paperM, poseAdviser, sitAndThink, wood, woodDark } from "../props.js";
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

// Hands of the wall clock; they turn only while the stopwatch runs.
const clock = { run: false, a: 0, h1: null, h2: null };
// The tubes' point lights; the middle one flickers.
const tubeLights = [];

// Tiled surfaces, 256 pixels to a unit, as on floor 3.
function tile(draw) { const t = canvasTex(256, 256, draw); t.wrapS = t.wrapT = THREE.RepeatWrapping; return t; }
const rep = (t, x, y) => { const c = t.clone(); c.needsUpdate = true; c.repeat.set(x, y); return c; };
const carpetTex = tile((g) => {
  for (let i = 0; i < 4; i++) { g.fillStyle = i === 1 || i === 2 ? "#343d4b" : "#2c333f"; g.fillRect((i % 2) * 128, (i >> 1) * 128, 128, 128); }
  for (let k = 0; k < 1500; k++) { g.fillStyle = k % 3 ? "rgba(82,94,114,0.45)" : "rgba(14,18,24,0.45)"; g.fillRect((k * 67) % 256, (k * 113 + (k >> 2)) % 256, 2, 2); }
  g.fillStyle = "rgba(11,14,19,0.55)"; g.fillRect(0, 0, 256, 2); g.fillRect(0, 128, 256, 2); g.fillRect(0, 0, 2, 256); g.fillRect(128, 0, 2, 256);
});

(() => {
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 14), mat(0xffffff, { map: rep(carpetTex, 30, 14), roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  // The room's shell: side walls and a low ceiling band, so it reads as a box and not a backdrop.
  const plaster = mat(0x323b4b), plasterSide = mat(0x2d3646);
  for (const sx of [-4.6, 10.9]) {
    const w = new THREE.Mesh(new THREE.PlaneGeometry(6.6, 6), plasterSide);
    w.rotation.y = sx < 0 ? Math.PI / 2 : -Math.PI / 2;
    w.position.set(sx, 3, 0.7); w.receiveShadow = true; scene.add(w);
    box(0.06, 0.12, 6.6, mat(0x262c37), sx, 0.06, 0.7);
  }
  const ceil = new THREE.Mesh(new THREE.PlaneGeometry(15.5, 6.6), mat(0x262c37, { roughness: 1 }));
  ceil.rotation.x = Math.PI / 2; ceil.position.set(3.15, 5.0, 0.7); scene.add(ceil);
  // The back wall, built in panels around the window opening at x −4.3…8.7, y 2.05…3.68.
  for (const [w, h, x, y] of [[30, 2.05, 0, 1.025], [30, 2.32, 0, 4.84], [10.7, 1.63, -9.65, 2.865], [6.3, 1.63, 11.85, 2.865]]) {
    const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), plaster);
    p.position.set(x, y, -2.6); p.receiveShadow = true; scene.add(p);
  }
  // Evening glass in the opening; the lit skyline sits eight to twelve units beyond it.
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(13, 1.63), new THREE.MeshBasicMaterial({ color: 0x141a24, transparent: true, opacity: 0.45 }));
  glass.position.set(2.2, 2.865, -2.72); scene.add(glass);
  // Skyline windows on a tileable grid, repeated at the size of each building face.
  const skyTile = canvasTex(128, 128, (g) => {
    g.fillStyle = "#0b0f16"; g.fillRect(0, 0, 128, 128);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const i = r * 4 + c;
      if ((i * 7) % 5 < 2) { g.fillStyle = i % 3 ? "#d8b46a" : "#9fb5c9"; g.globalAlpha = 0.3 + ((i * 13) % 50) / 100; g.fillRect(c * 32 + 9, r * 32 + 8, 14, 10); }
    }
    g.globalAlpha = 1;
  });
  skyTile.wrapS = skyTile.wrapT = THREE.RepeatWrapping;
  for (const [x, h, w, z] of [[-7, 5, 3.4, -7.5], [-2.5, 7, 3.2, -9.5], [2, 4.5, 3, -8], [6.5, 6.5, 3.4, -10], [11, 5, 3, -8.5], [15, 7, 3.2, -9.5]]) {
    const t = skyTile.clone(); t.needsUpdate = true; t.repeat.set(w / 1.6, h / 1.6);
    const m = new THREE.MeshBasicMaterial({ map: t });
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, 2), [mat(0x161a21), mat(0x161a21), mat(0x161a21), mat(0x161a21), m, mat(0x161a21)]);
    b.position.set(x, h / 2, z); scene.add(b);
  }
  box(30, 0.12, 0.06, mat(0x262c37), 0, 0.06, -2.56);
  // Chair rail, stopped at the window jambs like a real one.
  for (const [w, x] of [[10.7, -9.65], [6.3, 11.85]]) box(w, 0.05, 0.04, mat(0x2a313d), x, 1.72, -2.56);
  for (let i = 1; i < 8; i++) box(0.06, 1.63, 0.06, mat(0x262c37), -4.3 + i * 13 / 8, 2.865, -2.55);
  box(13.2, 0.08, 0.1, mat(0x262c37), 2.2, 2.02, -2.55);
  box(13.2, 0.08, 0.1, mat(0x262c37), 2.2, 3.68, -2.55);
  for (const x of [-4.38, 8.78]) box(0.1, 1.8, 0.12, mat(0x262c37), x, 2.85, -2.54);
  box(13.4, 0.06, 0.22, mat(0x3a4250), 2.2, 1.97, -2.47);
  sign("FLOOR 1", 1.05, 0.4, -0.8, 4.05, -2.56);
  sign("QUIET, PLEASE", 1.4, 0.32, 2.4, 4.02, -2.56, "#d9c9a8", "#3a2a22");
  keyLight(2.5, 7, 4, 2.5, Math.PI / 4.5, 26);
  for (const x of [-0.6, 3.2, 7]) {
    tubeLights.push(tube(x, 4.4, 0.4));
    halo(0xdfe9e4, 1.1, x, 4.28, 0.45, scene, 0.13);
    pool(scene, 0xc8daf0, 1.7, x, 0.02, 0.4, 0.055);
    for (const dx of [-0.7, 0.7]) box(0.012, 0.58, 0.012, mat(0x8a9099, { metalness: 0.6, roughness: 0.4 }), x + dx, 4.69, 0.4);
  }
  // The ends are lit less than the middle: the key tube does the work, corners fall dark.
  tubeLights[0].intensity = 0.36; tubeLights[2].intensity = 0.3;
  // A warm reading lamp by the shelves, against the cool tubes.
  const lampWood = mat(0x3b3029);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 0.03, 16), lampWood); base.position.set(7.5, 0.015, -0.8); scene.add(base);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, 1.5, 10), lampWood); pole.position.set(7.5, 0.76, -0.8); pole.castShadow = true; scene.add(pole);
  const shade = new THREE.Mesh(new THREE.ConeGeometry(0.24, 0.26, 20, 1, true), mat(0xc9b39d, { side: THREE.DoubleSide, emissive: 0xffc070, emissiveIntensity: 0.5, roughness: 0.8 }));
  shade.position.set(7.5, 1.55, -0.8); scene.add(shade);
  halo(0xffb45c, 0.7, 7.5, 1.5, -0.8, scene, 0.4);
  const warm = new THREE.PointLight(0xffc27a, 0.75, 8, 1.6); warm.position.set(7.5, 1.45, -0.8); scene.add(warm);
  pool(scene, 0xffa850, 1.4, 7.5, 0.02, -0.8, 0.15);
  for (const x of [5.2, 9.5]) { box(1.8, 0.08, 0.9, wood, x, 0.98, -1.6); box(0.5, 0.9, 0.8, woodDark, x + 0.55, 0.47, -1.6); }
  const plant = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 0), mat(0x3f5a4b, { flatShading: true }));
  plant.position.set(-3.4, 0.72, -1.9); plant.castShadow = true; scene.add(plant);
  box(0.36, 0.5, 0.36, mat(0x55473a), -3.4, 0.25, -1.9);
  // The courier's drop: a low table with the night's envelopes waiting.
  box(0.7, 0.05, 0.5, woodDark, -2.7, 0.72, -1.0);
  for (const [dx, dz] of [[-0.3, -0.2], [0.3, -0.2], [-0.3, 0.2], [0.3, 0.2]]) box(0.04, 0.7, 0.04, woodDark, -2.7 + dx, 0.35, -1.0 + dz);
  for (const [dy, r] of [[0, 0.1], [0.014, -0.08], [0.028, 0.05]]) {
    const e = box(0.3, 0.012, 0.2, mat(0xdfe3ea, { roughness: 0.6 }), -2.72, 0.76 + dy, -1.0);
    e.rotation.y = r;
  }
  // A filing cabinet under the cork board; notices pinned to both.
  box(0.52, 1.2, 0.5, mat(0x4d535c, { roughness: 0.55 }), -1.3, 0.6, -2.15);
  for (const y of [0.35, 0.7, 1.05]) box(0.12, 0.02, 0.02, mat(0x8a9099, { metalness: 0.6, roughness: 0.4 }), -1.3, y, -1.88);
  box(0.16, 0.2, 0.004, paperM, -1.4, 0.98, -1.889, scene).rotation.z = 0.04;
  box(0.1, 0.1, 0.004, mat(0xd8c27a), -1.16, 0.86, -1.889, scene).rotation.z = -0.07;
  const cork = canvasTex(512, 340, (g, w, h) => {
    g.fillStyle = "#8a6a45"; g.fillRect(0, 0, w, h);
    for (let k = 0; k < 700; k++) { g.fillStyle = "rgba(60,42,24,0.35)"; g.fillRect((k * 83) % w, (k * 131) % h, 3, 3); }
    const note = (x, y, nw, nh, bg, rot) => {
      g.save(); g.translate(x, y); g.rotate(rot);
      g.fillStyle = "rgba(0,0,0,0.25)"; g.fillRect(4, 5, nw, nh);
      g.fillStyle = bg; g.fillRect(0, 0, nw, nh);
      g.fillStyle = "#5a5f68";
      for (let r0 = 0; r0 < 4; r0++) g.fillRect(10, 16 + r0 * 16, nw - 20 - ((r0 * 37) % 40), 5);
      g.fillStyle = "#c2453d"; g.beginPath(); g.arc(nw / 2, 8, 6, 0, 7); g.fill();
      g.restore();
    };
    note(60, 60, 150, 110, "#f2f2ec", 0.05); note(250, 40, 130, 90, "#f2e6c0", -0.06); note(150, 190, 170, 100, "#e8eef2", 0.02);
  });
  const corkBoard = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 0.66), new THREE.MeshStandardMaterial({ map: cork, roughness: 0.9 }));
  corkBoard.position.set(-1.3, 1.9, -2.55); scene.add(corkBoard);
  for (const [w, h, x, y] of [[1.12, 0.05, -1.3, 2.255], [1.12, 0.05, -1.3, 1.545], [0.05, 0.72, -1.83, 1.9], [0.05, 0.72, -0.77, 1.9]]) box(w, h, 0.05, woodDark, x, y, -2.54);
  // The wall clock on the band above the window; its hands are driven from the stopwatch.
  const face = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.04, 28), mat(0xe9ebe6, { emissive: 0x30343a, emissiveIntensity: 0.3 }));
  face.rotation.x = Math.PI / 2; face.position.set(5.0, 3.97, -2.55); scene.add(face);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.03, 6, 28), mat(0x1a1f27));
  rim.position.set(5.0, 3.97, -2.53); scene.add(rim);
  for (let k = 0; k < 12; k++) box(0.016, k % 3 ? 0.032 : 0.055, 0.01, mat(0x1a1f27), 5.0 + Math.sin(k * Math.PI / 6) * 0.23, 3.97 + Math.cos(k * Math.PI / 6) * 0.23, -2.52, scene).rotation.z = -k * Math.PI / 6;
  for (const [len, ref] of [[0.15, "h1"], [0.22, "h2"]]) {
    const pivot = new THREE.Group(); pivot.position.set(5.0, 3.97, -2.5); scene.add(pivot);
    box(0.02, len, 0.012, mat(0x1a1f27), 0, len / 2 - 0.01, 0, pivot);
    clock[ref] = pivot;
  }
})();
makeDesk(0.35, 1.35);
// Just behind where the adviser stands, so sitting down lands on the seat.
makeChair(0.35, 0.15);
// A runner along the adviser's walk, so the path crosses a furnished floor and not bare carpet.
const runnerTex = canvasTex(512, 112, (g, w, h) => {
  g.fillStyle = "#232c3a"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#55637a"; g.lineWidth = 7; g.strokeRect(12, 10, w - 24, h - 20);
  g.strokeStyle = "#8a6a45"; g.lineWidth = 3; g.strokeRect(26, 20, w - 52, h - 40);
  g.fillStyle = "rgba(120,132,150,0.22)";
  for (let x = 44; x < w - 40; x += 42) { g.save(); g.translate(x, h / 2); g.rotate(Math.PI / 4); g.fillRect(-10, -10, 20, 20); g.restore(); }
  for (let k = 0; k < 500; k++) { g.fillStyle = "rgba(16,20,26,0.4)"; g.fillRect((k * 97) % w, (k * 61) % h, 2, 2); }
});
const runner = new THREE.Mesh(new THREE.PlaneGeometry(4.4, 1.0), new THREE.MeshStandardMaterial({ map: runnerTex, roughness: 1 }));
runner.rotation.x = -Math.PI / 2; runner.position.set(3.2, 0.031, 0.05); runner.receiveShadow = true; scene.add(runner);

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
    // The one manual the adviser reads through each run, in the desk book's colour.
    const manual = s === 1 && !shelfManual && x > -0.1 && x < 0;
    const b = box(w, h, 0.36, mat(manual ? 0x4f6a63 : spineCols[k % spineCols.length]), x + w / 2, y + 0.025 + h / 2, 0.02, shelfBody);
    if (manual) shelfManual = b;
    if (k % 9 === 4) b.rotation.z = 0.12;
    x += w + 0.008; k++;
  }
  box(0.03, 0.3, 0.34, woodDark, 0.5, y + 0.17, 0.04, shelfBody).rotation.z = -0.16;
}
sign("MANUALS", 0.85, 0.24, 0, 2.62, 0.06, "#d9c9a8", "#3a2a22", shelfBody);
shelf.position.set(FAR, 0, 0);
shelf.rotation.y = -0.18;
let shelfRest = true;

const HOME = 0.35;
const A1 = makeAdviser({ x: HOME, z: 0.45 });
const adv = A1.group;
// A dark cap with an amber band and a badge, so the adviser is a person and not the stock figure.
const capC = mat(0x42352a, { roughness: 0.95 });
const cap = new THREE.Mesh(new THREE.SphereGeometry(0.178, 18, 10, 0, Math.PI * 2, 0, Math.PI / 2.05), capC);
cap.position.y = 0.17; cap.castShadow = true; A1.head.add(cap);
box(0.26, 0.016, 0.17, capC, 0, 0.19, 0.17, A1.head).rotation.x = 0.12;
box(0.29, 0.02, 0.014, mat(0xe8a33d), 0, 0.185, 0.11, A1.head);
box(0.05, 0.07, 0.012, paperM, 0.12, 0.42, 0.236, A1.body);
box(0.05, 0.014, 0.012, mat(0xe8a33d), 0.12, 0.46, 0.236, A1.body);
// The manual, carried flat across both hands; rotated back out of the forearm's tilt in the carry pose.
const handBook = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.07, 0.24), mat(0x4f6a63));
handBook.position.set(-0.25, 0.02, -0.04); handBook.rotation.x = 1.95; handBook.visible = false; A1.armR.hand.add(handBook);

const E1 = makeEnvelope();
const ENV_IN = new THREE.Vector3(0.15, 1.03, 1.1), ENV_OUT = new THREE.Vector3(-2.7, 1.6, -1.0);
E1.group.position.copy(ENV_OUT);
const deskBook = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, 0.22), mat(0x4f6a63));
deskBook.position.set(0.75, 1.05, 1.05); deskBook.rotation.y = 0.3; deskBook.castShadow = true; deskBook.visible = false; scene.add(deskBook);

// Warmth from the desk lamp, on the desktop and the floor.
pool(scene, 0xffb060, 0.5, -0.28, 1.02, 1.12, 0.38);
pool(scene, 0xffa040, 1.0, 0.1, 0.02, 1.5, 0.1);

// Dust drifting in the tubes' light, so the room is not perfectly still.
const motes = reduced ? [] : Array.from({ length: 6 }, (_, i) => {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color: 0xd8dde6, transparent: true, opacity: 0.05, depthWrite: false }));
  s.scale.setScalar(0.05 + (i % 3) * 0.015);
  s.position.set(0.4 + i * 0.55 + Math.random(), 1.3 + Math.random() * 1.5, 0.1 + Math.random() * 0.6);
  s.userData.ph = i * 1.3;
  scene.add(s); return s;
});

// The walk, drawn on the floor in the reveal
const pathGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(HOME, 0.02, 0.6), new THREE.Vector3(FAR - 0.8, 0.02, 0.6)]);
const pathLine = new THREE.Line(pathGeo, new THREE.LineDashedMaterial({ color: 0xe8a33d, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0 }));
pathLine.computeLineDistances(); scene.add(pathLine);
function setPath() {
  const p = pathGeo.attributes.position;
  p.setX(1, Math.max(HOME + 0.3, shelf.position.x - 0.8)); p.needsUpdate = true;
  pathLine.computeLineDistances();
}

const camera1 = { look: new THREE.Vector3(2.4, 1.05, 0), wide: new THREE.Vector3(2.6, 3.0, 8.2), narrow: new THREE.Vector3(2.6, 3.4, 13.0), fog: [13, 32], par: 1 };
// A tall window sees little sideways; widen its fov so desk and shelf both stay in frame.
camera1.place = () => {
  if (innerWidth / innerHeight < 0.9) { view.baseFov = 74; return camera1.narrow; }
  view.baseFov = 42;
  return camera1.wide;
};
const anchors = [
  [$("aAdv"), () => up(adv, 2.2)],
  [$("aShelf"), () => up(shelf, 2.65)],
  [$("aBook"), () => up(deskBook, 0.2)],
  [$("aPath"), () => new THREE.Vector3((HOME + shelf.position.x - 0.8) / 2, 0.1, 0.6)],
];

const squash = (u, kk) => shelfBody.scale.set(1 + 0.6 * kk * Math.sin(u * Math.PI), 1 - kk * Math.sin(u * Math.PI), 1 + 0.6 * kk * Math.sin(u * Math.PI));

// The shelf crouches, pops its legs out, looks both ways, waddles to `to` in `seconds`, and sets itself down.
async function moveShelf(to, seconds) {
  shelfRest = false;
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
  await tween(0.3, (u) => squash(u, 0.07));
  shelfBody.scale.set(1, 1, 1);
  shelfRest = true;
}

// Floor 1: one word delivered and stamped. The clock counts only the adviser's fetching and thinking.
async function deliver(run) {
  setClock(0, 0);
  E1.stamp.visible = false; E1.stamp.scale.set(1, 1, 1); deskBook.visible = false; shelfManual.visible = true;
  clock.run = true; clock.a = 0;
  await tween(0.7, (u) => {
    E1.group.position.lerpVectors(ENV_OUT, ENV_IN, ease(u));
    E1.group.rotation.z = (1 - ease(u)) * Math.sin(u * 13) * 0.18;
  });
  E1.group.rotation.z = 0;
  await sleep(0.2);
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
    if (A1.reach) shelfManual.visible = false;  // the manual comes off this shelf
    handBook.visible = A1.carry = !there;
    setClock(run.fetch * u, 0);
  });
  A1.walking = 0;
  adv.position.x = HOME;
  handBook.visible = A1.carry = false; deskBook.visible = true;
  await sitAndThink(A1, run.fetch, run.think);
  E1.stamp.visible = true;
  clock.run = false;
  if (!reduced) tween(0.25, (u) => { const s = 1 + 0.6 * Math.sin(u * Math.PI); E1.stamp.scale.set(s, 1, s); });
  addHistory(run);
  await sleep(0.4);
}

function resetFloor1() {
  shelf.position.x = FAR; setPath();
  shelfBody.position.y = 0; shelfBody.scale.set(1, 1, 1); shelfRest = true;
  adv.position.set(HOME, 0, 0.45); adv.rotation.y = A1.heading = 0; A1.sit = 0; A1.reach = false;
  E1.group.position.copy(ENV_OUT); E1.group.rotation.z = 0; E1.stamp.visible = false; E1.stamp.scale.set(1, 1, 1);
  deskBook.visible = false; shelfManual.visible = true; clock.run = false;
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
  await speak("faster1");
  await deliver(RUNS.faster);
  findEnding("thinking", "The Thinking Ending");
  await speak("faster2");
  if (endings[1].has("shelf")) choose([["See what it means", reveal]]);
  else choose([["Try the other door", closer]]);
}
async function closer() {
  clearStage();
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
let tubeLit = 1;

export default {
  ...config, anchors, camera: camera1, enter: resetFloor1, start: intro,
  frame(t, dt) {
    poseAdviser(A1, t, dt);
    if (clock.run) { clock.a += dt * 1.2; clock.h2.rotation.z = -clock.a * 2.4; clock.h1.rotation.z = -clock.a * 0.1; }
    if (!reduced) {
      if (shelfRest) shelfBody.position.y = (1 + Math.sin(t * 1.3)) * 0.008;  // the shelf, waiting, breathing
      const dip = (t * 0.53 + 1.7) % 9 < 0.05 ? 0.15 : 1;
      tubeLit += (dip - tubeLit) * Math.min(1, dt * 12);
      tubeLights[1].intensity = 0.45 * tubeLit;
      for (const m of motes) {
        m.position.y += dt * 0.03;
        m.position.x += Math.sin(t * 0.5 + m.userData.ph) * dt * 0.02;
        if (m.position.y > 2.9) m.position.y = 1.2;
        m.material.opacity = 0.035 + 0.03 * Math.sin(t * 0.8 + m.userData.ph);
      }
    }
  },
  onLabels(on) { pathLine.material.opacity = on ? 0.95 : 0; },
};
