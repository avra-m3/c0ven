// Floor 3, musical desks, around x = X3. Which experts sit on the card follows the conversation: the player runs one
// night by hand, then the narrator runs the same night with the engine's swap rule.
import { $, box, camera, canvasTex, halo, haloTex, mat, pool, reduced, scene, sign, toScreen, tubes, up, v, view, windowsTex } from "../world.js";
import { backOut, ease, lerp, tween } from "../motion.js";
import { makeAdviser, makeChair, makeEnvelope, makeFax, makeFaxPage, paperM, poseAdviser, wood, woodDark, wrapAngle } from "../props.js";
import { speak, speaking } from "../narration.js";
import { addHistory, choose, clearStage, endings, findEnding, goFloor, labels, revealButtons, setClock } from "../hud.js";

const config = {
  number: 3, name: "Musical Desks", minutes: 3, scale: 30, keys: ["Moving", "Thinking"], watch: "Time for the night's words",
  labels: ["aDesk3", "aAnnex3", "aBoard", "aMover"],
  map: [
    ["Desks", "the graphics card's room for experts: only some fit"],
    ["Annex", "main memory, which holds every expert; the CPU works out the ones not on the card"],
    ["Tally board", "how often each expert was used lately; the counts fade every round"],
    ["A move", "a swap: copying an expert onto the card, in place of a quiet one"],
    ["The letter", "a new prompt: the experts it used count too, and a round follows it"],
  ],
  lesson: "Which experts a conversation needs changes as it goes. c0ven counts what each word uses, and every few words swaps the busiest outsiders in for the quietest residents, but only when the difference is clear. The copying runs alongside the work, so the card pauses for only a moment.",
  next: "Floor 4 · coming next.",
};

// Two rooms of four desks, the main office and the annex, either side of a lane ("row") or, on a tall screen, one
// behind the other ("stack"). Each room is built around its own origin and layout3 places them. Slots 0-3 are the
// office's chairs and 4-7 the annex's, each room's in the order back left, back right, front left, front right.
// A desk sits 1.2 in front of its chair.
const X3 = 120;
const wireMat = new THREE.LineBasicMaterial({ color: 0x14171c });
const COL = 1.3, ROWZ = [-1.8, 0.4], RW = 6.0, RD = 5.2, LANE = 3.4;
const SLOTS = Array.from({ length: 8 }, () => [0, 0]);
const STAFF = [
  { name: "Pat", topic: "Cooking", color: 0xe07a3b, suit: 0x56657d, hair: 0x3b322b, skin: 0xe0c3a8, hat: "toque" },
  { name: "Dev", topic: "Sport", color: 0x4fb36b, suit: 0x5b6f63, hair: 0x2a2420, skin: 0x8d5a3b, hat: "cap", he: true },
  { name: "Ines", topic: "Travel", color: 0x4f8fd0, suit: 0x6a5f78, hair: 0x6b4a2e, skin: 0xb98a64, hat: "beret" },
  { name: "Otto", topic: "Weather", color: 0x7fcfd0, suit: 0x4d5868, hair: 0x8a7a62, skin: 0xd9b9a0, hat: "souwester", he: true },
  { name: "Margaret", topic: "Law", color: 0xb05ac0, suit: 0x6b5a48, hair: 0xc4c4bf, skin: 0xe3cbb5, hat: "wig", old: true },
  { name: "Hugh", topic: "Contracts", color: 0xd2556b, suit: 0x3f4a5c, hair: 0x4a3b2c, skin: 0xa77555, hat: "bowler", he: true },
  { name: "Rosa", topic: "Music", color: 0xe0c050, suit: 0x7a5a5a, hair: 0x2a2020, skin: 0x6e4630, hat: "phones" },
  { name: "Bram", topic: "Gardens", color: 0x9aa64a, suit: 0x55604a, hair: 0x6b5a3a, skin: 0xd2a789, hat: "straw", he: true },
];
const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const MFAX = new THREE.Vector3(), AFAX = new THREE.Vector3();
// The layout in use and what hangs off it: each room's aisle, the board's face, the label points, who the
// current question needs (need), who the rule would move in (hot) and who the reveal's mover label follows.
const L3 = { mode: "", aisle: [0, 0], board: new THREE.Vector3(), office: new THREE.Vector3(), annex: new THREE.Vector3(), need: null, mover: null, letterMark: false, hot: new Set() };
const rooms = [new THREE.Group(), new THREE.Group()];
const R3 = {};
// The tally board: a chalkboard redrawn from the counts, lit from within so it reads at a distance.
const boardTex = canvasTex(1024, 640, () => {});
const boardMat = new THREE.MeshStandardMaterial({ map: boardTex, emissiveMap: boardTex, emissive: 0xffffff, emissiveIntensity: 0.32, roughness: 0.9 });
const BW = 3.2, BH = 2.0;

// Tiled surfaces, 256 pixels to a unit; rep() gives a surface its own repeat.
function tile(draw) { const t = canvasTex(256, 256, draw); t.wrapS = t.wrapT = THREE.RepeatWrapping; return t; }
const rep = (t, x, y) => { const c = t.clone(); c.needsUpdate = true; c.repeat.set(x, y); return c; };
const carpetTex = tile((g) => {
  for (let i = 0; i < 4; i++) { g.fillStyle = i === 1 || i === 2 ? "#2a3240" : "#2f3846"; g.fillRect((i % 2) * 128, (i >> 1) * 128, 128, 128); }
  for (let k = 0; k < 1600; k++) { g.fillStyle = k % 3 ? "rgba(70,84,104,0.5)" : "rgba(14,18,24,0.5)"; g.fillRect((k * 53) % 256, (k * 97 + (k >> 3)) % 256, 2, 2); }
  g.fillStyle = "rgba(10,13,18,0.55)"; g.fillRect(0, 0, 256, 2); g.fillRect(0, 128, 256, 2); g.fillRect(0, 0, 2, 256); g.fillRect(128, 0, 2, 256);
});
const boardsTex = tile((g) => {
  for (let r = 0; r < 5; r++) {
    const y = Math.round(r * 51.2);
    g.fillStyle = ["#4c3828", "#563f2c", "#45332a", "#513b29", "#4a3729"][r]; g.fillRect(0, y, 256, 52);
    g.strokeStyle = "rgba(255,226,190,0.05)"; g.lineWidth = 2;
    for (let k = 0; k < 4; k++) { g.beginPath(); g.moveTo(0, y + 10 + k * 10); g.bezierCurveTo(80, y + 5 + k * 11, 170, y + 15 + k * 9, 256, y + 10 + k * 10); g.stroke(); }
    g.fillStyle = "rgba(18,11,7,0.7)"; g.fillRect(0, y, 256, 2); g.fillRect((r * 89 + 40) % 256, y, 2, 52);
  }
});
const brickTex = tile((g) => {
  g.fillStyle = "#33231e"; g.fillRect(0, 0, 256, 256);
  for (let r = 0; r < 12; r++) for (let c = -1; c < 4; c++) {
    g.fillStyle = ["#5e3d33", "#664236", "#563a30", "#6b4639"][(r * 3 + c * 5 + 8) % 4];
    g.fillRect(c * 64 + (r % 2) * 32 + 2, Math.round(r * 21.33) + 2, 60, 18);
  }
});
const cobbleTex = tile((g) => {
  g.fillStyle = "#121519"; g.fillRect(0, 0, 256, 256);
  for (let r = 0; r < 5; r++) for (let c = -1; c < 5; c++) {
    g.fillStyle = ["#2a2e35", "#31353c", "#25292f", "#363a41"][(r * 7 + c * 3 + 8) % 4];
    g.beginPath(); g.ellipse(c * 64 + (r % 2) * 32 + 32, r * 51.2 + 25.6, 28, 22, 0, 0, Math.PI * 2); g.fill();
  }
});
const slabTex = tile((g) => {
  g.fillStyle = "#3a3f47"; g.fillRect(0, 0, 256, 256);
  g.fillStyle = "#30353c"; g.fillRect(0, 0, 256, 3); g.fillRect(0, 0, 3, 256);
  for (let k = 0; k < 500; k++) { g.fillStyle = k % 2 ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.08)"; g.fillRect((k * 71) % 256, (k * 113) % 256, 3, 3); }
});
// A side wall, cut away towards the front so the room stays open to the camera.
function sideWall(parent, x, m, back, front) {
  const sh = new THREE.Shape();
  sh.moveTo(-RD / 2, 0); sh.lineTo(RD / 2, 0); sh.lineTo(RD / 2, front); sh.lineTo(-RD / 2, back); sh.closePath();
  const w = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, { depth: 0.12, bevelEnabled: false }), m);
  // The shape's x runs along z, back to front; the extrusion is the wall's thickness, along -x.
  w.rotation.y = -Math.PI / 2; w.position.set(x + 0.06, 0, 0); w.castShadow = w.receiveShadow = true;
  parent.add(w); return w;
}

// A desk centred at (x, z) of a room, the adviser behind it facing out. The office's: pale laminate, a slim lamp, a
// laptop. The annex's: dark oak, a green banker's lamp, books. Lamps are emissive with a halo and a pool of light.
const laminate = mat(0x9ba1a9, { roughness: 0.5 }), steel3 = mat(0x6d737c, { metalness: 0.5, roughness: 0.45 }), plastic = mat(0x23272f, { roughness: 0.5 });
const oak = mat(0x4a3324, { roughness: 0.6 }), leather = mat(0x2d4a39, { roughness: 0.7 }), brass = mat(0xb08a4a, { metalness: 0.7, roughness: 0.35 });
function desk3(room, x, z, annex) {
  const top = box(1.8, 0.07, 0.9, annex ? oak : laminate, x, 0.985, z, room);
  top.castShadow = true;
  if (annex) {
    box(1.4, 0.006, 0.55, leather, x - 0.1, 1.022, z + 0.05, room);
    box(1.7, 0.8, 0.05, oak, x, 0.53, z + 0.4, room);
    for (const sx of [-0.85, 0.85]) box(0.07, 0.95, 0.86, oak, x + sx, 0.475, z, room);
    for (const [w, h, c, dy] of [[0.3, 0.07, 0x6b3a32, 0], [0.27, 0.06, 0x2f4a5c, 0.065], [0.29, 0.05, 0x7a6a3a, 0.12]]) {
      const b = box(w, h, 0.21, mat(c), x + 0.62, 1.055 + dy, z - 0.2, room); b.rotation.y = 0.1 - dy;
    }
    // Banker's lamp: brass stem, green half-shade lit from under.
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.09, 0.03, 16), brass); base.position.set(x - 0.62, 1.035, z - 0.22); room.add(base);
    box(0.02, 0.26, 0.02, brass, x - 0.62, 1.17, z - 0.22, room);
    const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.38, 16, 1, true, 0, Math.PI), mat(0x2f7a4c, { side: THREE.DoubleSide, emissive: 0x2f7a4c, emissiveIntensity: 0.35, roughness: 0.4 }));
    shade.rotation.z = Math.PI / 2; shade.position.set(x - 0.62, 1.31, z - 0.18); room.add(shade);
    box(0.34, 0.01, 0.12, mat(0xffe2a8, { emissive: 0xffc070, emissiveIntensity: 1.4 }), x - 0.62, 1.305, z - 0.18, room);
    halo(0xffb45c, 0.6, x - 0.62, 1.27, z - 0.1, room, 0.42);
    pool(room, 0xffb060, 0.6, x - 0.4, 1.027, z + 0.05, 0.45);
    pool(room, 0xffa040, 1.1, x - 0.3, 0.106, z + 0.75, 0.16);
  } else {
    for (const sx of [-0.82, 0.82]) {
      for (const dz of [-0.38, 0.38]) box(0.045, 0.95, 0.045, steel3, x + sx, 0.475, z + dz, room);
      box(0.045, 0.045, 0.8, steel3, x + sx, 0.08, z, room);
    }
    box(0.46, 0.6, 0.78, mat(0x5a606a, { roughness: 0.55 }), x + 0.6, 0.62, z, room);
    for (const dy of [0.48, 0.74]) box(0.14, 0.02, 0.02, steel3, x + 0.6, dy, z + 0.4, room);
    // A laptop open towards the adviser; the screen's glow spills round the lid.
    box(0.36, 0.02, 0.25, plastic, x + 0.02, 1.03, z - 0.14, room);
    const lid = box(0.36, 0.24, 0.014, plastic, x + 0.02, 1.15, z - 0.02, room); lid.rotation.x = 0.22;
    halo(0x9fc4ff, 0.5, x + 0.02, 1.16, z - 0.12, room, 0.3);
    // Slim arm lamp, cool white.
    box(0.14, 0.02, 0.14, steel3, x - 0.72, 1.03, z - 0.28, room);
    const arm = box(0.022, 0.44, 0.022, steel3, x - 0.7, 1.24, z - 0.24, room); arm.rotation.x = 0.25;
    box(0.24, 0.04, 0.09, mat(0xdde3ea, { emissive: 0xeaf2ff, emissiveIntensity: 1.2 }), x - 0.62, 1.44, z - 0.16, room);
    halo(0xdfeaff, 0.6, x - 0.62, 1.4, z - 0.13, room, 0.38);
    pool(room, 0xbcd4ff, 0.6, x - 0.45, 1.027, z - 0.02, 0.38);
    pool(room, 0x9fb8e0, 1.1, x - 0.3, 0.106, z + 0.75, 0.12);
  }
  const stack = box(0.28, 0.03, 0.21, paperM, x - 0.15, 1.035, z + 0.22, room); stack.rotation.y = 0.18;
  const mug = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.04, 0.1, 12), mat(annex ? 0xd9d2c0 : 0xe8a33d, { roughness: 0.5 }));
  mug.position.set(x + 0.32, 1.07, z + 0.2); mug.castShadow = true; room.add(mug);
}

// Hats and headphones, one per adviser, so each reads at a glance; they ride on the head and turn with it.
function dress3(A, kind, color) {
  const h = A.head, m = (c, o) => mat(c, o);
  const cyl = (rt, rb, ht, c, y, seg = 18) => { const k = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, ht, seg), c); k.position.y = y; k.castShadow = true; h.add(k); return k; };
  const ball = (r, c, x, y, z, sx = 1, sy = 1, sz = 1) => { const b = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 10), c); b.position.set(x, y, z); b.scale.set(sx, sy, sz); b.castShadow = true; h.add(b); return b; };
  if (kind === "toque") { cyl(0.13, 0.15, 0.18, m(0xf2f2ec), 0.36); ball(0.17, m(0xf2f2ec), 0, 0.47, 0, 1.05, 0.55, 1.05); }
  if (kind === "cap") { ball(0.172, m(color), 0, 0.17, 0, 1, 0.75, 1); const b = box(0.24, 0.016, 0.16, m(color), 0, 0.21, 0.17, h); b.rotation.x = 0.12; }
  if (kind === "beret") { const b = ball(0.19, m(color), 0.02, 0.29, -0.01, 1.05, 0.32, 1.05); b.rotation.z = 0.25; ball(0.02, m(color), 0.03, 0.36, 0); }
  if (kind === "souwester") { cyl(0.11, 0.17, 0.13, m(color, { roughness: 0.35 }), 0.31); cyl(0.26, 0.26, 0.015, m(color, { roughness: 0.35 }), 0.245, 24); }
  if (kind === "wig") {
    const wig = m(0xece8dc, { roughness: 0.95 });
    ball(0.18, wig, 0, 0.2, -0.01, 1, 0.75, 1);
    for (const sx of [-1, 1]) for (let k = 0; k < 3; k++) ball(0.045, wig, sx * 0.165, 0.18 - k * 0.075, -0.03, 1, 0.8, 1);
  }
  if (kind === "bowler") { cyl(0.2, 0.2, 0.014, m(0x1c2027), 0.255, 24); ball(0.14, m(0x1c2027), 0, 0.27, 0, 1, 0.85, 1); cyl(0.142, 0.142, 0.03, m(color), 0.275); }
  if (kind === "phones") {
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.018, 6, 20, Math.PI), m(0x23272f)); band.position.y = 0.15; h.add(band);
    for (const sx of [-1, 1]) { const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.05, 14), m(color)); cup.rotation.z = Math.PI / 2; cup.position.set(sx * 0.175, 0.14, 0); h.add(cup); }
  }
  if (kind === "straw") { cyl(0.3, 0.3, 0.014, m(0xd8c27a, { roughness: 0.9 }), 0.25, 24); cyl(0.13, 0.15, 0.12, m(0xd8c27a, { roughness: 0.9 }), 0.31); cyl(0.152, 0.152, 0.035, m(color), 0.275); }
}

(() => {
  const [office, annex] = rooms;
  scene.add(office, annex);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(70, 50), mat(0x13161b, { roughness: 1 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(X3, 0, 0); ground.receiveShadow = true; scene.add(ground);

  // The main office: blue-grey carpet, plaster walls, a tube overhead, windows onto the city
  const plaster = mat(0x3e495a), trimM = mat(0x2a313d);
  box(RW, 0.1, RD, mat(0xffffff, { map: rep(carpetTex, RW, RD), roughness: 1 }), 0, 0.05, 0, office);
  box(RW, 4.6, 0.12, plaster, 0, 2.3, -RD / 2, office);
  box(RW, 0.07, 0.05, trimM, 0, 1.05, -RD / 2 + 0.08, office);
  box(RW, 0.14, 0.03, mat(0x1d222b), 0, 0.17, -RD / 2 + 0.075, office);
  for (const sx of [-1, 1]) sideWall(office, sx * RW / 2, plaster, 4.6, 1.3);
  R3.officeSign = sign("MAIN OFFICE", 2.4, 0.42, 0, 4.22, -RD / 2 + 0.07, undefined, undefined, office);
  R3.officeWin = new THREE.Group(); office.add(R3.officeWin);
  for (const sx of [-1, 1]) {
    const t = rep(windowsTex, 0.22, 1); t.offset.x = sx > 0 ? 0.55 : 0.12;
    const w = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.15), new THREE.MeshBasicMaterial({ map: t, color: 0xaab4c2 }));
    w.position.set(sx * 1.75, 2.75, -RD / 2 + 0.07); R3.officeWin.add(w);
    for (const [fw, fh, fx, fy] of [[1.5, 0.07, 0, 0.6], [1.5, 0.1, 0, -0.6], [0.07, 1.2, -0.72, 0], [0.07, 1.2, 0.72, 0], [0.04, 1.15, 0, 0]]) box(fw, fh, 0.05, trimM, sx * 1.75 + fx, 2.75 + fy, -RD / 2 + 0.09, R3.officeWin);
  }
  const clock = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.04, 24), mat(0xe9ebe6, { emissive: 0x30343a, emissiveIntensity: 0.3 }));
  clock.rotation.x = Math.PI / 2; clock.position.set(-2.4, 3.6, -RD / 2 + 0.09); office.add(clock);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.025, 6, 24), mat(0x1a1f27)); rim.position.set(-2.4, 3.6, -RD / 2 + 0.11); office.add(rim);
  for (let k = 0; k < 12; k++) box(0.015, k % 3 ? 0.03 : 0.05, 0.01, mat(0x1a1f27), -2.4 + Math.sin(k * Math.PI / 6) * 0.17, 3.6 + Math.cos(k * Math.PI / 6) * 0.17, -RD / 2 + 0.115, office).rotation.z = -k * Math.PI / 6;
  for (const [len, rot] of [[0.16, 1.05], [0.1, 2.15]]) { const hnd = box(0.018, len, 0.01, mat(0x1a1f27), -2.4 + Math.sin(rot) * len / 2, 3.6 + Math.cos(rot) * len / 2, -RD / 2 + 0.12, office); hnd.rotation.z = -rot; }
  // Hanging tube: emissive with a glow; the room's one real light sits under it.
  const tubeM = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.05, 0.16), new THREE.MeshBasicMaterial({ color: 0xe6efe9 }));
  tubeM.position.set(0, 3.75, -0.7); office.add(tubeM); tubes.push(tubeM);
  halo(0xdfe9e4, 1.6, 0, 3.7, -0.65, office, 0.16);
  for (const wx of [-1.0, 1.0]) box(0.01, 0.85, 0.01, steel3, wx, 4.2, -0.7, office);
  const cool = new THREE.PointLight(0xdfe9e4, 0.55, 10, 1.6); cool.position.set(0, 3.5, 0.2); office.add(cool);
  pool(office, 0xc8daf0, 2.6, 0, 0.104, -0.4, 0.1);
  // Water cooler, plant, filing cabinet, coat stand
  box(0.34, 0.9, 0.34, mat(0xc9ccc4, { roughness: 0.5 }), -2.55, 0.55, -2.15, office);
  const bottle = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.42, 16), mat(0x7fb3d9, { transparent: true, opacity: 0.75, roughness: 0.15, emissive: 0x1d3a52, emissiveIntensity: 0.5 }));
  bottle.position.set(-2.55, 1.21, -2.15); office.add(bottle);
  box(0.36, 0.4, 0.36, mat(0x55473a), 2.55, 0.3, -2.2, office);
  const plant = new THREE.Mesh(new THREE.IcosahedronGeometry(0.4, 0), mat(0x3f5a4b, { flatShading: true }));
  plant.position.set(2.55, 0.86, -2.2); plant.scale.y = 1.3; plant.castShadow = true; office.add(plant);
  box(0.5, 1.15, 0.5, mat(0x4d535c, { roughness: 0.5 }), -2.68, 0.68, -0.6, office);
  for (const dy of [0.45, 0.8, 1.1]) box(0.02, 0.02, 0.16, steel3, -2.42, dy, -0.6, office);
  const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, 1.8, 8), mat(0x3b3029)); stand.position.set(2.7, 1.0, 2.25); office.add(stand);
  const coat = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.17, 0.8, 10), mat(0x6a4a3a)); coat.position.set(2.62, 1.35, 2.25); office.add(coat);

  // The annex: floorboards, brick, every manual nobody visits, warm lamps
  box(RW, 0.1, RD, mat(0xffffff, { map: rep(boardsTex, RW, RD), roughness: 0.8 }), 0, 0.05, 0, annex);
  R3.annexBack = new THREE.Group(); annex.add(R3.annexBack);
  box(RW, 4.6, 0.12, mat(0xffffff, { map: rep(brickTex, RW, 4.6), roughness: 0.95 }), 0, 2.3, -RD / 2, R3.annexBack);
  const library = canvasTex(1024, 640, (g, w, h) => {
    g.fillStyle = "#2c2520"; g.fillRect(0, 0, w, h);
    const cols = ["#5d6b7f", "#7a6a55", "#4f6a63", "#6d7486", "#8a5a48", "#3f4e66", "#6b5a3a"];
    let n = 0;
    for (let row = 0; row < 5; row++) {
      const y1 = 14 + row * 126 + 108;
      let x = 10;
      while (x < w - 16) {
        const bw = 9 + ((n * 7) % 6) * 3, bh = 80 + ((n * 5) % 7) * 4;
        g.fillStyle = cols[n % cols.length]; g.fillRect(x, y1 - bh, bw, bh);
        x += bw + 2; n++;
      }
      g.fillStyle = "#4a3d31"; g.fillRect(0, y1, w, 12);
    }
  });
  const lib = new THREE.Mesh(new THREE.PlaneGeometry(RW - 0.7, 3.1), new THREE.MeshStandardMaterial({ map: library, roughness: 0.9 }));
  lib.position.set(0, 1.75, -RD / 2 + 0.07); R3.annexBack.add(lib);
  for (const lx of [-1, 1]) { const rail = box(0.05, 3.3, 0.05, mat(0x6b5845), 2.1 + lx * 0.2, 1.65, -RD / 2 + 0.35, R3.annexBack); rail.rotation.x = -0.1; }
  for (let k = 0; k < 7; k++) box(0.45, 0.03, 0.04, mat(0x6b5845), 2.1, 0.3 + k * 0.45, -RD / 2 + 0.35 - 0.04 * k + 0.12, R3.annexBack);
  sign("ANNEX", 1.4, 0.42, 0, 4.22, -RD / 2 + 0.07, "#d9c9a8", "#3a2a22", R3.annexBack);
  R3.annexSides = [-1, 1].map((sx) => sideWall(annex, sx * RW / 2, mat(0xffffff, { map: brickTex, roughness: 0.95 }), 4.6, 1.3));
  // On a tall screen the annex stands in front: a low parapet with a gap at the aisle replaces its back wall.
  R3.parapet = new THREE.Group(); annex.add(R3.parapet);
  for (const sx of [-1, 1]) box(RW / 2 - 0.62, 0.5, 0.16, mat(0xffffff, { map: rep(brickTex, 2.4, 0.5), roughness: 0.95 }), sx * (RW / 4 + 0.31), 0.25, -RD / 2, R3.parapet);
  sign("ANNEX", 1.1, 0.33, -1.6, 0.72, -RD / 2 + 0.09, "#d9c9a8", "#3a2a22", R3.parapet);
  const rug = canvasTex(128, 256, (g, w, h) => {
    g.fillStyle = "#5a2a2a"; g.fillRect(0, 0, w, h);
    g.strokeStyle = "#b08a4a"; g.lineWidth = 5; g.strokeRect(10, 10, w - 20, h - 20);
    g.strokeStyle = "#3a1c1e"; g.lineWidth = 3; g.strokeRect(22, 22, w - 44, h - 44);
    g.fillStyle = "#7a3a36"; for (let k = 0; k < 5; k++) { g.beginPath(); g.ellipse(64, 48 + k * 40, 22, 12, 0, 0, Math.PI * 2); g.fill(); }
  });
  const rugM = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 3.8), new THREE.MeshStandardMaterial({ map: rug, roughness: 1 }));
  rugM.rotation.x = -Math.PI / 2; rugM.position.set(0, 0.104, -0.3); rugM.receiveShadow = true; annex.add(rugM);
  const warm = new THREE.PointLight(0xffc27a, 0.75, 10, 1.6); warm.position.set(0, 3.4, 0.4); annex.add(warm);
  // Books piled by the wall, and the annex cat asleep on them
  for (const [bx, bz, n] of [[2.65, 0.4, 5], [2.62, -0.25, 3], [-2.66, -0.5, 4]]) for (let k = 0; k < n; k++) {
    const b = box(0.34 - (k % 2) * 0.04, 0.07, 0.26, mat([0x6b3a32, 0x2f4a5c, 0x7a6a3a, 0x4f6a63][(k + n) % 4]), bx, 0.135 + k * 0.07, bz, annex); b.rotation.y = (k * 0.7) % 0.5 - 0.25;
  }
  const fur = mat(0x9a7b5a, { roughness: 1 });
  const cat = new THREE.Group(); cat.position.set(2.65, 0.47, 0.4); cat.rotation.y = -0.6; annex.add(cat);
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.15, 14, 10), fur); body.scale.set(1.3, 0.62, 1); body.castShadow = true; cat.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 10), fur); head.position.set(0.17, 0.0, 0.05); cat.add(head);
  for (const ez of [-0.035, 0.035]) { const ear = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.05, 6), fur); ear.position.set(0.18, 0.07, 0.05 + ez); cat.add(ear); }
  const tail = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.022, 6, 14, Math.PI * 1.1), fur); tail.rotation.x = Math.PI / 2; tail.position.set(-0.02, -0.04, 0.02); cat.add(tail);
  R3.cat = body;

  for (const [r, room] of rooms.entries()) for (const row of ROWZ) for (const sx of [-1, 1]) desk3(room, sx * COL, row + 1.2, r === 1);

  // A side table with a fax machine, at each room's lane side; layout3 places them.
  R3.fax = rooms.map((room, r) => {
    const g = new THREE.Group(); room.add(g);
    box(0.6, 0.06, 0.5, r ? oak : laminate, 0, 0.75, 0, g);
    for (const [dx, dz] of [[-0.25, -0.2], [0.25, -0.2], [-0.25, 0.2], [0.25, 0.2]]) box(0.04, 0.72, 0.04, r ? oak : steel3, dx, 0.36, dz, g);
    g.add(makeFax(0, 0.79, 0, r ? 0.5 : -0.5));
    return g;
  });

  // The lane between the rooms, the pavement in front, and the road beyond (side by side only)
  R3.lane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshStandardMaterial({ map: rep(cobbleTex, 1, 1), roughness: 0.55, metalness: 0.1 }));
  R3.lane.rotation.x = -Math.PI / 2; R3.lane.receiveShadow = true; scene.add(R3.lane);
  // The street lamp's reflection in a wet patch of the lane.
  R3.puddle = pool(scene, 0xffb45c, 0.5, 0, 0.02, 0, 0.3); R3.puddle.scale.set(0.9, 2.2, 1);
  R3.street = new THREE.Group(); scene.add(R3.street);
  box(34, 0.06, 1.6, mat(0xffffff, { map: rep(slabTex, 34 / 0.8, 2), roughness: 0.9 }), 0, 0.03, RD / 2 + 0.8, R3.street);
  box(34, 0.12, 0.14, mat(0x4a4f58), 0, 0.06, RD / 2 + 1.65, R3.street);
  box(34, 0.02, 4.0, mat(0x181b21, { roughness: 0.95 }), 0, 0.01, RD / 2 + 3.7, R3.street);
  for (let x = -16; x <= 16; x += 1.4) box(0.7, 0.025, 0.08, mat(0xd8d2b8, { roughness: 0.8 }), x, 0.025, RD / 2 + 3.4, R3.street);

  // Two telephone poles and the line the fax travels on; one pole carries a street lamp.
  R3.poles = [0, 1].map(() => {
    const g = new THREE.Group(); scene.add(g);
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 3.7, 10), mat(0x3b3029)); p.position.y = 1.85; p.castShadow = true; g.add(p);
    g.arm = new THREE.Group(); g.add(g.arm);
    box(0.6, 0.06, 0.06, mat(0x3b3029), 0, 3.55, 0, g.arm);
    return g;
  });
  // The lamp's arm points along its local +z; layout3 turns it into the lane.
  R3.lamp = new THREE.Group(); R3.poles[0].add(R3.lamp);
  box(0.06, 0.05, 0.55, steel3, 0, 3.15, 0.27, R3.lamp);
  box(0.26, 0.08, 0.2, steel3, 0, 3.12, 0.55, R3.lamp);
  box(0.22, 0.03, 0.16, mat(0xffd59a, { emissive: 0xffb45c, emissiveIntensity: 1.3 }), 0, 3.07, 0.55, R3.lamp);
  halo(0xffb45c, 1.4, 0, 3.02, 0.6, R3.lamp, 0.42);
  R3.lampPool = pool(scene, 0xffa850, 1.6, 0, 0.03, 0, 0.32);
  R3.wire = new THREE.Line(new THREE.BufferGeometry(), wireMat); scene.add(R3.wire);

  // The board on its frame. Side by side it stands at the back of the lane on two posts; one behind the other it
  // leans against the office's back wall.
  R3.board = new THREE.Group(); scene.add(R3.board);
  R3.boardFace = new THREE.Group(); R3.board.add(R3.boardFace);
  const face = new THREE.Mesh(new THREE.PlaneGeometry(BW, BH), boardMat); R3.boardFace.add(face);
  for (const [w, h, x, y] of [[BW + 0.16, 0.08, 0, BH / 2 + 0.04], [BW + 0.16, 0.08, 0, -BH / 2 - 0.04], [0.08, BH, -BW / 2 - 0.04, 0], [0.08, BH, BW / 2 + 0.04, 0]]) box(w, h, 0.07, wood, x, y, 0, R3.boardFace);
  box(BW, 0.04, 0.12, woodDark, 0, -BH / 2 - 0.08, 0.06, R3.boardFace);
  R3.boardPosts = new THREE.Group(); R3.board.add(R3.boardPosts);
  for (const sx of [-1, 1]) box(0.1, 2.0, 0.1, woodDark, sx * (BW / 2 - 0.2), 1.0, -0.06, R3.boardPosts);

  const cityMat = new THREE.MeshBasicMaterial({ map: windowsTex, color: 0x8a93a0 });
  [[-15, 7, 3], [-10, 10, 3.5], [-4, 6, 4], [2.5, 12, 3], [8, 8, 4], [13, 11, 3.2], [18, 7, 3]].forEach(([dx, h, w]) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, 2), [mat(0x161a21), mat(0x161a21), mat(0x161a21), mat(0x161a21), cityMat, mat(0x161a21)]);
    b.position.set(X3 + dx, h / 2, -15); scene.add(b);
  });
  // A broad, soft key light over both rooms, for shadows; no hard pool on the ground.
  const key = new THREE.SpotLight(0xe8f0ea, 1.0, 50, 0.85, 1, 1);
  key.position.set(X3 + 1.5, 15, 10); key.target.position.set(X3, 0, 0);
  key.castShadow = true; key.shadow.mapSize.set(1024, 1024); key.shadow.bias = -0.0005;
  scene.add(key, key.target);
})();

// Legs that fold out from under a chair's seat; the same scaling trick as the shelf's.
function chairLegs(chair) {
  const legs = [];
  for (const [x, z, phase] of [[-0.17, 0.15, 0], [0.17, -0.15, 0], [0.17, 0.15, Math.PI], [-0.17, -0.15, Math.PI]]) {
    const pivot = new THREE.Group(); pivot.position.set(x, 0.44, z); chair.add(pivot);
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.022, 1, 8), mat(0x2b2620, { roughness: 0.7 }));
    leg.position.y = -0.5; leg.castShadow = true; pivot.add(leg);
    const foot = new THREE.Group(); foot.position.y = -1; pivot.add(foot);
    box(0.09, 0.05, 0.14, mat(0xb8893a, { metalness: 0.4, roughness: 0.4 }), 0, 0.01, 0.035, foot);
    pivot.visible = false;
    legs.push({ pivot, foot, phase });
  }
  return legs;
}
function setChairLegs(legs, len) {
  const L = Math.max(len, 0.001);
  for (const l of legs) { l.pivot.visible = len > 0.01; l.pivot.scale.y = L; l.foot.scale.y = 1 / L; }
}
const cardboard = mat(0xa07a4e, { roughness: 0.9 });
// The box an adviser packs while a move is planned: open flaps, a label in their colour.
function packBox(color) {
  const g = new THREE.Group(); scene.add(g);
  box(0.34, 0.24, 0.28, cardboard, 0, 0, 0, g);
  box(0.2, 0.08, 0.005, mat(color), 0, 0.02, 0.142, g);
  const flaps = [[0, 0.12, 0.14, 0.34, 0.14], [0, 0.12, -0.14, 0.34, 0.14]].map(([x, y, z, w, d]) => {
    const p = new THREE.Group(); p.position.set(x, y, z); g.add(p);
    box(w, 0.01, d, cardboard, 0, 0, Math.sign(z) * d / 2, p); return p;
  });
  g.visible = false;
  return { group: g, flaps };
}
for (const [i, s] of STAFF.entries()) {
  s.chair = makeChair(0, 0);
  s.chair.rotation.order = "YXZ";
  // The chair's fabric in the adviser's colour, so a moving pair is easy to follow.
  s.chair.traverse((o) => { if (o.material && o.material.color && o.material.color.getHex() === 0x2f3644) o.material.color.setHex(s.color).lerp(new THREE.Color(0x1c2027), 0.45); });
  s.legs = chairLegs(s.chair);
  s.A = makeAdviser({ suit: s.suit, hair: s.hair, tie: s.color, skin: s.skin, nodPhase: i * 0.9, old: !!s.old });
  s.A.group.rotation.order = "YXZ";
  dress3(s.A, s.hat, s.color);
  s.A.sit = 1;
  s.slot = i;
  s.box = packBox(s.color);
  // A ring on the floor: their colour while a question waits for them, amber while picked.
  s.ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.6, 32), new THREE.MeshBasicMaterial({ color: s.color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  s.ring.rotation.x = -Math.PI / 2; scene.add(s.ring);
}
// A column of light on whoever the current question needs.
const beamTex = canvasTex(16, 128, (g, w, h) => {
  const l = g.createLinearGradient(0, 0, 0, h); l.addColorStop(0, "rgba(255,255,255,0)"); l.addColorStop(1, "rgba(255,255,255,0.9)");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
});
const beam3 = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.62, 3.4, 24, 1, true), new THREE.MeshBasicMaterial({ map: beamTex, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
scene.add(beam3);
// Dust kicked up where a chair sets off and lands.
const puffs = [0, 1, 2, 3].map(() => {
  const p = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color: 0xb9bdc4, transparent: true, opacity: 0, depthWrite: false }));
  scene.add(p); return p;
});
let puffAt = 0;
function puff(x, z) {
  if (reduced) return;
  const p = puffs[puffAt++ % puffs.length];
  p.position.set(x, 0.25, z);
  tween(0.5, (u) => { p.scale.setScalar(0.4 + u * 0.9); p.material.opacity = 0.5 * (1 - u); });
}
// A streak behind a chair on the narrator's quick moves.
const streaks = STAFF.map((s) => {
  const k = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color: s.color, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  k.scale.set(1.6, 0.4, 1); scene.add(k); return k;
});

// Places the rooms for a layout and recomputes what hangs off them.
function layout3(mode) {
  const row = mode === "row";
  L3.mode = mode;
  const half = RW / 2 + LANE / 2;
  const o = row ? [X3 - half, 0] : [X3, -(RD / 2 + 0.9)], a = row ? [X3 + half, 0] : [X3, RD / 2 + 0.9];
  rooms[0].position.set(o[0], 0, o[1]); rooms[1].position.set(a[0], 0, a[1]);
  L3.aisle = [o[0], a[0]];
  for (let i = 0; i < 8; i++) {
    const [rx, rz] = i < 4 ? o : a;
    SLOTS[i][0] = rx + (i % 2 ? COL : -COL); SLOTS[i][1] = rz + ROWZ[(i >> 1) & 1];
  }
  R3.annexBack.visible = row; R3.parapet.visible = !row;
  for (const w of R3.annexSides) w.scale.y = row ? 1 : 0.3;
  R3.officeWin.visible = R3.officeSign.visible = row;
  R3.street.visible = row;
  R3.street.position.x = X3;
  // The fax tables sit at each room's lane side.
  R3.fax[0].position.set(RW / 2 - 0.45, 0, 1.95); R3.fax[0].rotation.y = 0;
  R3.fax[1].position.set(row ? -(RW / 2 - 0.45) : RW / 2 - 0.45, 0, row ? 1.95 : -1.95); R3.fax[1].rotation.y = 0;
  MFAX.set(o[0] + R3.fax[0].position.x, 1.05, o[1] + R3.fax[0].position.z);
  AFAX.set(a[0] + R3.fax[1].position.x, 1.05, a[1] + R3.fax[1].position.z);
  const P = row ? [[X3 - LANE / 2 + 0.12, 2.3], [X3 + LANE / 2 - 0.12, 2.3]] : [[X3 + RW / 2 + 0.3, -1.1], [X3 + RW / 2 + 0.3, 1.1]];
  // Cross-arms square to the wire; the lamp leans over the lane, or over the rooms' side.
  R3.poles.forEach((p, k) => { p.position.set(P[k][0], 0, P[k][1]); p.arm.rotation.y = row ? Math.PI / 2 : 0; });
  R3.lamp.rotation.y = row ? Math.PI / 2 : -Math.PI / 2;
  R3.lampPool.position.set(P[0][0] + (row ? 0.55 : -0.55), 0.03, P[0][1]);
  const p0 = new THREE.Vector3(P[0][0], 3.55, P[0][1]), p1 = new THREE.Vector3(P[1][0], 3.55, P[1][1]);
  const mid = p0.clone().lerp(p1, 0.5); mid.y = 3.15;
  R3.wire.geometry.dispose();
  R3.wire.geometry = new THREE.BufferGeometry().setFromPoints(new THREE.QuadraticBezierCurve3(p0, mid, p1).getPoints(30));
  const rise = (f, p) => new THREE.Vector3(lerp(f.x, p.x, 0.7), 2.9, lerp(f.z, p.z, 0.7));
  L3.faxRoute = new THREE.CatmullRomCurve3([MFAX.clone(), rise(MFAX, p0), p0.clone().setY(3.62), mid.clone().setY(3.3), p1.clone().setY(3.62), rise(AFAX, p1), AFAX.clone()]);
  R3.lane.position.set(X3, 0.012, 0);
  R3.lane.scale.set(row ? LANE : RW, row ? RD : 1.8, 1);
  R3.lane.material.map.repeat.set(row ? LANE : RW, row ? RD : 1.8);
  R3.puddle.position.set(row ? X3 - LANE / 2 + 0.75 : X3 + RW / 2 - 0.3, 0.02, row ? 1.4 : 0);
  // The board: on posts at the back of the lane, or leaning on the office's back wall.
  R3.board.position.set(X3, 0, row ? -RD / 2 + 0.35 : o[1] - RD / 2 + 0.22);
  R3.boardFace.position.y = row ? 3.0 : 3.5;
  R3.boardFace.rotation.x = row ? 0 : -0.16;
  R3.boardPosts.visible = row;
  L3.board.set(X3, R3.boardFace.position.y, R3.board.position.z + 0.08);
  L3.office.set(o[0], 4.5, o[1] - RD / 2 + 0.1);
  L3.annex.set(row ? a[0] : a[0] - 1.6, row ? 4.5 : 0.95, a[1] - RD / 2 + 0.1);
}
function placeAt(s, i) {
  const [x, z] = SLOTS[i];
  s.slot = i;
  s.chair.position.set(x, 0, z); s.chair.rotation.set(0, 0, 0);
  s.A.group.position.set(x, 0, z + 0.05); s.A.group.rotation.set(0, 0, 0); s.A.heading = 0;
  s.A.lift = 0; s.A.sit = 1; s.A.busy = false; s.A.walking = 0; s.A.cheer = false;
  s.box.group.position.set(x + 0.5, 1.15, z + 1.25); s.box.group.rotation.set(0, -0.2, 0);
  setChairLegs(s.legs, 0);
}
function pack(s, on) {
  const B = s.box;
  if (B.group.visible === on) return;
  B.group.visible = on;
  for (const f of B.flaps) f.rotation.x = -Math.sign(f.position.z) * 1.2;
  if (on) { B.group.scale.setScalar(0.01); tween(0.35, (u) => B.group.scale.setScalar(Math.max(0.01, backOut(u)))); }
}

const E3 = makeEnvelope(); E3.group.visible = false; E3.group.scale.setScalar(1.5);
// A band across the envelope in the colour of whoever it's for.
const E3band = new THREE.Mesh(new THREE.BoxGeometry(0.43, 0.016, 0.07), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.35 }));
E3band.position.set(0, 0.002, 0.09); E3.group.add(E3band);
const { group: page3, sheet: page3Sheet } = makeFaxPage();
// The letter that opens the night's second half: a fat envelope that drops onto the tally board, sealed in the
// colours of the two advisers it is about.
const letter3 = new THREE.Group(); letter3.visible = false; scene.add(letter3);
box(0.7, 0.08, 0.48, paperM, 0, 0, 0, letter3);
for (const [i, c] of [[0, 0xb05ac0], [1, 0xd2556b]]) {
  const st = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.01, 20), mat(c, { emissive: c, emissiveIntensity: 0.8 }));
  st.position.set(0.2 - i * 0.16, 0.045, -0.1); letter3.add(st);
}

function resetFloor3() {
  const want = pickLayout3();
  if (want !== L3.mode) layout3(want);
  STAFF.forEach((s, i) => { s.moves = 0; s.moving = false; s.ready = 0; s.A.watch = null; placeAt(s, i); pack(s, false); });
  E3.group.visible = false; page3.visible = false; letter3.visible = false;
  L3.need = null; L3.letterMark = false; L3.hot.clear();
}

// ---------- Camera ----------
const camera3 = { look: new THREE.Vector3(X3, 1.3, 0.6), offset: true, fog: [20, 46], par: 0.7, fov: 36 };

// Floor 3's camera looks down at a fixed pitch from whatever distance and screen offset fit the layout's outline
// into the part of the screen the HUD leaves clear: beside the side panel or under it, whichever shows it larger.
const fit3 = { key: "", at: 0, fov: 36, d: 20, snap: true, pos: new THREE.Vector3(), look: new THREE.Vector3(), off: [0, 0], tPos: new THREE.Vector3(), tLook: new THREE.Vector3(), tOff: [0, 0] };
const FIT3 = { row: { pitch: 0.44, pts: [] }, stack: { pitch: 0.82, pts: [] } };
for (const [m, hx, z0, z1, hb, hf] of [["row", 7.75, -2.6, RD / 2 + 1.5, 4.7, 1.8], ["stack", RW / 2 + 0.4, -RD - 0.9, RD + 0.9, 4.7, 1.6]])
  for (const sx of [-1, 1]) FIT3[m].pts.push(new THREE.Vector3(X3 + sx * hx, 0, z0), new THREE.Vector3(X3 + sx * hx, hb, z0), new THREE.Vector3(X3 + sx * hx, 0, z1), new THREE.Vector3(X3 + sx * hx, hf, z1));
const probe = new THREE.PerspectiveCamera(), pv = new THREE.Vector3();
function freeRects() {
  const W = innerWidth, H = innerHeight, narrow = W <= 720, g = 12;
  const top = document.querySelector(".topbar").getBoundingClientRect().bottom + 6;
  const sd = document.querySelector(".side").getBoundingClientRect();
  const bottom = H - (narrow ? 172 : 104);
  const under = { x: g, y: sd.bottom + 6, w: W - 2 * g, h: bottom - sd.bottom - 6 };
  return narrow ? [under] : [{ x: g, y: top, w: sd.left - 2 * g, h: bottom - top }, under];
}
function fitPoints(m, rect) {
  const { pitch, pts } = FIT3[m], W = innerWidth, H = innerHeight;
  probe.fov = fit3.fov; probe.aspect = W / H; probe.near = 0.1; probe.far = 300; probe.updateProjectionMatrix();
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
const bestFit = (m, rects) => rects.map((r) => fitPoints(m, r)).reduce((a, b) => (b.scale > a.scale ? b : a));
// One behind the other wins only where it shows the rooms clearly larger.
function pickLayout3() {
  const rects = freeRects();
  return bestFit("stack", rects).scale > bestFit("row", rects).scale * 1.12 ? "stack" : "row";
}
function cam3(dt) {
  const now = performance.now();
  if (!fit3.key || now - fit3.at > 200) {
    fit3.at = now;
    const rects = freeRects(), key = L3.mode + innerWidth + "x" + innerHeight + rects.map((r) => [r.x, r.y, r.w, r.h].map(Math.round).join(",")).join("|");
    if (key !== fit3.key) {
      fit3.key = key;
      const b = bestFit(L3.mode, rects);
      fit3.tPos.copy(b.c).addScaledVector(b.dir, b.d); fit3.tLook.copy(b.c); fit3.tOff = b.off; fit3.d = b.d;
    }
  }
  const k = fit3.snap || reduced ? 1 : 1 - Math.exp(-dt * 4);
  fit3.snap = false;
  fit3.pos.lerp(fit3.tPos, k); fit3.look.lerp(fit3.tLook, k);
  fit3.off[0] += (fit3.tOff[0] - fit3.off[0]) * k; fit3.off[1] += (fit3.tOff[1] - fit3.off[1]) * k;
  camera.setViewOffset(innerWidth, innerHeight, fit3.off[0], fit3.off[1], innerWidth, innerHeight);
  camera3.look.copy(fit3.look);
  scene.fog.near = fit3.d * 0.95; scene.fog.far = fit3.d * 2.4;
  return fit3.pos;
}

camera3.place = cam3;

// Screen boxes {x (centre), y (top), w, h} nudged apart vertically; a few passes settle small clusters.
function declutter(items, gap = 4) {
  for (let pass = 0; pass < 4; pass++) for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
    const a = items[i], b = items[j];
    if (Math.abs(a.x - b.x) * 2 >= a.w + b.w + gap) continue;
    const [t, u] = a.y <= b.y ? [a, b] : [b, a], over = t.y + t.h + gap - u.y;
    if (over > 0) { t.y -= over / 2; u.y += over / 2; }
  }
}
// Element sizes, read once after a change rather than every frame.
const tagSize = new Map();
const sizeOf = (el) => { let s = tagSize.get(el); if (!s) { s = [el.offsetWidth, el.offsetHeight]; tagSize.set(el, s); } return s; };

// The reveal's labels. This floor places them itself, in dom3, to keep them apart.
const anchors3 = [
  [$("aDesk3"), () => L3.office],
  [$("aAnnex3"), () => L3.annex],
  [$("aBoard"), () => L3.board.clone().setY(L3.board.y + BH / 2 + 0.12)],
  [$("aMover"), () => up((L3.mover || STAFF[4]).A.group, 2.0)],
];

// Floor 3, before each render: the rings and the column of light, the cat's breathing, and a queued chair's legs
// peeking out and jittering, ready to run.
function scene3(t, dt) {
  const k = reduced ? 1 : Math.min(1, dt * 10), pulse = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(t * 7);
  for (const [i, s] of STAFF.entries()) {
    const need = L3.need === s, picked = sel === i && tagsLive;
    s.ring.position.set(s.chair.position.x, 0.11, s.chair.position.z + 0.35);
    s.ring.material.color.setHex(picked ? 0xe8a33d : s.color);
    s.ring.material.opacity = need ? 0.6 + 0.35 * pulse : picked ? 0.9 : s.queued ? 0.45 : 0;
    if (s.moving) continue;
    const r0 = s.ready;
    s.ready += ((s.queued ? 1 : 0) - s.ready) * k;
    if (s.ready < 0.002 && r0 < 0.002) continue;
    setChairLegs(s.legs, 0.22 * s.ready);
    for (const l of s.legs) l.pivot.rotation.x = reduced ? 0 : Math.sin(t * 26 + l.phase) * 0.5 * s.ready;
    s.chair.position.x = SLOTS[s.slot][0] + (reduced ? 0 : Math.sin(t * 41 + i) * 0.018 * s.ready);
  }
  const n = L3.need;
  beam3.material.opacity = n ? 0.26 + 0.14 * pulse : 0;
  if (n) { beam3.position.set(n.A.group.position.x, 1.75, n.A.group.position.z + 0.1); beam3.material.color.setHex(n.color); }
  if (!reduced) R3.cat.scale.y = 0.62 + Math.sin(t * 1.7) * 0.025;
}
// Floor 3, after each render: the name tags, and the reveal's labels, kept from covering one another. Side by
// side, the front row's tags sit on the floor in front of their desks and the back row's above their heads; one
// behind the other, all above their heads.
function dom3() {
  if (!tagsBox.hidden) {
    const items = STAFF.map((s) => {
      const front = L3.mode === "row" && s.slot % 4 >= 2 && !s.moving, g = s.A.group.position;
      const [x, y] = front ? toScreen(v.set(g.x, 0.1, g.z + 1.95)) : toScreen(v.set(g.x, 2.02 + s.A.lift, g.z));
      const [w, h] = sizeOf(s.tag);
      return { s, x, w, h, y: front ? y : y - h };
    });
    declutter(items);
    for (const it of items) it.s.tag.style.transform = `translate(${Math.round(Math.min(innerWidth - it.w - 4, Math.max(4, it.x - it.w / 2)))}px, ${Math.round(it.y)}px)`;
  }
  const labs = [];
  for (const [el, at] of anchors3) {
    if (!el.classList.contains("on")) continue;
    const [x, y] = toScreen(at()), [w, h] = sizeOf(el);
    labs.push({ el, x, w, h: h + 6, y: y - h - 6 });
  }
  declutter(labs, 6);
  for (const it of labs) it.el.style.transform = `translate(${Math.round(Math.min(innerWidth - it.w / 2 - 6, Math.max(it.w / 2 + 6, it.x)))}px, ${Math.round(it.y + it.h)}px) translate(-50%, -100%)`;
}

// The night: who each question needs, by STAFF index. The first twelve spread over the office; then a letter
// arrives and the questions turn legal.
const NIGHT = [0, 2, 1, 0, 2, 3, 0, 2, 6, 0, 2, 1, 4, 5, 4, 4, 5, 0, 4, 5, 2, 4, 5, 4, 0, 5, 4, 5];
const SHIFT_AT = 12;
// Seconds on the stopwatch. A desk answers in LOCAL; a fax goes out, is thought over in the annex, and comes back.
// The player's swap stops the office for SWAP; the narrator's round holds it for ROUND, its packing done behind.
const T3 = { LOCAL: 0.5, FAX_OUT: 0.35, FAX_THINK: 0.9, SWAP: 1.0, ROUND: 0.1 };
// The narrator's chairs move in ZIP: the round's ROUND on the clock, the rest during the next question.
const ZIP = 0.3;
// The swap rule of the engine (core/models/qwen4exp/swap.cpp), at the scale of eight advisers: a round every 3
// questions and one after the letter; counts fade by 0.8 a round; a newcomer replaces the quietest resident only
// with a count of at least 2 and over 1.5 times theirs plus 1; a round's choice moves in at the next round.
const ROUND_EVERY = 3, FADE = 0.8, LETTER_WEIGHT = 3;
const counts = new Array(STAFF.length).fill(0);
const inOffice = (s) => s.slot < 4;

function chooseSwaps() {
  const ins = STAFF.filter((s) => !inOffice(s)).sort((a, b) => counts[STAFF.indexOf(b)] - counts[STAFF.indexOf(a)]);
  const outs = STAFF.filter(inOffice).sort((a, b) => counts[STAFF.indexOf(a)] - counts[STAFF.indexOf(b)]);
  const out = [];
  for (let n = 0; n < ins.length && n < outs.length; n++) {
    const ui = counts[STAFF.indexOf(ins[n])], uo = counts[STAFF.indexOf(outs[n])];
    if (ui < 2 || ui < 1.5 * uo + 1) break;
    out.push([ins[n], outs[n]]);
  }
  return out;
}
// The narrator's night without the animation: its total, and the time after each question.
function engineNight() {
  const saved = STAFF.map((s) => s.slot), c0 = counts.slice();
  counts.fill(0);
  let t = 0, since = 0, pending = [];
  const after = [];
  NIGHT.forEach((e, i) => {
    if (i === SHIFT_AT) { counts[4] += LETTER_WEIGHT; counts[5] += LETTER_WEIGHT; since = ROUND_EVERY - 1; }
    t += inOffice(STAFF[e]) ? T3.LOCAL : T3.FAX_OUT * 2 + T3.FAX_THINK;
    counts[e] += 1;
    if (++since >= ROUND_EVERY) {
      since = 0;
      if (pending.length) { t += T3.ROUND; for (const [a, b] of pending) [a.slot, b.slot] = [b.slot, a.slot]; }
      pending = chooseSwaps();
      for (let k = 0; k < counts.length; k++) counts[k] *= FADE;
    }
    after.push(t);
  });
  STAFF.forEach((s, i) => { s.slot = saved[i]; });
  counts.splice(0, counts.length, ...c0);
  return { total: Math.round(t * 10) / 10, after };
}
const ENGINE = engineNight();

// Real-time tweens: the shift's pace is the stopwatch's, with or without reduced motion.
function rt(seconds, fn) {
  return new Promise((done) => {
    const t0 = performance.now();
    const tick = (now) => { const u = Math.min(1, Math.max(0, (now - t0) / (seconds * 1000))); fn(u); if (u < 1) requestAnimationFrame(tick); else done(); };
    requestAnimationFrame(tick);
  });
}
const until = (cond) => new Promise((r) => { const c = () => cond() ? r() : setTimeout(c, 80); c(); });

// The board in two columns, desks and annex, each row a bar of its adviser's colour. An annex row in L3.hot is
// ringed, with an arrow; `packing` are the ones the narrator has boxed up. Tags get the same bars and arrows.
function drawBoard(packing = []) {
  const c = boardTex.image, g = c.getContext("2d"), w = c.width, h = c.height;
  g.fillStyle = "#1c2925"; g.fillRect(0, 0, w, h);
  g.fillStyle = "rgba(236,236,226,0.035)";
  for (let k = 0; k < 9; k++) { g.beginPath(); g.ellipse((k * 211) % w, 110 + (k * 137) % (h - 140), 160, 40, (k % 3) * 0.3, 0, Math.PI * 2); g.fill(); }
  const chalk = "rgba(236,236,226,0.94)", font = (wt, px) => `${wt} ${px}px 'Barlow Condensed', 'Arial Narrow', sans-serif`;
  g.textBaseline = "alphabetic";
  g.fillStyle = chalk; g.font = font(600, 58); g.fillText("WHO'S BEEN ASKED", 30, 70);
  g.fillStyle = "rgba(236,236,226,0.6)"; g.font = font(500, 38);
  g.fillText("AT A DESK", 30, 132); g.fillText("IN THE ANNEX", 540, 132);
  g.fillRect(511, 104, 3, h - 130);
  const cols = [STAFF.filter(inOffice), STAFF.filter((s) => !inOffice(s))].map((l) => l.sort((a, b) => a.slot - b.slot));
  const MAXC = 7, BAR = 300;
  cols.forEach((list, ci) => list.forEach((s, r) => {
    const i = STAFF.indexOf(s), x0 = ci ? 540 : 30, y0 = 214 + r * 112, n = Math.min(counts[i], MAXC), hot = L3.hot.has(s);
    if (hot) {
      g.strokeStyle = "rgba(232,163,61,0.95)"; g.lineWidth = 6;
      g.beginPath(); g.ellipse(x0 + 222, y0 - 2, 232, 52, -0.01, 0, Math.PI * 2); g.stroke();
    }
    g.fillStyle = hex(s.color);
    g.beginPath(); g.arc(x0 + 16, y0 - 34, 16, 0, Math.PI * 2); g.fill();
    g.fillStyle = chalk; g.font = font(600, 50);
    g.fillText(s.name.toUpperCase(), x0 + 42, y0 - 16);
    // The bar, laid on in chalky strokes; the faint track behind it is the scale.
    const len = BAR * n / MAXC, bx = x0 + 10, by = y0 - 4;
    g.fillStyle = "rgba(236,236,226,0.09)"; g.fillRect(bx, by, BAR, 36);
    g.fillStyle = hex(s.color);
    for (let k = 0; k < 4; k++) { g.globalAlpha = 0.6 + k * 0.1; g.fillRect(bx, by + k * 9, Math.max(0, len - (k % 2) * 7), 7); }
    g.globalAlpha = 1;
    if (L3.letterMark && (i === 4 || i === 5)) { g.fillStyle = "#f2d38a"; g.font = font(700, 46); g.fillText("+3", bx + BAR + 18, by + 34); }
    else if (hot) { g.fillStyle = "rgba(232,163,61,1)"; g.beginPath(); g.moveTo(bx + BAR + 34, by - 4); g.lineTo(bx + BAR + 52, by + 34); g.lineTo(bx + BAR + 16, by + 34); g.fill(); }
    if (packing.includes(s)) {
      const px = x0 + 50 + g.measureText(s.name.toUpperCase()).width + 30;
      g.fillStyle = "#a07a4e"; g.fillRect(px, y0 - 58, 46, 40);
      g.strokeStyle = "#5a4024"; g.lineWidth = 4; g.strokeRect(px, y0 - 58, 46, 40); g.beginPath(); g.moveTo(px, y0 - 46); g.lineTo(px + 46, y0 - 46); g.stroke();
    }
    s.bar.style.width = (100 * n / MAXC).toFixed(1) + "%";
    s.tag.classList.toggle("hot", hot);
  }));
  boardTex.needsUpdate = true;
}
// Who the rule would move in now, from the counts as they stand before a round's fade.
const markHot = () => { L3.hot = new Set(chooseSwaps().map(([a]) => a)); };

// Name tags on the advisers: buttons while the player runs the desks. Each shows the name, the topic on wide
// screens, and the tally as a bar along its foot.
const tagsBox = $("tags");
STAFF.forEach((s, i) => {
  const b = document.createElement("button");
  b.type = "button"; b.className = "tag";
  b.style.setProperty("--c", hex(s.color));
  b.innerHTML = `<i></i>${s.name}<small>${s.topic}</small><span class="n"></span>`;
  b.setAttribute("aria-label", `${s.name}, ${s.topic}`);
  b.addEventListener("click", () => tapTag(i));
  s.tag = b; s.bar = b.querySelector(".n"); tagsBox.appendChild(b);
});
let tagsLive = false, sel = -1, queue = [], react = null, nightOn = false;
const swapping = new Set();
function showTags(on, live) {
  tagsBox.hidden = !on; tagsLive = !!live; sel = -1; queue = []; swapping.clear();
  tagsBox.classList.toggle("live", !!live);
  paintTags();
}
const queuedHas = (i) => queue.some(([a, b]) => a === i || b === i);
function paintTags() {
  const picking = tagsLive && sel >= 0;
  tagsBox.classList.toggle("picking", picking);
  STAFF.forEach((s, i) => {
    s.queued = queuedHas(i);
    s.tag.classList.toggle("sel", sel === i);
    s.tag.classList.toggle("queued", s.queued);
    s.tag.classList.toggle("moving", swapping.has(i));
    s.tag.classList.toggle("can", picking && sel !== i && !s.queued && !swapping.has(i) && inOffice(STAFF[sel]) !== inOffice(s));
    s.tag.disabled = !tagsLive;
  });
  hint3();
  tagSize.clear();
}
// The line under the scene says what a tap will do next.
function hint3() {
  const h = $("hint3");
  h.hidden = !tagsLive;
  if (!tagsLive) return;
  const nm = (k) => `<b style="color:${hex(STAFF[k].color)}">${STAFF[k].name}</b>`;
  if (swapping.size) h.innerHTML = "The office is on hold while they move";
  else if (sel >= 0) h.innerHTML = `Now pick someone ${inOffice(STAFF[sel]) ? "in the annex" : "in the office"} for ${nm(sel)}`;
  else if (queue.length) h.innerHTML = `${queue.map(([a, b]) => `${nm(a)} ⇄ ${nm(b)}`).join(", ")} after this question: the office stops ${queue.length * T3.SWAP} s. Tap a name again to cancel`;
  else h.innerHTML = "Pick a name in the annex, then one in the office, to swap them";
}
function tapTag(i) {
  if (!tagsLive || swapping.has(i)) return;
  const q = queue.findIndex(([a, b]) => a === i || b === i);
  if (q >= 0) queue.splice(q, 1);   // a queued name cancels its swap
  else if (sel < 0 || sel === i) sel = sel === i ? -1 : i;
  else if (inOffice(STAFF[sel]) === inOffice(STAFF[i])) sel = i;
  else { queue.push([sel, i]); if (react) react.queued(sel, i); sel = -1; }
  paintTags();
}

// A chair moves from slot `from` to slot `to`, out along its room's aisle, across, and in; k picks one of two
// tracks, so the two chairs of a swap pass each other. The player's swap: legs pop out, the chair scurries and lands,
// its rider's arms up. The narrator's: the packed chair zips across with the box on the rider's lap.
function route3(from, to, k) {
  const [x0, z0] = SLOTS[from], [x1, z1] = SLOTS[to];
  if (L3.mode === "row") {
    const g0 = L3.aisle[from < 4 ? 0 : 1], g1 = L3.aisle[to < 4 ? 0 : 1], lz = RD / 2 + 0.65 + 0.5 * k;
    return [[x0, z0], [g0, z0], [g0, lz], [g1, lz], [g1, z1], [x1, z1]];
  }
  const tx = X3 + (k ? 0.28 : -0.28);
  return [[x0, z0], [tx, z0], [tx, z1], [x1, z1]];
}
const LEG3 = 0.78;
async function scoot(s, from, to, style, k, clockFn) {
  const pts = route3(from, to, k), lens = [];
  let total = 0;
  for (let n = 1; n < pts.length; n++) { const d = Math.hypot(pts[n][0] - pts[n - 1][0], pts[n][1] - pts[n - 1][1]); lens.push(d); total += d; }
  const player = style === "player", streak = streaks[STAFF.indexOf(s)];
  let h = 0, last = performance.now(), px = pts[0][0], pz = pts[0][1];
  s.moving = true; s.ready = 0; s.A.busy = false; s.A.watch = null;
  if (player) puff(pts[0][0], pts[0][1]);
  await rt(player ? T3.SWAP : ZIP, (u) => {
    const now = performance.now(), dt = Math.min(0.05, (now - last) / 1000); last = now;
    // Player: legs out in the first 16%, run to 86%, land. Narrator: one smooth glide.
    const run = player ? Math.min(1, Math.max(0, (u - 0.16) / 0.7)) : u;
    let d = ease(run) * total, n = 0;
    while (n < lens.length - 1 && d > lens[n]) { d -= lens[n]; n++; }
    const [ax, az] = pts[n], [bx, bz] = pts[n + 1], f = lens[n] ? Math.min(1, d / lens[n]) : 1;
    const x = lerp(ax, bx, f), z = lerp(az, bz, f), moving = run > 0 && run < 1;
    const want = moving ? Math.atan2(bx - ax, bz - az) : 0;
    h += reduced ? wrapAngle(want - h) : wrapAngle(want - h) * Math.min(1, dt * 16);
    const speed = Math.hypot(x - px, z - pz) / Math.max(dt, 0.001); px = x; pz = z;
    let lift = 0, lean = 0;
    if (player) {
      const len = LEG3 * (u < 0.16 ? backOut(u / 0.16) : u > 0.86 ? 1 - ease((u - 0.86) / 0.14) : 1);
      const ph = ease(run) * total * 8, amp = reduced ? 0 : Math.min(1, run * 8, (1 - run) * 8);
      setChairLegs(s.legs, len);
      for (const l of s.legs) l.pivot.rotation.x = Math.sin(ph + l.phase) * 0.6 * amp;
      lift = Math.max(0, len - 0.44) + Math.abs(Math.sin(ph)) * 0.06 * amp;
      lean = reduced ? 0 : 0.12 * amp + (u < 0.16 ? -0.15 * Math.sin(u / 0.16 * Math.PI) : 0);
      s.A.cheer = u > 0.06 && u < 0.9;
    } else {
      lift = 0.08 * Math.sin(u * Math.PI);
      lean = reduced ? 0 : 0.2 * Math.sin(u * Math.PI);
      if (!reduced) {
        streak.position.set(x - Math.sin(h) * 0.7, 0.75, z - Math.cos(h) * 0.7);
        streak.material.opacity = Math.min(0.55, speed / 40);
        // The streak lies along the motion as seen on screen.
        const a = v.set(x, 0.75, z).project(camera), b2 = new THREE.Vector3(x - Math.sin(h), 0.75, z - Math.cos(h)).project(camera);
        streak.material.rotation = Math.atan2((a.y - b2.y) / camera.aspect, a.x - b2.x);
      }
      s.box.group.position.set(x + Math.sin(h) * 0.25, 0.92 + lift, z + Math.cos(h) * 0.25);
      s.box.group.rotation.set(0, h, 0);
    }
    s.chair.position.set(x, lift, z); s.chair.rotation.set(lean, h, 0);
    s.A.group.position.set(x + Math.sin(h) * 0.05, 0, z + Math.cos(h) * 0.05);
    s.A.group.rotation.set(lean, h, 0); s.A.heading = h; s.A.lift = lift;
    if (clockFn) clockFn(u);
  });
  streak.material.opacity = 0;
  pack(s, false);
  placeAt(s, to);
  s.moving = false;
  if (player) { puff(pts[pts.length - 1][0], pts[pts.length - 1][1]); if (!reduced) view.shake = 0.03; }
}
// The pair's places swap the moment the swap starts; their chairs catch up.
async function swapPair(a, b, style, clockFn) {
  const ia = a.slot, ib = b.slot;
  a.moves++; b.moves++;
  a.slot = ib; b.slot = ia;
  await Promise.all([scoot(a, ia, ib, style, 0, clockFn), scoot(b, ib, ia, style, 1)]);
}
// While the player's swap stops the office, everyone else downs tools and watches the chairs go.
function hold3(on, movers) {
  document.querySelector(".watch").classList.toggle("hold", on);
  for (const s of STAFF) if (!movers.includes(s)) { s.A.watch = on ? movers[0].chair.position : null; if (on) s.A.busy = false; }
}

// One question, from the envelope's drop to its stamp. The clock counts st.mov and st.th.
async function answer(e, st) {
  const s = STAFF[e], [x, z] = SLOTS[s.slot];
  E3.stamp.material.color.setHex(s.color); E3.stamp.material.emissive.setHex(s.color);
  E3band.material.color.setHex(s.color); E3band.material.emissive.setHex(s.color);
  E3.stamp.visible = false; E3.group.visible = true;
  s.tag.classList.add("need"); L3.need = s;
  const m0 = st.mov, t0 = st.th;
  if (inOffice(s)) {
    const from = new THREE.Vector3(x - 0.3, 4.6, z + 1.25), to = new THREE.Vector3(x - 0.3, 1.04, z + 1.25);
    s.A.busy = true;
    await rt(T3.LOCAL, (u) => {
      E3.group.position.lerpVectors(from, to, ease(Math.min(1, u / 0.35)));
      E3.stamp.visible = u > 0.75;
      st.th = t0 + T3.LOCAL * u; setClock(st.mov, st.th);
    });
    st.th = t0 + T3.LOCAL;
    st.faxRun = 0; st.localRun++;
  } else {
    const from = new THREE.Vector3(MFAX.x, 4.6, MFAX.z), to = MFAX.clone().add(new THREE.Vector3(0, 0.2, 0));
    page3.visible = false;
    wireMat.color.setHex(0xe8a33d);
    const place = (u) => {
      const end = Math.min(1, u * 8, (1 - u) * 8);
      page3.position.copy(L3.faxRoute.getPointAt(u));
      page3Sheet.position.y = -0.3 * end; page3Sheet.scale.setScalar(0.3 + 0.7 * end);
      page3Sheet.rotation.set(-0.2 + 0.15 * Math.sin(u * 9), 0.35 + Math.sin(u * 7) * 0.3, Math.sin(u * 13) * 0.12);
    };
    await rt(T3.FAX_OUT, (u) => {
      const a = Math.min(1, u / 0.3);
      E3.group.position.lerpVectors(from, to, ease(a));
      if (u > 0.3) { E3.group.visible = false; page3.visible = true; place((u - 0.3) / 0.7); }
      st.mov = m0 + T3.FAX_OUT * u; setClock(st.mov, st.th);
    });
    page3.visible = false; wireMat.color.setHex(0x14171c);
    s.A.busy = true;
    await rt(T3.FAX_THINK, (u) => { st.th = t0 + T3.FAX_THINK * u; setClock(st.mov, st.th); });
    s.A.busy = false;
    wireMat.color.setHex(0xe8a33d); page3.visible = true;
    await rt(T3.FAX_OUT, (u) => { place(1 - u); st.mov = m0 + T3.FAX_OUT * (1 + u); setClock(st.mov, st.th); });
    page3.visible = false; wireMat.color.setHex(0x14171c);
    st.mov = m0 + 2 * T3.FAX_OUT; st.th = t0 + T3.FAX_THINK;
    E3.group.position.copy(to); E3.group.visible = true; E3.stamp.visible = true;
    await rt(0.12, () => {});
    st.faxes++; st.faxRun++; st.localRun = 0; st.faxTimes.push(performance.now() / 1000);
    if (e === 4 || e === 5) st.legalFaxes++;
  }
  s.A.busy = false;
  s.tag.classList.remove("need"); L3.need = null;
  E3.group.visible = false;
}
// The letter drops onto the board; Margaret's and Hugh's counts jump and the board flashes. Not on the clock.
async function dropLetter() {
  const B = L3.board, from = B.clone().add(new THREE.Vector3(0, 2.6, 1.4)), to = B.clone().add(new THREE.Vector3(0, 0, 0.12));
  letter3.visible = true;
  await rt(0.6, (u) => {
    letter3.position.lerpVectors(from, to, ease(u));
    letter3.rotation.set(Math.PI / 2 * ease(u), 0, reduced ? 0 : Math.sin(u * 6) * 0.2);
    letter3.scale.setScalar(1 - 0.6 * Math.max(0, u - 0.7) / 0.3);
  });
  letter3.visible = false;
  counts[4] += LETTER_WEIGHT; counts[5] += LETTER_WEIGHT;
  L3.letterMark = true;
  markHot(); drawBoard();
  await rt(0.6, (u) => { boardMat.emissiveIntensity = 0.32 + (reduced ? 0.3 : 0.9) * (1 - u); });
  boardMat.emissiveIntensity = 0.32;
}

// The night, run by the player ("player") or by the narrator ("engine"). Resolves to the night's counters.
async function night(mode, hooks = {}) {
  resetFloor3();
  nightOn = true;
  counts.fill(0); drawBoard();
  const st = { mov: 0, th: 0, since: 0, pending: [], faxes: 0, legalFaxes: 0, faxRun: 0, localRun: 0, faxTimes: [], i: 0, swaps: 0 };
  const zips = [];
  setClock(0, 0);
  showTags(true, mode === "player");
  for (let i = 0; i < NIGHT.length; i++) {
    st.i = i;
    if (i === SHIFT_AT) {
      if (hooks.letter) await hooks.letter(st);
      await dropLetter();
      st.since = ROUND_EVERY - 1;
      if (hooks.afterLetter) hooks.afterLetter(st);
    }
    if (i === SHIFT_AT + 3) { L3.letterMark = false; drawBoard(st.pending.flat()); }
    while (mode === "player" && queue.length) {
      const [ia, ib] = queue.shift(), a = STAFF[ia], b = STAFF[ib];
      if (inOffice(a) === inOffice(b)) { paintTags(); continue; }
      swapping.add(ia); swapping.add(ib); paintTags();
      const m0 = st.mov;
      hold3(true, [a, b]);
      await swapPair(a, b, "player", (u) => { st.mov = m0 + T3.SWAP * u; setClock(st.mov, st.th); });
      hold3(false, [a, b]);
      st.mov = m0 + T3.SWAP; st.swaps++;
      swapping.delete(ia); swapping.delete(ib);
      markHot(); drawBoard();
      if (hooks.swapped) hooks.swapped(a, b, st);
      paintTags();
    }
    await answer(NIGHT[i], st);
    counts[NIGHT[i]] += 1;
    markHot();
    if (++st.since >= ROUND_EVERY) {
      st.since = 0;
      if (mode === "engine") {
        if (st.pending.length) {
          if (hooks.beforeApply) await hooks.beforeApply(st.pending, st);
          const m0 = st.mov, moving = st.pending;
          // The boxes close and the chairs set off: the round holds the office for ROUND; they land during the
          // next question.
          for (const s of moving.flat()) tween(0.12, (u) => { for (const f of s.box.flaps) f.rotation.x = -Math.sign(f.position.z) * lerp(1.2, Math.PI, u); });
          zips.push(Promise.all(moving.map(([a, b]) => swapPair(a, b, "engine"))));
          await rt(T3.ROUND, (u) => { st.mov = m0 + T3.ROUND * u; setClock(st.mov, st.th); });
          st.mov = m0 + T3.ROUND;
          if (hooks.applied) hooks.applied(moving, st);
          markHot();
        }
        st.pending = chooseSwaps();
        for (const s of st.pending.flat()) pack(s, true);
        if (st.pending.length && hooks.chose) hooks.chose(st.pending, st);
      }
      for (let k = 0; k < counts.length; k++) counts[k] *= FADE;
    }
    drawBoard(st.pending.flat());
    if (hooks.answered) hooks.answered(i, st);
  }
  await Promise.all(zips);
  showTags(false);
  nightOn = false;
  return st;
}

// The narrator's remarks while the player runs the night: each at most once, one at most every 5.5 s.
function makeReact() {
  const now = () => performance.now() / 1000;
  const R = { used: new Set(), last: -1e9, talking: false, shiftT: null, legalIn: null, swapTs: [], lastSwap: now(), hold: false };
  R.say = (id, force) => {
    if (R.talking || R.used.has(id) || R.hold) return false;
    if (!force && now() - R.last < 5.5) return false;
    R.used.add(id); R.talking = true; R.last = now();
    speak(id).then(() => {
      R.talking = false;
      setTimeout(() => { if (!R.talking && !speaking() && view.floor === 3) $("caption").hidden = true; }, 900);
    });
    return true;
  };
  R.any = (ids, force) => {
    const free = ids.filter((x) => !R.used.has(x));
    return free.length ? R.say(free[Math.floor(Math.random() * free.length)], force) : false;
  };
  R.queued = (a, b) => {
    const t = now();
    R.swapTs.push(t); R.lastSwap = t;
    const A = STAFF[a], B = STAFF[b], incoming = inOffice(A) ? B : A;
    if (R.shiftT && (incoming === STAFF[4] || incoming === STAFF[5]) && !R.legalIn) R.legalIn = t;
    if (STAFF[4].moves >= 2 && (A === STAFF[4] || B === STAFF[4])) { R.say("f3thrash2"); return; }
    if (R.swapTs.filter((x) => t - x < 6).length >= 3) R.any(["f3thrash1", "f3thrash3"]);
  };
  R.swapped = (a, b) => {
    // Back in the seat they started the night in, after more than one move.
    const back = [a, b].find((s) => s.slot === STAFF.indexOf(s) && s.moves >= 2 && s.he);
    if (back) R.say("f3thrash4");
    if (inOffice(STAFF[4]) && inOffice(STAFF[5]) && R.shiftT && now() - R.shiftT < 9) R.say("f3well1");
  };
  R.tick = (st) => {
    if (R.talking || R.hold) return;
    const t = now(), recentFax = st.faxTimes.filter((x) => t - x < 8).length;
    if (st.i >= SHIFT_AT - 2 && !R.shiftT) return;  // the floor is kept clear for the letter
    // Once the shift line is over, a player who has not reached for the legal advisers hears the nudge first.
    if (R.lookDue()) { if (t - R.shiftT > 3.5) R.say("f3look", true); return; }
    if (t - R.lastSwap > 8 && recentFax >= 2) R.any(["f3idle1", "f3idle2", "f3idle3", "f3idle4", "f3idle5"]);
  };
  R.lookDue = () => R.shiftT && !R.legalIn && !R.used.has("f3look") && !inOffice(STAFF[4]) && !inOffice(STAFF[5]);
  R.answered = (i, st) => {
    if (R.talking) return;
    if (i === NIGHT.length - 3) { R.any(["f3final1", "f3final2"], true); return; }
    if (R.lookDue()) return;
    if (st.faxes === 6 && st.faxRun > 0) { R.say("f3behind1"); return; }
    if (st.faxRun >= 3) { R.say("f3behind2"); return; }
    if (R.shiftT && st.legalFaxes >= 2 && st.faxRun > 0) { R.say("f3behind3"); return; }
    if (R.shiftT && st.legalFaxes >= 4 && st.faxRun > 0) { R.say("f3behind4"); return; }
    if (R.shiftT && st.localRun >= 5) { R.say("f3well4"); return; }
    if (R.shiftT && st.localRun >= 4) { R.say("f3well2"); return; }
    if (i >= SHIFT_AT + 6 && st.mov + st.th <= ENGINE.after[i]) R.say("f3well3");
  };
  return R;
}

let f3Tries = 0, f3Replayed = false;
async function f3Intro() {
  f3Tries = 0; f3Replayed = false;
  await speak("f3intro");
  $("caption").hidden = true;
  await f3Shift();
}
async function f3Shift() {
  clearStage();
  f3Tries++;
  const R = react = makeReact();
  let cur = { i: 0, faxTimes: [] };
  const timer = setInterval(() => R.tick(cur), 250);
  const st = await night("player", {
    letter: async () => { await until(() => !R.talking); },
    afterLetter: (s) => {
      R.shiftT = performance.now() / 1000;
      // A legal adviser already at a desk means the player got there before the letter: he says so instead.
      if (inOffice(STAFF[4]) || inOffice(STAFF[5])) { R.used.add("f3noticed"); R.say("f3cheat", true); return; }
      R.say("f3shift", true);
      // Spoken once the line about the letter is over, if the player moved the legal advisers in quickly.
      until(() => !R.talking).then(() => { if (R.legalIn && R.legalIn - R.shiftT < 4) R.say("f3noticed", true); });
    },
    swapped: R.swapped,
    answered: (i, s) => { cur = s; R.answered(i, s); },
  });
  clearInterval(timer);
  R.hold = true;
  await until(() => !R.talking);
  react = null;
  const mine = { label: f3Tries > 1 ? `Your night (${f3Tries})` : "Your night", fetch: st.mov, think: st.th };
  addHistory(mine);
  const T = Math.round((st.mov + st.th) * 10) / 10;
  if (!f3Replayed) {
    await speak(T <= ENGINE.total + 1 ? "f3afterAhead" : "f3afterBehind");
    await f3Replay();
  }
  const beat = T < ENGINE.total;
  // Losing with more than the two moves that matter is the stopping; with fewer, it was being late.
  await speak(beat ? "f3lose" : st.swaps > 2 ? "f3win" : "f3winLate");
  if (beat) findEnding("beat", "Employee of the Month");
  else findEnding("second", "Second Place");
  choose([["What just happened?", reveal3]]);
}
// The narrator's night. His lines wait for their moments: the night pauses for him at the letter and before his
// move, never on the clock, and "there they go" starts as the chairs do.
async function f3Replay() {
  f3Replayed = true;
  let narr = speak("f3p1");
  const st = await night("engine", {
    letter: async () => { await narr; },
    chose: () => { narr = narr.then(() => speak("f3p2")); },
    // While he finishes explaining, the office keeps typing; nothing is on the clock.
    beforeApply: async () => {
      const office = STAFF.filter(inOffice);
      for (const s of office) s.A.busy = true;
      await narr;
      for (const s of office) s.A.busy = false;
      narr = speak("f3p3");
    },
  });
  await narr;
  addHistory({ label: "The narrator", fetch: st.mov, think: st.th });
}
async function f3Again() {
  clearStage();
  await speak("f3again");
  $("caption").hidden = true;
  await f3Shift();
}
async function reveal3() {
  // The mover label follows someone the last night moved into the office, else Margaret.
  L3.mover = STAFF.find((s, i) => i >= 4 && inOffice(s)) || STAFF[4];
  labels(true);
  const more = !(endings[3].has("beat") && endings[3].has("second"));
  $("nextNote").textContent = config.next + (more ? " One ending on this floor is still hidden." : " Both endings found.");
  revealButtons([]);
  await speak("f3reveal", (line) => { if (line === "f3r2") $("reveal").hidden = false; });
  $("reveal").hidden = false;
  $("caption").hidden = true;
  revealButtons([
    [endings[3].has("beat") ? "Run the night again" : "Try to beat him", f3Again, !endings[3].has("beat")],
    ["Replay floor 3", () => { endings[3].clear(); goFloor(3); }],
    ["Back to floor 2", () => goFloor(2)],
  ]);
}

resetFloor3();
drawBoard();

export default {
  ...config, camera: camera3, start: f3Intro,
  enter() { fit3.snap = true; fit3.key = ""; resetFloor3(); counts.fill(0); drawBoard(); },
  frame(t, dt) {
    for (const s of STAFF) poseAdviser(s.A, t, dt);
    scene3(t, dt);
  },
  afterRender: dom3,
  onLabels() { tagSize.clear(); },
  // Between nights a turned or resized screen may suit the other layout.
  resized() {
    tagSize.clear(); fit3.key = "";
    if (view.floor === 3 && !nightOn) {
      const want = pickLayout3();
      if (want !== L3.mode) { layout3(want); STAFF.forEach((s) => placeAt(s, s.slot)); fit3.snap = true; }
    }
  },
};
