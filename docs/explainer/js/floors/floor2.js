// Floor 2, the annex, around x = X2. An expert the card lacks is worked out where it sits: move the question, not the knowledge.
import { $, box, camera, canvasTex, halo, keyLight, mat, reduced, scene, sign, tube, up, windowsTex } from "../world.js";
import { backOut, ease, lerp, sleep, tween } from "../motion.js";
import { makeAdviser, makeChair, makeDesk, makeEnvelope, makeFax, makeFaxPage, makePhone, poseAdviser, sitAndThink, walkTo, wood, woodDark } from "../props.js";
import { speak } from "../narration.js";
import { addHistory, choose, clearStage, endings, findEnding, goFloor, labels, revealButtons, setClock } from "../hud.js";

const config = {
  number: 2, name: "The Annex", minutes: 2, scale: 8, keys: ["Moving", "Thinking"],
  labels: ["aMain", "aAnnex", "aExpert", "aFax", "aVan"],
  map: [
    ["Main office", "the graphics card: fast, with room for only some of the experts"],
    ["Annex", "the CPU and the computer's main memory: slower, with room for all of them"],
    ["Adviser", "an expert: one of 512 in each layer. Each word uses 10"],
    ["The van", "copying an expert's weights onto the card: megabytes"],
    ["The fax", "sending the word's numbers to the CPU and back: about 10 KB"],
  ],
  lesson: "c0ven keeps the busiest experts on the graphics card. When a word needs one that isn't there, the CPU works it out where it already sits, and only the small answer comes back. Moving the question is far cheaper than moving the knowledge.",
  next: "Floor 3 · Musical Desks · open now.",
};

const RUNS = {
  drive: { label: "Drive them over", fetch: 6.8, think: 0.6 },
  fax: { label: "Send a fax", fetch: 0.7, think: 0.9 },
};

// Far enough from floor 1 that each floor's fog hides the other.
const X2 = 60;
const MAIN_X = X2 - 4.6, ANNEX_X = X2 + 6.0;   // where each office's adviser stands, at z = 0.45
const VAN_OFFICE = X2 - 3.6, VAN_ANNEX = X2 + 5.4, VAN_LIGHT = X2 + 2.7, VAN_Z = 3.85;
const colleagues = [];
const trafficLamps = {};
const wireMat = new THREE.LineBasicMaterial({ color: 0x14171c });
let wireLine = null;
(() => {
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(44, 34), mat(0x1f232b, { roughness: 1 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(X2, 0, 2); ground.receiveShadow = true; scene.add(ground);
  box(17.5, 0.06, 0.7, mat(0x3a3f48), X2 + 0.6, 0.03, 2.75);
  box(17.5, 0.02, 1.8, mat(0x191c22, { roughness: 1 }), X2 + 0.6, 0.01, 3.95);
  for (let x = -8; x <= 9; x += 1.2) box(0.6, 0.025, 0.07, mat(0xd8d2b8, { roughness: 0.8 }), X2 + x, 0.025, 3.95);
  box(17.5, 0.06, 0.5, mat(0x3a3f48), X2 + 0.6, 0.03, 5.1);

  // The main office: a cutaway room, busy
  const officeWall = mat(0x323b4b);
  box(5.0, 0.1, 5.2, mat(0x2f3540, { roughness: 1 }), X2 - 4.7, 0.05, 0);
  box(5.0, 3.6, 0.12, officeWall, X2 - 4.7, 1.8, -2.6);
  box(0.12, 3.6, 5.2, officeWall, X2 - 7.2, 1.8, 0);
  const win = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 1.1), new THREE.MeshBasicMaterial({ map: windowsTex }));
  win.position.set(X2 - 4.7, 2.7, -2.53); scene.add(win);
  sign("MAIN OFFICE", 2.4, 0.42, X2 - 4.7, 3.92, -2.52);
  tube(X2 - 4.7, 3.5, 0.2);
  makeDesk(X2 - 4.6, 1.35);
  makeChair(X2 - 4.6, 0.15);
  // Colleagues: the popular advisers, busy at their desks
  for (const [cx, suit, hair, ph] of [[X2 - 6.0, 0x6a5f78, 0x2a2420, 1.3], [X2 - 3.4, 0x5b6f63, 0x6b4a2e, 2.9]]) {
    makeDesk(cx, -0.95, false);
    makeChair(cx, -2.0);
    makePhone(cx + 0.45, -1.1, -0.4);
    const c = makeAdviser({ x: cx, z: -2.0, suit, hair, nodPhase: ph });
    c.sit = 1; c.busy = true; colleagues.push(c);
  }
  // A side table with the fax machine
  box(0.6, 0.06, 0.5, wood, X2 - 6.1, 0.75, 0.45);
  for (const [dx, dz] of [[-0.25, -0.2], [0.25, -0.2], [-0.25, 0.2], [0.25, 0.2]]) box(0.04, 0.72, 0.04, woodDark, X2 - 6.1 + dx, 0.36, 0.45 + dz);
  makeFax(X2 - 6.1, 0.79, 0.45, 0.6);

  // The annex: older, smaller, and lined with every manual nobody visits
  const brick = mat(0x5a3b32, { roughness: 0.95 });
  box(4.4, 0.1, 5.2, mat(0x3a332e, { roughness: 1 }), X2 + 6.1, 0.05, 0);
  box(4.4, 3.6, 0.12, brick, X2 + 6.1, 1.8, -2.6);
  box(0.12, 3.6, 5.2, brick, X2 + 8.3, 1.8, 0);
  const library = canvasTex(1024, 768, (g, w, h) => {
    g.fillStyle = "#2c2520"; g.fillRect(0, 0, w, h);
    const cols = ["#5d6b7f", "#7a6a55", "#4f6a63", "#6d7486", "#8a5a48", "#3f4e66", "#6b5a3a"];
    let n = 0;
    for (let row = 0; row < 6; row++) {
      const y0 = 14 + row * 126, y1 = y0 + 112;
      let x = 10;
      while (x < w - 16) {
        const bw = 9 + ((n * 7) % 6) * 3, bh = 82 + ((n * 5) % 7) * 4;
        g.fillStyle = cols[n % cols.length]; g.fillRect(x, y1 - bh, bw, bh);
        x += bw + 2; n++;
      }
      g.fillStyle = "#4a3d31"; g.fillRect(0, y1, w, 12);
    }
  });
  const lib = new THREE.Mesh(new THREE.PlaneGeometry(4.1, 3.1), new THREE.MeshStandardMaterial({ map: library, roughness: 0.9 }));
  lib.position.set(X2 + 6.1, 1.65, -2.53); scene.add(lib);
  sign("ANNEX", 1.4, 0.42, X2 + 6.1, 3.92, -2.52, "#d9c9a8", "#3a2a22");
  const warm = new THREE.PointLight(0xffc27a, 0.7, 7, 1.6); warm.position.set(X2 + 6.1, 3.2, 0.4); scene.add(warm);
  makeDesk(X2 + 6.0, 1.35);
  makeChair(X2 + 6.0, 0.15);
  makeFax(X2 + 6.75, 1.07, 1.1, 0.75);

  // The street: a lamp and a traffic light
  const steel = mat(0x5d636d, { metalness: 0.5, roughness: 0.5 });
  const lampPole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 3.4, 10), steel);
  lampPole.position.set(X2 + 0.6, 1.7, 2.6); lampPole.castShadow = true; scene.add(lampPole);
  const lampArm = box(0.5, 0.05, 0.07, steel, X2 + 0.82, 3.4, 2.6); lampArm.rotation.z = 0.12;
  box(0.3, 0.08, 0.22, steel, X2 + 1.04, 3.36, 2.6);
  box(0.26, 0.04, 0.18, mat(0xffd59a, { emissive: 0xffb45c, emissiveIntensity: 1.2 }), X2 + 1.04, 3.31, 2.6);
  halo(0xffb45c, 1.3, X2 + 1.04, 3.28, 2.66, scene, 0.45);
  const streetLight = new THREE.PointLight(0xffb45c, 1.1, 9, 1.6); streetLight.position.set(X2 + 1.0, 3.1, 2.8); scene.add(streetLight);
  const tlPole = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 2.4, 10), steel);
  tlPole.position.set(X2 + 1.9, 1.2, 2.85); scene.add(tlPole);
  box(0.14, 0.03, 0.14, steel, X2 + 1.9, 0.015, 2.85);
  const housing = mat(0x1c2027, { roughness: 0.6 });
  box(0.26, 0.74, 0.22, housing, X2 + 1.9, 2.6, 2.85);
  box(0.3, 0.03, 0.26, housing, X2 + 1.9, 2.98, 2.87);
  [["red", 0xd2412f, 2.8], ["amber", 0xe8a33d, 2.6], ["green", 0x4fb36b, 2.4]].forEach(([name, c, y]) => {
    const m = new THREE.MeshStandardMaterial({ color: 0x222222, emissive: c, emissiveIntensity: 0.05 });
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.065, 12, 10), m); s.position.set(X2 + 1.9, y, 2.96); scene.add(s);
    box(0.2, 0.02, 0.1, housing, X2 + 1.9, y + 0.08, 3.0);
    trafficLamps[name] = { m, halo: halo(c, 0.55, X2 + 1.9, y, 3.03, scene, 0) };
  });

  // Telephone poles and the line the fax travels on
  for (const px of [X2 - 2.0, X2 + 3.9]) {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 4.2, 10), mat(0x3b3029));
    p.position.set(px, 2.1, -0.3); p.castShadow = true; scene.add(p);
    box(0.7, 0.06, 0.06, mat(0x3b3029), px, 4.0, -0.3);
  }
  const sag = new THREE.QuadraticBezierCurve3(new THREE.Vector3(X2 - 2.0, 4.05, -0.3), new THREE.Vector3(X2 + 0.95, 3.2, -0.3), new THREE.Vector3(X2 + 3.9, 4.05, -0.3));
  wireLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(sag.getPoints(40)), wireMat);
  scene.add(wireLine);

  // Background city
  const cityMat = new THREE.MeshBasicMaterial({ map: windowsTex, color: 0x9aa3b0 });
  [[-14, 7, 3], [-9, 10, 3.5], [-3, 6, 4], [2.5, 12, 3], [8, 8, 4], [13, 11, 3.2], [18, 7, 3]].forEach(([dx, h, w]) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, 2), [mat(0x161a21), mat(0x161a21), mat(0x161a21), mat(0x161a21), cityMat, mat(0x161a21)]);
    b.position.set(X2 + dx, h / 2, -11); scene.add(b);
  });
  keyLight(X2 + 0.5, 14, 9, X2 + 0.5, Math.PI / 4, 40);
})();

function setTraffic(on) { for (const [n, l] of Object.entries(trafficLamps)) { l.m.emissiveIntensity = n === on ? 1.4 : 0.05; l.halo.material.opacity = n === on ? 0.8 : 0; } }

const M = makeAdviser({ x: MAIN_X, z: 0.45 });
const N = makeAdviser({ x: ANNEX_X, z: 0.45, suit: 0x6b5a48, hair: 0xc4c4bf, tie: null, nodPhase: 0.7, old: true });

// The van: cab at +x, cargo box behind, a roof rack for the four crates of manuals. vanBody carries the
// suspension's squat and dip; the wheels sit on `van` and turn with its travel.
const van = new THREE.Group(); scene.add(van);
const vanBody = new THREE.Group(); van.add(vanBody);
const paint = mat(0xc9ccc4, { roughness: 0.55 }), glass = mat(0x1b2230, { roughness: 0.2, metalness: 0.4 }), trim = mat(0x2a2e36, { roughness: 0.6 });
box(1.9, 1.0, 0.95, paint, -0.25, 0.78, 0, vanBody);
box(0.78, 0.42, 0.95, paint, 1.06, 0.49, 0, vanBody);
box(0.55, 0.5, 0.95, paint, 0.95, 0.95, 0, vanBody);
const screen = box(0.03, 0.42, 0.85, glass, 1.21, 0.93, 0, vanBody); screen.rotation.z = 0.28;
for (const sz of [-1, 1]) {
  box(0.42, 0.3, 0.02, glass, 0.98, 0.98, sz * 0.475, vanBody);
  box(0.05, 0.1, 0.06, trim, 1.2, 0.98, sz * 0.52, vanBody);
  box(0.02, 0.5, 0.02, trim, 0.4, 0.78, sz * 0.475, vanBody);
  box(0.02, 0.5, 0.02, trim, -0.5, 0.78, sz * 0.475, vanBody);
  box(0.9, 0.02, 0.02, trim, -0.05, 1.03, sz * 0.475, vanBody);
  const label = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.26), new THREE.MeshStandardMaterial({ transparent: true, roughness: 0.6, map: canvasTex(512, 112, (g, w, h) => {
    g.fillStyle = "#56657d"; g.font = "700 72px 'Barlow Condensed', Arial Narrow, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText("MERIDIAN WORDS", w / 2, h / 2);
  }) }));
  label.position.set(-0.25, 0.88, sz * 0.478); label.rotation.y = sz > 0 ? 0 : Math.PI; vanBody.add(label);
}
box(1.9, 0.12, 0.96, mat(0xe8a33d), -0.25, 0.62, 0, vanBody);
box(0.08, 0.14, 0.98, trim, 1.44, 0.3, 0, vanBody);
box(0.08, 0.14, 0.98, trim, -1.22, 0.3, 0, vanBody);
box(0.02, 0.18, 0.5, trim, 1.46, 0.52, 0, vanBody);
for (const rz of [-0.4, 0.4]) box(1.7, 0.04, 0.04, trim, -0.3, 1.32, rz, vanBody);
for (const rx of [-1.0, -0.3, 0.4]) box(0.04, 0.04, 0.86, trim, rx, 1.32, 0, vanBody);
const tailM = mat(0xd2412f, { emissive: 0xd2412f, emissiveIntensity: 0.4 });
const beams = [];
for (const hz of [-0.32, 0.32]) {
  box(0.03, 0.1, 0.16, mat(0xfff2c9, { emissive: 0xfff2c9, emissiveIntensity: 1 }), 1.46, 0.52, hz, vanBody);
  box(0.03, 0.1, 0.1, tailM, -1.21, 0.62, hz, vanBody);
  const beam = new THREE.Mesh(new THREE.ConeGeometry(0.5, 2.4, 14, 1, true), new THREE.MeshBasicMaterial({ color: 0xfff2c9, transparent: true, opacity: 0.1, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
  beam.position.set(2.66, 0.45, hz); beam.rotation.z = Math.PI / 2; beam.visible = false; vanBody.add(beam); beams.push(beam);
}
const headLight = new THREE.SpotLight(0xfff2c9, 1.2, 7, 0.55, 0.6, 1.2);
headLight.position.set(1.4, 0.55, 0); headLight.target.position.set(6, -0.5, 0); vanBody.add(headLight, headLight.target);
headLight.visible = false;
function vanLights(on) { headLight.visible = on; for (const b of beams) b.visible = on; }
function brake(on) { tailM.emissiveIntensity = on ? 1.6 : 0.4; }
const wheels = [], hub = mat(0x8a9099, { metalness: 0.6, roughness: 0.4 });
for (const [wx, wz] of [[-0.75, -0.48], [-0.75, 0.48], [0.95, -0.48], [0.95, 0.48]]) {
  const p = new THREE.Group(); p.position.set(wx, 0.22, wz); van.add(p); wheels.push(p);
  const w = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.14, 16), mat(0x1c2027)); w.rotation.x = Math.PI / 2; w.castShadow = true; p.add(w);
  const h = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.15, 12), hub); h.rotation.x = Math.PI / 2; p.add(h);
  box(0.17, 0.03, 0.16, trim, 0, 0, 0, p);
}
const vanLast = new THREE.Vector3();
const crates = [];
for (const [cx, cz, cy] of [[-0.75, -0.22, 1.54], [-0.75, 0.22, 1.54], [0.1, -0.22, 1.54], [-0.32, 0, 1.94]]) {
  const c = new THREE.Group(); c.position.set(cx, cy, cz); vanBody.add(c);
  box(0.42, 0.4, 0.4, mat(0x8a6a44, { roughness: 0.9 }), 0, 0, 0, c);
  for (const dy of [-0.12, 0.12]) box(0.44, 0.04, 0.42, mat(0x5e4630, { roughness: 0.9 }), 0, dy, 0, c);
  c.visible = false; crates.push(c);
}

// The fax page's route runs up from the main office's fax, along the line, and down to the annex's fax.
const { group: pageG, sheet: page, geo: pageGeo } = makeFaxPage();

const faxRoute = new THREE.CatmullRomCurve3([
  new THREE.Vector3(X2 - 6.1, 1.05, 0.45), new THREE.Vector3(X2 - 6.0, 3.5, 0.3), new THREE.Vector3(X2 - 2.0, 4.1, -0.3),
  new THREE.Vector3(X2 + 0.95, 3.3, -0.3), new THREE.Vector3(X2 + 3.9, 4.1, -0.3), new THREE.Vector3(X2 + 6.7, 3.5, 1.0),
  new THREE.Vector3(X2 + 6.75, 1.3, 1.1),
]);
const E2 = makeEnvelope();
const ENV2_IN = new THREE.Vector3(MAIN_X - 0.2, 1.03, 1.1), ENV2_OUT = new THREE.Vector3(X2 - 10, 1.6, 0.8);
E2.group.position.copy(ENV2_OUT);

function resetFloor2() {
  van.position.set(VAN_OFFICE, 0, VAN_Z); van.rotation.y = 0; vanBody.scale.set(1, 1, 1); vanBody.position.y = 0; vanBody.rotation.set(0, 0, 0);
  vanLast.copy(van.position); vanLights(false); brake(false);
  for (const c of crates) { c.visible = false; c.scale.setScalar(1); }
  for (const [A, x] of [[M, MAIN_X], [N, ANNEX_X]]) {
    A.group.visible = true; A.group.position.set(x, 0, 0.45); A.group.rotation.y = A.heading = 0;
    A.sit = 0; A.walking = 0; A.reach = false; A.carry = false; A.lift = 0;
  }
  setTraffic("green");
  pageG.visible = false;
  E2.group.position.copy(ENV2_OUT); E2.stamp.visible = false;
}

const camera2 = { look: new THREE.Vector3(X2 + 0.6, 0.8, 1.6), offset: true, fit: 8.8, fog: [20, 46], par: 1.6 };

// Backs the camera off until `fit` units either side of the look point are in view, at any aspect.
const fitPos = new THREE.Vector3();
// On wide screens the view centres on the part the side panel leaves clear.
function fitCam(C) {
  const side = innerWidth > 720 ? 350 : 0;
  camera.setViewOffset(innerWidth, innerHeight, side / 2, -innerHeight * 0.1, innerWidth, innerHeight);
  const half = Math.tan(camera.fov * Math.PI / 360), aspect = (innerWidth - side) / innerHeight;
  const d = Math.max(C.fit / (half * aspect), 6 / half) * 1.05;
  scene.fog.near = d * 0.8; scene.fog.far = d * 2.4;
  return fitPos.set(C.look.x, C.look.y + d * 0.36, C.look.z + d);
}

camera2.place = () => fitCam(camera2);
const anchors = [
  [$("aMain"), () => new THREE.Vector3(X2 - 4.7, 4.35, -2.5)],
  [$("aAnnex"), () => new THREE.Vector3(X2 + 6.1, 4.35, -2.5)],
  [$("aExpert"), () => up(N.group, 2.2)],
  [$("aFax"), () => new THREE.Vector3(X2 + 0.95, 3.75, -0.3)],
  [$("aVan"), () => up(van, 2.0)],
];

// Floor 2: a counted stretch of moving; the clock shows `moved` so far plus this stretch, and `thought` so far.
function counter() {
  const c = { moved: 0, thought: 0 };
  c.seg = (s, fn) => tween(s, (u) => { if (fn) fn(u); setClock(c.moved + s * u, c.thought); }).then(() => { c.moved += s; });
  return c;
}
async function envelopeIn() {
  setClock(0, 0);
  await tween(0.7, (u) => E2.group.position.lerpVectors(ENV2_OUT, ENV2_IN, ease(u)));
  await sleep(0.2);
}
// The van's suspension: a squat as it pulls away, a dip as it brakes, a shiver while it idles.
const TURN_R = 0.28;
function vanRide(u, dip) { vanBody.rotation.z = dip * Math.sin(u * Math.PI); vanBody.position.y = -0.04 + Math.abs(Math.sin(u * 14)) * 0.015; }
// The van fetches the annex adviser and four crates of manuals, stopping at the light on the way back.
async function driveRun() {
  resetFloor2();
  await envelopeIn();
  const c = counter();
  vanLights(true);
  await c.seg(1.4, (u) => {
    van.position.x = lerp(VAN_OFFICE, VAN_ANNEX, ease(u));
    vanRide(u, u < 0.5 ? 0.035 : -0.045); brake(u > 0.6);
  });
  brake(false);
  await Promise.all([c.seg(0.6), walkTo(N, VAN_ANNEX, 2.95, 0.6)]);
  // N climbs in at the kerb side and the van rocks; the crates drop onto the rack as it U-turns into the near lane.
  N.heading = 0; N.walking = 1;
  await c.seg(0.8, (u) => {
    const inU = ease(Math.min(1, u / 0.3));
    N.group.position.set(VAN_ANNEX, 0, 2.95 + 0.45 * inU); N.lift = 0.35 * inU; N.walkPhase += 0.2;
    if (u >= 0.3) N.group.visible = false;
    vanBody.rotation.x = -0.05 * Math.sin(inU * Math.PI);
    crates.forEach((cr, i) => {
      const s = Math.min(1, Math.max(0, (u - 0.1) * 5 - i));
      cr.visible = s > 0; cr.scale.setScalar(s > 0 ? backOut(s) : 1); cr.position.y = (i === 3 ? 1.94 : 1.54) + (1 - s) * 0.5;
    });
    const a = Math.PI * ease(Math.max(0, (u - 0.3) / 0.7));
    van.rotation.y = -a;
    van.position.set(VAN_ANNEX + TURN_R * Math.sin(a), 0, VAN_Z + TURN_R * (1 - Math.cos(a)));
    vanBody.position.y = -0.04 * Math.min(1, u * 4);
  });
  N.walking = 0; N.lift = 0;
  vanBody.scale.y = 0.94; vanBody.position.y = -0.04;
  setTraffic("amber");
  walkTo(M, MAIN_X - 0.95, 0.55, 0.8);
  await c.seg(1.0, (u) => {
    van.position.x = lerp(VAN_ANNEX, VAN_LIGHT, Math.sin(u * Math.PI / 2));
    if (u > 0.5) setTraffic("red");
    vanRide(u, u > 0.5 ? -0.05 : 0); brake(u > 0.45);
  });
  await c.seg(1.0, (u) => { vanBody.position.y = -0.04 + Math.sin(u * 60) * 0.006; vanBody.rotation.z = 0; if (u > 0.85) setTraffic("green"); });
  setTraffic("green");
  await c.seg(1.0, (u) => {
    van.position.x = lerp(VAN_LIGHT, VAN_OFFICE, ease(u));
    vanRide(u, u < 0.5 ? 0.035 : -0.045); brake(u > 0.65);
  });
  brake(false); vanLights(false); vanBody.rotation.z = 0;
  M.heading = 0;
  // N hops out on the road side and crosses to the main office's desk.
  N.group.position.set(VAN_OFFICE, 0, van.position.z - 0.5); N.group.rotation.y = N.heading = Math.PI; N.lift = 0.35; N.group.visible = true;
  tween(0.2, (u) => { N.lift = 0.35 * (1 - ease(u)); });
  await Promise.all([c.seg(1.0), (async () => {
    await walkTo(N, MAIN_X + 1.4, 0.5, 0.45);
    await walkTo(N, MAIN_X, 0.45, 0.55);
  })()]);
  await sitAndThink(N, c.moved, RUNS.drive.think);
  E2.stamp.visible = true;
  addHistory(RUNS.drive);
  await sleep(0.4);
}
// The main office faxes the question; the annex adviser works it out there and faxes the answer back.
async function faxRun() {
  resetFloor2();
  await envelopeIn();
  await walkTo(M, MAIN_X - 0.95, 0.45, 0.45);
  M.heading = -Math.PI / 2; M.reach = true;
  const c = counter();
  pageG.visible = true;
  // The page leaves the slot flat and small, hangs below the clip over the line, and shrinks into the far slot.
  const place = (u) => {
    const end = Math.min(1, u * 8, (1 - u) * 8);
    pageG.position.copy(faxRoute.getPointAt(u));
    page.position.y = -0.3 * end; page.scale.setScalar(0.3 + 0.7 * end);
    page.rotation.set(-0.2 + 0.15 * Math.sin(u * 9), 0.35 + Math.sin(u * 7) * 0.3, Math.sin(u * 13) * 0.12);
  };
  // The line glows while a page is on it.
  wireMat.color.setHex(0xe8a33d);
  await c.seg(0.35, (u) => { place(u); if (u > 0.6) { N.heading = 0.9; N.reach = true; } });
  pageG.visible = false; M.reach = false; N.reach = false;
  wireMat.color.setHex(0x14171c);
  await sitAndThink(N, c.moved, RUNS.fax.think);
  c.thought = RUNS.fax.think;
  N.heading = 0.9; N.reach = true;
  pageG.visible = true;
  wireMat.color.setHex(0xe8a33d);
  await c.seg(0.35, (u) => { place(1 - u); if (u > 0.3) N.reach = false; if (u > 0.6) M.reach = true; });
  pageG.visible = false; M.reach = false;
  wireMat.color.setHex(0x14171c);
  await walkTo(M, MAIN_X, 0.45, 0.45);
  M.heading = N.heading = 0;
  E2.stamp.visible = true;
  addHistory(RUNS.fax);
  await sleep(0.4);
}

async function f2Intro() {
  await speak("f2intro");
  choose([["Drive them over", drive], ["Send a fax", fax]]);
}
async function drive() {
  clearStage();
  await speak("f2drive1");
  await driveRun();
  findEnding("drive", "The Traffic Ending");
  await speak("f2drive2");
  if (endings[2].has("fax")) choose([["See what it means", reveal2]]);
  else choose([["Try the other door", fax]]);
}
async function fax() {
  clearStage();
  await speak("f2fax1");
  await faxRun();
  findEnding("fax", "One Page Each Way");
  await speak("f2fax2");
  choose([["What just happened?", reveal2]]);
}
async function reveal2() {
  labels(true);
  const more = !endings[2].has("drive");
  $("nextNote").textContent = config.next + (more ? " One ending on this floor is still hidden." : " Both endings found.");
  revealButtons([]);
  await speak("f2reveal", (line) => { if (line === "f2r2") $("reveal").hidden = false; });
  $("reveal").hidden = false;
  $("caption").hidden = true;
  revealButtons([
    ["Take the lift to floor 3", () => goFloor(3), true],
    ...(more ? [["Try the other door", drive]] : []),
    ["Replay floor 2", () => { endings[2].clear(); goFloor(2); }],
    ["Back to floor 1", () => goFloor(1)],
  ]);
}

resetFloor2();

export default {
  ...config, anchors, camera: camera2, enter: resetFloor2, start: f2Intro,
  frame(t, dt) {
    for (const A of [M, N, ...colleagues]) poseAdviser(A, t, dt);
    // Wheels turn with the van's travel along its own length.
    const fwd = (van.position.x - vanLast.x) * Math.cos(van.rotation.y) - (van.position.z - vanLast.z) * Math.sin(van.rotation.y);
    vanLast.copy(van.position);
    for (const w of wheels) w.rotation.z -= fwd / 0.22;
    if (pageG.visible && !reduced) {
      const p = pageGeo.attributes.position;
      for (let i = 0; i < p.count; i++) p.setZ(i, Math.sin(p.getX(i) * 14 + t * 22) * 0.015 + Math.sin(p.getY(i) * 9 + t * 17) * 0.01);
      p.needsUpdate = true;
    }
  },
  onLabels(on) { wireMat.color.setHex(on ? 0xe8a33d : 0x14171c); },
};
