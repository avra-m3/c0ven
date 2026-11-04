// Floor 2, the annex, around x = X2. An expert the card lacks is worked out where it sits: move the question, not the knowledge.
// A cross-section of the street: the main office two storeys up the tower, the annex level with it in the old building,
// the road between them far below.
import { $, box, camera, canvasTex, closeFloor, halo, keyLight, mat, openFloor, pool, reduced, rep, scene, sign, tubes, up } from "../world.js";
import { backOut, ease, lerp, sleep, tween } from "../motion.js";
import { makeAbacus, makeAdviser, makeCabinet, makeCat, makeChair, makeClock, makeCoatStand, makeCooler, makeEnvelope, makeFax, makeFaxPage, makeLadder, makeLibrary, makePhone, makePlant, makeRug, makeVan, paperM, poseAdviser, roomDesk, sitAndThink, steel3, walkTo, wood, woodDark } from "../props.js";
import { asphaltTex, block, boardsTex, brickTex, carpetTex, city, curtainWall, floorDrop, makeHoist, makeLift, plaster, skirtM, slabTex, trimM, wallWithDoor, windowView } from "../tower.js";
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
const X2 = 60, opened = openFloor();
const MAIN_X = X2 - 4.6, ANNEX_X = X2 + 6.0;   // where each office's adviser stands, at z = 0.45
const VAN_OFFICE = X2 - 3.6, VAN_ANNEX = X2 + 5.4, VAN_LIGHT = X2 + 4.4, VAN_Z = 3.85;
const S = -floorDrop(2);   // the street's height; the rooms' floors are at 0
const LIFT_X = X2 - 1.2, LIFT_Z = -0.3, TOWER_X0 = X2 - 7.2, TOWER_X1 = X2 - 0.08, ANNEX_X0 = X2 + 3.9, ANNEX_X1 = X2 + 8.3, H2 = 4.4;
const WIRE_Z = -2.45;   // the telephone line runs behind both rooms' open sides
const colleagues = [], phones = [];
const trafficLamps = {};
const wireMat = new THREE.LineBasicMaterial({ color: 0x14171c });
let wireLine = null, cityUpdate = null, cat = null, beads = [], clockHands = null;
const street = new THREE.Group(); street.position.set(0, S, 0); scene.add(street);
// A side wall, full height at the back and low at the front, so the room stays open to the camera.
function sideWall(x, m, h, front = 1.3) {
  const sh = new THREE.Shape();
  sh.moveTo(-2.6, 0); sh.lineTo(2.6, 0); sh.lineTo(2.6, front); sh.lineTo(-2.6, h); sh.closePath();
  const w = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, { depth: 0.12, bevelEnabled: false }), m);
  w.rotation.y = -Math.PI / 2; w.position.set(x + 0.06, 0, 0); w.castShadow = w.receiveShadow = true; scene.add(w);
}
// The room's hanging tube: emissive with a glow; one real light under it.
function hangingTube(x, z, y = 3.5) {
  const t = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.05, 0.16), new THREE.MeshBasicMaterial({ color: 0xe6efe9 }));
  t.position.set(x, y, z); scene.add(t); tubes.push(t);
  halo(0xdfe9e4, 1.6, x, y - 0.05, z + 0.05, scene, 0.16);
  for (const wx of [-1.0, 1.0]) box(0.01, 0.9, 0.01, steel3, x + wx, y + 0.45, z);
  const l = new THREE.PointLight(0xdfe9e4, 0.55, 10, 1.6); l.position.set(x, y - 0.25, z + 0.9); scene.add(l);
}
const lift = makeLift(scene, { x: LIFT_X, y: 0.1, z: LIFT_Z, number: 2, name: "The Annex", signSide: 0, signAt: [0, 3.15, 1.2] });
(() => {
  // ---------- The street, below ----------
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 12), mat(0x161a20, { roughness: 1 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(X2, 0, -3); ground.receiveShadow = true; street.add(ground);
  // The buildings either side, so the two offices stand in a street.
  block(street, X2 + 13.5, -1.4, 10.3, 8, 0, 10.5, 2, 0x5c5450);
  block(street, X2 + 23.5, -1.4, 9.2, 8, 0, 15, 0, 0x565b64);
  block(street, X2 - 12.2, -1.4, 10, 8, 0, 8.5, 1, 0x565b64);
  block(street, X2 - 22, -1.4, 9.2, 8, 0, 13, 0, 0x5c5450);
  box(60, 0.12, 0.75, mat(0xffffff, { map: rep(slabTex, 75, 1), roughness: 0.9 }), X2, 0.06, 2.72, street);
  box(60, 0.14, 0.1, mat(0x4a4f58), X2, 0.07, 3.1, street);
  const road = new THREE.Mesh(new THREE.PlaneGeometry(60, 1.95), mat(0xffffff, { map: rep(asphaltTex, 30, 1), roughness: 0.5, metalness: 0.18 }));
  road.rotation.x = -Math.PI / 2; road.position.set(X2, 0.01, 4.1); road.receiveShadow = true; street.add(road);
  for (let x = -30; x <= 30; x += 1.2) box(0.6, 0.012, 0.06, mat(0xd8d2b8, { roughness: 0.8 }), X2 + x, 0.02, 4.1, street);
  box(60, 0.14, 0.1, mat(0x4a4f58), X2, 0.07, 5.08, street);
  box(60, 0.12, 2.4, mat(0xffffff, { map: rep(slabTex, 75, 3), roughness: 0.9 }), X2, 0.06, 6.3, street);
  // A wet shine under the lamp, and the light of the tower's doors on the pavement.
  pool(street, 0xffa850, 2.5, X2 + 2.14, 0.13, 2.9, 0.32);
  pool(street, 0xffa850, 1.8, X2 + 2.14, 0.025, 4.2, 0.2);
  pool(street, 0xffdcae, 2.1, VAN_OFFICE, 0.13, 2.9, 0.26);
  pool(street, 0xffdcae, 1.6, VAN_OFFICE, 0.025, 3.7, 0.1);

  // The street lamp and the traffic light.
  const steel = mat(0x5d636d, { metalness: 0.5, roughness: 0.5 });
  const lampPole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 3.0, 10), steel);
  lampPole.position.set(X2 + 1.7, 1.5, 2.6); lampPole.castShadow = true; street.add(lampPole);
  const lampArm = box(0.5, 0.05, 0.07, steel, X2 + 1.92, 3.0, 2.6, street); lampArm.rotation.z = 0.12;
  box(0.3, 0.08, 0.22, steel, X2 + 2.14, 2.96, 2.6, street);
  box(0.26, 0.04, 0.18, mat(0xffd59a, { emissive: 0xffb45c, emissiveIntensity: 1.2 }), X2 + 2.14, 2.91, 2.6, street);
  halo(0xffb45c, 1.3, X2 + 2.14, 2.88, 2.66, street, 0.45);
  const streetLight = new THREE.PointLight(0xffb45c, 1.1, 9, 1.6); streetLight.position.set(X2 + 2.1, 2.8, 2.9); street.add(streetLight);
  const TL = X2 + 0.2;
  const tlPole = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 2.4, 10), steel);
  tlPole.position.set(TL, 1.2, 2.85); street.add(tlPole);
  box(0.14, 0.03, 0.14, steel, TL, 0.13, 2.85, street);
  const housing = mat(0x1c2027, { roughness: 0.6 });
  box(0.3, 0.8, 0.22, housing, TL, 2.6, 2.85, street);
  box(0.34, 0.03, 0.26, housing, TL, 2.98, 2.87, street);
  [["red", 0xd2412f, 2.84], ["amber", 0xe8a33d, 2.6], ["green", 0x4fb36b, 2.36]].forEach(([name, c, y]) => {
    const m = new THREE.MeshStandardMaterial({ color: 0x222222, emissive: c, emissiveIntensity: 0.05 });
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 10), m); s.position.set(TL, y, 2.96); street.add(s);
    box(0.2, 0.02, 0.1, housing, TL, y + 0.08, 3.0, street);
    trafficLamps[name] = { m, halo: halo(c, 0.95, TL, y, 3.03, street, 0) };
  });
  // The narrator says the annex is across town. So does the sign.
  const signPole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 2.1, 8), steel); signPole.position.set(X2 + 3.1, 1.05, 2.72); street.add(signPole);
  const roadSign = canvasTex(320, 128, (g, w, h) => {
    g.fillStyle = "#1f4f8a"; g.fillRect(0, 0, w, h); g.strokeStyle = "#f2f2ec"; g.lineWidth = 6; g.strokeRect(6, 6, w - 12, h - 12);
    g.fillStyle = "#f2f2ec"; g.textAlign = "center"; g.textBaseline = "middle"; g.font = "700 46px 'Barlow Condensed', Arial Narrow, sans-serif";
    g.fillText("ANNEX  →", w / 2, 44); g.font = "600 34px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("11 MILES", w / 2, 92);
  });
  const roadM = new THREE.Mesh(new THREE.PlaneGeometry(0.95, 0.38), new THREE.MeshStandardMaterial({ map: roadSign, roughness: 0.5, emissive: 0x1f2a3a, emissiveMap: roadSign, emissiveIntensity: 0.5 }));
  roadM.position.set(X2 + 3.1, 1.95, 2.76); street.add(roadM);
  // A pillar box, and bins down the alley.
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.9, 14), mat(0xa32a22, { roughness: 0.5 })); post.position.set(X2 - 6.2, 0.55, 2.82); post.castShadow = true; street.add(post);
  for (const [x, z] of [[X2 + 0.5, -1], [X2 + 1.1, -0.8]]) { const bin = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.17, 0.6, 12), mat(0x3a3f48)); bin.position.set(x, 0.3, z); street.add(bin); }

  // ---------- The tower ----------
  // Below the office: the ground floor with its doors, then floor 1, lit.
  box(TOWER_X1 - TOWER_X0, -S - 0.02, 12, mat(0x12151b), (TOWER_X0 + TOWER_X1) / 2, S / 2 - 0.01, -3.4);
  curtainWall(scene, (TOWER_X0 + TOWER_X1) / 2, 2.61, TOWER_X1 - TOWER_X0, S + 2.6, -0.14, 2.3, 0x83878e);
  const shop = new THREE.Mesh(new THREE.PlaneGeometry(TOWER_X1 - TOWER_X0, 2.1), new THREE.MeshBasicMaterial({ color: 0x8c8c8c, map: canvasTex(512, 128, (g, w, h) => {
    const l = g.createLinearGradient(0, 0, 0, h); l.addColorStop(0, "#e9cf98"); l.addColorStop(1, "#8d7650"); g.fillStyle = l; g.fillRect(0, 0, w, h);
    g.fillStyle = "#14181f"; for (let x = 0; x <= w; x += 56) g.fillRect(x, 0, 5, h); g.fillRect(0, 0, w, 7);
    g.fillStyle = "#40382a"; g.fillRect(186, 20, 70, 108);
  }) }));
  shop.position.set((TOWER_X0 + TOWER_X1) / 2, S + 1.17, 2.62); scene.add(shop);
  // The doors N walks in by, under a canopy.
  box(1.0, 1.9, 0.04, mat(0x3a3020), VAN_OFFICE, S + 1.07, 2.65);
  box(2.2, 0.1, 0.7, mat(0x20252e), VAN_OFFICE, S + 2.3, 2.92);
  sign("MERIDIAN WORDS", 2.0, 0.24, VAN_OFFICE, S + 2.48, 3.1, "#14171c", "#f0ad45").material.emissive = new THREE.Color(0x4a3210);
  // The floor's edge, marked with its number.
  box(TOWER_X1 - TOWER_X0 + 0.1, 0.22, 0.1, mat(0x4a4f58), (TOWER_X0 + TOWER_X1) / 2, -0.06, 2.62);
  // Above the office the tower runs on up, out of the frame.
  box(TOWER_X1 - TOWER_X0, 46, 10, mat(0x12151b), (TOWER_X0 + TOWER_X1) / 2, H2 + 23, -7.7);
  curtainWall(scene, (TOWER_X0 + TOWER_X1) / 2, -2.68, TOWER_X1 - TOWER_X0, H2, H2 + 46, 2.3, 0x83878e);

  // The main office: blue-grey carpet, plaster walls, a tube overhead, windows onto the city.
  const OW = 7.12, OX = (TOWER_X0 + TOWER_X1) / 2;
  box(OW, 0.1, 5.2, mat(0xffffff, { map: rep(carpetTex, OW, 5.2), roughness: 1 }), OX, 0.05, 0);
  box(5.0, H2, 0.12, plaster, X2 - 4.7, H2 / 2, -2.6);
  box(5.0, 0.07, 0.05, trimM, X2 - 4.7, 1.05, -2.52);
  box(5.0, 0.14, 0.03, skirtM, X2 - 4.7, 0.17, -2.525);
  sideWall(TOWER_X0, plaster, H2);
  box(5, 0.04, 0.01, trimM, X2 - 4.7, H2, -2.53);
  for (const sx of [-1, 1]) {
    const cx = X2 - 4.7 + sx * 1.25;
    const w = new THREE.Mesh(new THREE.PlaneGeometry(1.85, 1.15), new THREE.MeshBasicMaterial({ map: windowView(11 + sx * 3, 2) }));
    w.position.set(cx, 2.75, -2.53); scene.add(w);
    for (const [fw, fh, fx, fy] of [[1.95, 0.07, 0, 0.6], [1.95, 0.1, 0, -0.6], [0.07, 1.2, -0.95, 0], [0.07, 1.2, 0.95, 0], [0.04, 1.15, 0, 0]]) box(fw, fh, 0.05, trimM, cx + fx, 2.75 + fy, -2.51);
  }
  sign("MAIN OFFICE", 2.4, 0.42, X2 - 4.7, 3.95, -2.52);
  clockHands = makeClock(scene, X2 - 4.7, 2.8, -2.52, 0.2);
  // The lift core, on the office's right, hollow so the car fits inside it.
  wallWithDoor(scene, plaster, LIFT_X - 1.14, LIFT_X + 1.14, H2, LIFT_Z, LIFT_X, 0.12, 0.1);
  for (const sx of [-1, 1]) box(0.12, H2, 2.3, plaster, LIFT_X + sx * 1.08, H2 / 2, -1.45);
  for (const [x0, x1] of [[LIFT_X - 1.14, LIFT_X - 0.77], [LIFT_X + 0.77, LIFT_X + 1.14]]) box(x1 - x0, 0.14, 0.03, skirtM, (x0 + x1) / 2, 0.17, LIFT_Z + 0.015);
  box(2.28, 0.1, 2.5, mat(0x2a2e36), LIFT_X, H2 + 0.05, -1.45);
  hangingTube(X2 - 4.7, -0.7);
  pool(scene, 0xc8daf0, 2.6, X2 - 4.7, 0.104, -0.4, 0.1);
  keyLight(X2 + 0.5, 13, 9, X2 + 0.5, Math.PI / 4, 40);
  roomDesk(scene, MAIN_X, 1.35, false);
  makeChair(MAIN_X, 0.15);
  // Colleagues: the popular advisers, busy at their desks.
  for (const [cx, suit, hair, ph] of [[X2 - 6.0, 0x6a5f78, 0x2a2420, 1.3], [X2 - 3.4, 0x5b6f63, 0x6b4a2e, 2.9]]) {
    roomDesk(scene, cx, -0.95, false);
    makeChair(cx, -2.0);
    phones.push(makePhone(cx + 0.48, -1.22, -0.4));
    const c = makeAdviser({ x: cx, z: -2.0, suit, hair, nodPhase: ph });
    c.sit = 1; c.busy = true; colleagues.push(c);
  }
  // A side table with the fax machine.
  box(0.6, 0.06, 0.5, wood, X2 - 6.1, 0.75, 0.45);
  for (const [dx, dz] of [[-0.25, -0.2], [0.25, -0.2], [-0.25, 0.2], [0.25, 0.2]]) box(0.04, 0.72, 0.04, woodDark, X2 - 6.1 + dx, 0.36, 0.45 + dz);
  makeFax(X2 - 6.1, 0.79, 0.45, 0.6);
  makeCooler(scene, X2 - 6.85, -0.2);
  makePlant(scene, X2 - 2.6, 2.1, 1.05);
  makeCabinet(scene, X2 - 6.8, -2.2, 3); makeCabinet(scene, X2 - 2.65, -2.25, 3);
  makeCoatStand(scene, X2 - 6.85, 2.25);
  // Wire baskets of paper on the back wall's side, more paper than anyone needs.
  for (const [x, z, n] of [[X2 - 2.7, 1.1, 6], [X2 - 2.75, 0.55, 4]]) for (let k = 0; k < n; k++) { const b = box(0.32, 0.05, 0.24, k % 2 ? paperM : mat(0xd8d4c6, { roughness: 0.7 }), x, 0.125 + k * 0.05, z); b.rotation.y = (k * 0.9) % 0.4 - 0.2; }

  // ---------- The annex ----------
  // Its building: brick all the way down to the street, a door at the bottom and sash windows on the storey above it.
  const AW = ANNEX_X1 - ANNEX_X0, AX = (ANNEX_X0 + ANNEX_X1) / 2;
  const below = box(AW, -S - 0.02, 8, mat(0xffffff, { map: rep(brickTex, AW, -S), roughness: 0.95 }), AX, S / 2 - 0.01, -1.4);
  below.castShadow = false;
  box(AW + 0.08, 0.2, 0.08, mat(0x4a3d33), AX, -0.05, 2.62);
  for (const [wx, wy] of [[-1.3, -1.25], [0, -1.25], [1.3, -1.25]]) {
    box(0.74, 1.24, 0.03, mat(0x2a2320), AX + wx, wy, 2.61);
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 1.12), new THREE.MeshBasicMaterial({ color: wx === 0 ? 0xc99f5e : 0x1c2028 })); pane.position.set(AX + wx, wy, 2.63); scene.add(pane);
    box(0.62, 0.03, 0.02, mat(0x2a2320), AX + wx, wy, 2.64);
    box(0.8, 0.05, 0.08, mat(0x6e675c), AX + wx, wy - 0.65, 2.63);
  }
  box(0.9, 1.9, 0.05, mat(0x2d3a30), AX + 1.0, S + 1.07, 2.63);
  sign("THE ANNEX", 1.2, 0.26, AX + 1.0, S + 2.3, 2.63, "#d9c9a8", "#3a2a22");
  const notice = canvasTex(256, 128, (g, w, h) => { g.fillStyle = "#c9bd9a"; g.fillRect(0, 0, w, h); g.fillStyle = "#2a241c"; g.textAlign = "center"; g.font = "700 34px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("NO VISITORS", w / 2, 50); g.font = "500 28px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("(THERE NEVER ARE)", w / 2, 92); });
  const noticeM = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.31), new THREE.MeshStandardMaterial({ map: notice, roughness: 0.7 })); noticeM.position.set(AX - 0.9, S + 1.5, 2.63); scene.add(noticeM);
  // The room: floorboards, brick, every manual nobody visits, warm lamps.
  box(AW, 0.1, 5.2, mat(0xffffff, { map: rep(boardsTex, AW, 5.2), roughness: 0.8 }), AX, 0.05, 0);
  box(AW, H2, 0.12, mat(0xffffff, { map: rep(brickTex, AW, H2), roughness: 0.95 }), AX, H2 / 2, -2.6);
  sideWall(ANNEX_X1, mat(0xffffff, { map: brickTex, roughness: 0.95 }), H2);
  sideWall(ANNEX_X0, mat(0xffffff, { map: brickTex, roughness: 0.95 }), 1.1, 0.5);
  makeLibrary(scene, AX, 1.85, -2.53, AW - 0.6, 3.1, 5);
  makeLadder(scene, AX + 1.5, -2.25);
  sign("ANNEX", 1.4, 0.42, AX, 3.95, -2.52, "#d9c9a8", "#3a2a22");
  makeRug(scene, ANNEX_X - 0.1, 0.104, 0.1, 2.2, 3.8);
  const warm = new THREE.PointLight(0xffc27a, 0.8, 8, 1.6); warm.position.set(AX, 3.2, 0.4); scene.add(warm);
  roomDesk(scene, ANNEX_X, 1.35, true, { books: false });
  makeChair(ANNEX_X, 0.15);
  makeFax(X2 + 6.75, 1.07, 1.1, 0.75);
  beads = makeAbacus(scene, ANNEX_X + 0.12, 1.03, 1.08, 0.1);
  for (const [bx, bz, n] of [[X2 + 7.85, 0.4, 5], [X2 + 7.82, -0.25, 3], [X2 + 4.35, -1.3, 4], [X2 + 4.4, 1.9, 6]]) for (let k = 0; k < n; k++) {
    const b = box(0.34 - (k % 2) * 0.04, 0.07, 0.26, mat([0x6b3a32, 0x2f4a5c, 0x7a6a3a, 0x4f6a63][(k + n) % 4]), bx, 0.135 + k * 0.07, bz); b.rotation.y = (k * 0.7) % 0.5 - 0.25;
  }
  cat = makeCat(scene, X2 + 7.85, 0.47, 0.4);
  // The old building carries on up: one more storey of brick, a cornice and chimneys.
  box(AW, 3.0, 5, mat(0xffffff, { map: rep(brickTex, AW, 3), roughness: 0.95 }), AX, H2 + 1.5, -5.2);
  box(AW + 0.3, 0.2, 5.3, mat(0x4a3d33), AX, H2 + 3.1, -5.2);
  for (const dx of [-1.5, 1.5]) { box(0.6, 1.3, 0.6, mat(0xffffff, { map: rep(brickTex, 0.6, 1.3), roughness: 0.95 }), AX + dx, H2 + 3.8, -5.2); for (const p of [-0.13, 0.13]) { const pot = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 0.3, 10), mat(0x7a4a36)); pot.position.set(AX + dx + p, H2 + 4.6, -5.2); scene.add(pot); } }
  for (const dx of [-1.3, 0, 1.3]) {
    const w = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 1.2), new THREE.MeshBasicMaterial({ color: dx === 1.3 ? 0xc99f5e : 0x1c2028 })); w.position.set(AX + dx, H2 + 1.5, -2.68); scene.add(w);
    box(0.74, 0.05, 0.04, mat(0x6e675c), AX + dx, H2 + 0.87, -2.68);
  }

  // The telephone line between the buildings, on two short masts.
  for (const [px, base] of [[X2 - 0.5, H2 + 0.1], [ANNEX_X0 + 0.1, H2]]) {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 0.7, 8), mat(0x3b3029));
    p.position.set(px, base + 0.35, WIRE_Z); p.castShadow = true; scene.add(p);
    box(0.5, 0.05, 0.05, mat(0x3b3029), px, base + 0.62, WIRE_Z);
  }
  const sag = new THREE.QuadraticBezierCurve3(new THREE.Vector3(X2 - 0.5, H2 + 0.68, WIRE_Z), new THREE.Vector3(X2 + 1.8, H2 - 0.5, WIRE_Z), new THREE.Vector3(ANNEX_X0 + 0.1, H2 + 0.62, WIRE_Z));
  wireLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(sag.getPoints(40)), wireMat);
  scene.add(wireLine);

  cityUpdate = city(scene, X2, 2, { near: -14 });
})();

// The hoist: a beam out over the pavement, a chain, and a pallet that carries the annex adviser and the manuals down.
const HOIST_Z = 3.0, HOIST_TOP = H2 - 0.5;
box(0.12, 0.16, 5.95, mat(0x2a2e36, { metalness: 0.5, roughness: 0.5 }), VAN_ANNEX, HOIST_TOP + 0.08, 0.5);
const { pallet, set: setHoist } = makeHoist(scene, VAN_ANNEX, HOIST_Z, HOIST_TOP);
const palletCrates = [];
for (const [cx, cz] of [[-0.42, -0.08], [0.42, -0.08], [-0.42, 0.16], [0.42, 0.16]]) {
  const c = new THREE.Group(); c.position.set(cx, 0.17, cz); c.scale.setScalar(0.72); pallet.add(c);
  box(0.42, 0.4, 0.4, mat(0x8a6a44, { roughness: 0.9 }), 0, 0.03, 0, c);
  for (const dy of [-0.09, 0.15]) box(0.44, 0.04, 0.42, mat(0x5e4630, { roughness: 0.9 }), 0, dy, 0, c);
  palletCrates.push(c);
}
const HOIST_UP = 0.12, HOIST_DOWN = S + 0.16;

// A clerk with a stack of paper, for ever crossing the office.
const clerk = makeAdviser({ x: X2 - 6.4, z: -0.25, suit: 0x7a5a5a, hair: 0x2a2020, tie: 0x4f8fd0, nodPhase: 0.4 });
clerk.carry = true;
const clerkStack = box(0.34, 0.2, 0.24, paperM, 0, 1.24, 0.42, clerk.group); clerkStack.castShadow = false;

// Two cars that get to the red light first. Each is built with its nose towards -x.
function makeCar(color) {
  const g = new THREE.Group(); street.add(g);
  const body = mat(color, { roughness: 0.4, metalness: 0.2 }), glassC = mat(0x1b2230, { roughness: 0.2, metalness: 0.4 });
  box(1.5, 0.3, 0.72, body, 0, 0.32, 0, g);
  box(0.82, 0.25, 0.68, glassC, 0.08, 0.58, 0, g);
  box(0.74, 0.03, 0.66, body, 0.08, 0.71, 0, g);
  for (const hz of [-0.26, 0.26]) {
    box(0.02, 0.07, 0.12, mat(0xfff2c9, { emissive: 0xfff2c9, emissiveIntensity: 1.2 }), -0.76, 0.36, hz, g);
    box(0.02, 0.07, 0.12, mat(0xd2412f, { emissive: 0xd2412f, emissiveIntensity: 1 }), 0.76, 0.38, hz, g);
  }
  for (const [wx, wz] of [[-0.48, -0.37], [-0.48, 0.37], [0.48, -0.37], [0.48, 0.37]]) { const w = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.08, 14), mat(0x1c2027)); w.rotation.x = Math.PI / 2; w.position.set(wx, 0.15, wz); g.add(w); }
  g.visible = false;
  return g;
}
const queueCars = [makeCar(0x8a3a32), makeCar(0x3f5a7a)];
const CAR_Z = VAN_Z + 0.56;

function setTraffic(on) { for (const [n, l] of Object.entries(trafficLamps)) { l.m.emissiveIntensity = n === on ? 1.4 : 0.05; l.halo.material.opacity = n === on ? 0.8 : 0; } }

const M = makeAdviser({ x: MAIN_X, z: 0.45 });
const N = makeAdviser({ x: ANNEX_X, z: 0.45, suit: 0x6b5a48, hair: 0xc4c4bf, tie: null, nodPhase: 0.7, old: true });

// The van, whose roof rack carries the four crates of manuals.
const { group: van, body: vanBody, wheels, lights: vanLights, brake } = makeVan(); street.add(van);
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
  new THREE.Vector3(X2 - 6.1, 1.05, 0.45), new THREE.Vector3(X2 - 6.0, 3.5, 0.3), new THREE.Vector3(X2 - 0.5, H2 + 0.74, WIRE_Z),
  new THREE.Vector3(X2 + 1.8, H2 + 0.12, WIRE_Z), new THREE.Vector3(ANNEX_X0 + 0.1, H2 + 0.68, WIRE_Z), new THREE.Vector3(X2 + 6.7, 3.5, 1.0),
  new THREE.Vector3(X2 + 6.75, 1.3, 1.1),
]);
const E2 = makeEnvelope();
// The courier leaves it from the lift.
const ENV2_IN = new THREE.Vector3(MAIN_X - 0.2, 1.05, 1.5), ENV2_OUT = new THREE.Vector3(LIFT_X, 1.35, LIFT_Z + 0.2);
E2.group.position.copy(ENV2_OUT); E2.group.visible = false;

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
  E2.group.position.copy(ENV2_OUT); E2.group.visible = false; E2.stamp.visible = false;
  setHoist(HOIST_UP); for (const p of palletCrates) p.visible = true;
  for (const q of queueCars) q.visible = false;
}

const camera2 = { look: new THREE.Vector3(X2 + 0.5, -1.2, 1.6), offset: true, fit: 8.25, fog: [20, 90], par: 1.6 };

// Backs the camera off until `fit` units either side of the look point are in view, at any aspect.
const fitPos = new THREE.Vector3();
// On wide screens the view centres on the part the side panel leaves clear.
function fitCam(C) {
  const side = innerWidth > 720 ? 350 : 0, aspect = (innerWidth - side) / innerHeight;
  camera.setViewOffset(innerWidth, innerHeight, side / 2, aspect < 1 ? innerHeight * 0.03 : -innerHeight * 0.1, innerWidth, innerHeight);
  const half = Math.tan(camera.fov * Math.PI / 360);
  const d = Math.max(C.fit / (half * aspect), 6.6 / half) * 1.05;
  scene.fog.near = d * 1.1; scene.fog.far = d * 4.4;
  return fitPos.set(C.look.x, C.look.y + d * 0.3, C.look.z + d);
}

camera2.place = () => fitCam(camera2);
const anchors = [
  [$("aMain"), () => new THREE.Vector3(X2 - 4.7, 4.45, -2.5)],
  [$("aAnnex"), () => new THREE.Vector3(X2 + 6.1, 4.45, -2.5)],
  [$("aExpert"), () => up(N.group, 2.2)],
  [$("aFax"), () => new THREE.Vector3(X2 + 1.6, H2 - 0.75, WIRE_Z)],
  [$("aVan"), () => up(van, S + 2.0)],
];

// Floor 2: a counted stretch of moving; the clock shows `moved` so far plus this stretch, and `thought` so far.
function counter() {
  const c = { moved: 0, thought: 0 };
  c.seg = (s, fn) => tween(s, (u) => { if (fn) fn(u); setClock(c.moved + s * u, c.thought); }).then(() => { c.moved += s; });
  return c;
}
async function envelopeIn() {
  $("caption").hidden = true;   // the street runs under it
  setClock(0, 0);
  E2.group.visible = true;
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
  // Two cars come up the near lane and take the head of the queue at the light.
  queueCars.forEach((q, i) => { q.visible = true; q.position.set(X2 + 14 + i * 3, 0, CAR_Z); });
  const carAt = [VAN_LIGHT - 3.65, VAN_LIGHT - 1.95];
  tween(2.4, (u) => queueCars.forEach((q, i) => { q.position.x = lerp(X2 + 14 + i * 3, carAt[i], Math.sin(u * Math.PI / 2)); }));
  await c.seg(1.2, (u) => {
    van.position.x = lerp(VAN_OFFICE, VAN_ANNEX, ease(u));
    vanRide(u, u < 0.5 ? 0.035 : -0.045); brake(u > 0.6);
  });
  brake(false);
  // N walks out onto the hoist's pallet and rides down with the manuals.
  await Promise.all([c.seg(1.0), (async () => {
    await walkTo(N, VAN_ANNEX, HOIST_Z, 0.42);
    N.heading = 0;
    await tween(0.58, (u) => { const e = ease(u), y = lerp(HOIST_UP, HOIST_DOWN, e); setHoist(y); N.lift = y - 0.02; });
  })()]);
  // N climbs in at the kerb side and the van rocks; the crates hop onto the rack as it U-turns into the near lane.
  N.heading = 0; N.walking = 1;
  await c.seg(0.8, (u) => {
    const inU = ease(Math.min(1, u / 0.3));
    N.group.position.set(VAN_ANNEX, 0, HOIST_Z + 0.4 * inU); N.lift = HOIST_DOWN - 0.02 + 0.35 * inU; N.walkPhase += 0.2;
    if (u >= 0.3) N.group.visible = false;
    vanBody.rotation.x = -0.05 * Math.sin(inU * Math.PI);
    crates.forEach((cr, i) => {
      const s = Math.min(1, Math.max(0, (u - 0.1) * 5 - i));
      if (s > 0) palletCrates[i].visible = false;
      cr.visible = s > 0; cr.scale.setScalar(s > 0 ? backOut(s) : 1); cr.position.y = (i === 3 ? 1.94 : 1.54) + (1 - s) * 0.5;
    });
    const a = Math.PI * ease(Math.max(0, (u - 0.3) / 0.7));
    van.rotation.y = -a;
    van.position.set(VAN_ANNEX + TURN_R * Math.sin(a), 0, VAN_Z + TURN_R * (1 - Math.cos(a)));
    vanBody.position.y = -0.04 * Math.min(1, u * 4);
  });
  N.walking = 0; N.lift = 0;
  vanBody.scale.y = 0.94; vanBody.position.y = -0.04;
  tween(1.0, (u) => setHoist(lerp(HOIST_DOWN, HOIST_UP, ease(u))));
  setTraffic("amber");
  walkTo(M, MAIN_X - 0.95, 0.55, 0.8);
  await c.seg(1.0, (u) => {
    van.position.x = lerp(VAN_ANNEX, VAN_LIGHT, Math.sin(u * Math.PI / 2));
    if (u > 0.5) setTraffic("red");
    vanRide(u, u > 0.5 ? -0.05 : 0); brake(u > 0.45);
  });
  await c.seg(0.9, (u) => { vanBody.position.y = -0.04 + Math.sin(u * 60) * 0.006; vanBody.rotation.z = 0; if (u > 0.85) setTraffic("green"); });
  setTraffic("green");
  // Green: the queue clears, and the van follows it to the tower's door.
  tween(1.1, (u) => queueCars.forEach((q, i) => { q.position.x = lerp(carAt[i], carAt[i] - 16, u * u); }));
  await c.seg(0.9, (u) => {
    van.position.x = lerp(VAN_LIGHT, VAN_OFFICE, ease(u));
    vanRide(u, u < 0.5 ? 0.035 : -0.045); brake(u > 0.65);
  });
  brake(false); vanLights(false); vanBody.rotation.z = 0;
  M.heading = 0;
  // N hops out on the pavement side, goes in at the tower's door, and comes out of the lift upstairs.
  N.group.position.set(VAN_OFFICE, 0, van.position.z - 0.5); N.group.rotation.y = N.heading = Math.PI; N.lift = S + 0.35; N.group.visible = true;
  tween(0.15, (u) => { N.lift = S + 0.12 + 0.23 * (1 - ease(u)); });
  await Promise.all([c.seg(1.0), (async () => {
    await walkTo(N, VAN_OFFICE, 2.72, 0.24);
    N.group.visible = false; N.lift = 0;
    await tween(0.12, (u) => lift.open(ease(u)));
    N.group.position.set(LIFT_X, 0, LIFT_Z + 0.1); N.group.visible = true;
    tween(0.3, (u) => lift.open(1 - ease(Math.max(0, u - 0.4) / 0.6)));
    await walkTo(N, MAIN_X + 1.4, 0.5, 0.34);
    await walkTo(N, MAIN_X, 0.45, 0.3);
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
closeFloor(2, opened);

export default {
  ...config, anchors, camera: camera2, lift, enter: resetFloor2, start: f2Intro,
  frame(t, dt) {
    for (const A of [M, N, clerk, ...colleagues]) poseAdviser(A, t, dt);
    cityUpdate(t);
    if (!reduced) {
      cat.scale.y = 0.62 + Math.sin(t * 1.7) * 0.025;
      // Ten seconds across and back, with a pause at each end to put down paper nobody asked for.
      const cyc = (t * 0.1) % 1, leg = cyc < 0.5 ? cyc * 2 : (1 - cyc) * 2, p = Math.min(1, Math.max(0, (leg - 0.08) / 0.84)), moving = p > 0 && p < 1;
      const px = clerk.group.position.x;
      clerk.group.position.x = lerp(X2 - 6.25, X2 - 2.85, ease(p));
      clerk.walking = moving ? 1 : 0; clerk.heading = moving ? (cyc < 0.5 ? Math.PI / 2 : -Math.PI / 2) : 0;
      clerk.walkPhase += Math.abs(clerk.group.position.x - px) * 7.5;
      // One desk's phone rings for two seconds in every seven, and nobody looks up.
      const ring = (t % 7) < 2 && (t * 2 % 1) < 0.6;
      phones[0].rotation.z = ring ? Math.sin(t * 60) * 0.05 : 0;
      clockHands.minute.rotation.z = -1.05 - t * 0.02;
      // The abacus runs while its owner thinks.
      if (N.sit > 0.9 && N.group.position.x > X2) for (const { b, rest, r, k } of beads) b.position.x = rest + (Math.sin(t * 7 + r * 1.9) > (k - 2) * 0.35 ? 0.16 : 0);
    }
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
