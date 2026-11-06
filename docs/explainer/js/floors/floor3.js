// Floor 3, musical desks, around x = X3. Which experts sit on the card follows the conversation: the player runs one
// night by hand, then the narrator runs the same night with the engine's swap rule.
import { $, box, camera, canvasTex, closeFloor, halo, haloTex, mat, openFloor, pool, reduced, rep, scene, sign, toScreen, tubes, up, v, view } from "../world.js";
import { backOut, ease, lerp, rt, tween, until } from "../motion.js";
import { bestFit, makeFramer, placeLabelsApart, placeTags, sizeOf, tagSize } from "../framing.js";
import { dressAdviser, laminate, makeAdviser, makeCabinet, makeCat, makeChair, makeClock, makeCoatStand, makeCooler, makeEnvelope, makeFax, makeFaxPage, makeLadder, makeLibrary, makePlant, makeRug, makeVan, oak, paperM, PEOPLE, poseAdviser, roomDesk, steel3, wood, woodDark, wrapAngle } from "../props.js";
import { asphaltTex, boardsTex, brickTex, carpetTex, city, curtainWall, cutawayWall, floorDrop, makeHoist, makeLift, plaster, skirtM, slabTex, traffic, trimM, wallWithDoor, windowView } from "../tower.js";
import { parkCar } from "../lift.js";
import { speak, speaking } from "../narration.js";
import { addHistory, choose, clearStage, endings, findEnding, goFloor, labels, revealButtons, setClock } from "../hud.js";

const config = {
  number: 3, name: "Musical Desks", minutes: 3, scale: 30, keys: ["Moving", "Thinking"], watch: "Time for the night's words",
  labels: ["aDesk3", "aAnnex3", "aBoard", "aMover"],
  map: [
    ["Desks", "the graphics card's room for experts: only some fit"],
    ["Annex", "main memory, which holds every expert; the CPU works out the ones not on the card"],
    ["Tally board", "how often each expert was used lately; the counts fade every round"],
    ["A move", "a swap: copying an expert onto the card, in place of a quiet one, down the same road as floor 2's van"],
    ["An intern", "the stand-in while the copy runs: the CPU reads the leaving expert from a spare copy"],
    ["The letter", "a new prompt: the experts it used count too, and a round follows it"],
  ],
  lesson: "Which experts a conversation needs changes as it goes. c0ven counts what each word uses, and every few words swaps the busiest outsiders in for the quietest residents, but only when the difference is clear. The copying runs alongside the work, so the card pauses for only a moment.",
  next: "Floor 4 · The Briefing · open now.",
};

// Two rooms of four desks, the main office and the annex, either side of an alley ("row") or, on a tall screen, one
// behind the other ("stack"). Each room is built around its own origin and layout3 places them. Slots 0-3 are the
// office's chairs and 4-7 the annex's, each room's in the order back left, back right, front left, front right.
// A desk sits 1.2 in front of its chair.
const X3 = 120, opened = openFloor();
const wireMat = new THREE.LineBasicMaterial({ color: 0x14171c });
const COL = 1.3, ROWZ = [-1.8, 0.4], RW = 6.0, RD = 5.2, LANE = 3.4;
const G3 = -floorDrop(3), ROAD_NEAR = RD / 2 + 2.6, ROAD_FAR = RD / 2 + 4.4, HOIST_UP = 0.12, HOIST_DOWN = G3 + 0.16;   // the road's lanes are in R3.road's z
const SLOTS = Array.from({ length: 8 }, () => [0, 0]);
const STAFF = PEOPLE.map((p) => ({ ...p }));
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

const sideWall = (parent, x, m, back, front) => cutawayWall(parent, x, m, -RD / 2, RD / 2, back, front);

(() => {
  const [office, annex] = rooms;
  scene.add(office, annex);
  // The street, three storeys down, and the buildings' walls falling to it.
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(90, 60), mat(0x13161b, { roughness: 1 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(X3, G3, 0); ground.receiveShadow = true; scene.add(ground);
  // The road and pavements run in front of the nearer building, so layout3 moves them with it.
  R3.road = new THREE.Group(); scene.add(R3.road);
  const road = new THREE.Mesh(new THREE.PlaneGeometry(90, 5.6), mat(0xffffff, { map: rep(asphaltTex, 40, 3), roughness: 0.5, metalness: 0.15 }));
  road.rotation.x = -Math.PI / 2; road.position.set(X3, G3 + 0.02, RD / 2 + 4.5); R3.road.add(road);
  for (let x = -44; x <= 44; x += 2.2) box(1.1, 0.02, 0.1, mat(0xd8d2b8, { roughness: 0.8 }), X3 + x, G3 + 0.04, RD / 2 + 4.5, R3.road);
  box(90, 0.12, 1.5, mat(0xffffff, { map: rep(slabTex, 90, 2), roughness: 0.9 }), X3, G3 + 0.06, RD / 2 + 0.95, R3.road);
  box(90, 0.12, 1.5, mat(0xffffff, { map: rep(slabTex, 90, 2), roughness: 0.9 }), X3, G3 + 0.06, RD / 2 + 8.05, R3.road);
  R3.traffic = traffic(R3.road, G3 + 0.1, RD / 2 + 4.5, X3 - 42, X3 + 42, 8);
  for (const x of [-16, -7, 2, 11, 20]) pool(R3.road, 0xffa850, 2.4, X3 + x, G3 + 0.14, RD / 2 + 1.2, 0.26);
  // One light on the street, so the van reads from up here.
  const kerbLight = new THREE.PointLight(0xffe2b8, 0.9, 22, 1.2); kerbLight.position.set(X3, G3 + 6, RD / 2 + 3.5); R3.road.add(kerbLight);
  R3.van = makeVan(); R3.road.add(R3.van.group);
  curtainWall(office, 0, RD / 2 + 0.01, RW, G3, -0.1, 2.5);
  box(RW, -G3 - 0.1, RD, mat(0x12151b), 0, G3 / 2 - 0.05, 0, office);
  box(RW + 0.08, 0.2, RD + 0.08, mat(0x4a4f58), 0, -0.03, 0, office);
  box(RW, -G3 - 0.1, RD, mat(0xffffff, { map: rep(brickTex, RW, -G3), roughness: 0.95 }), 0, G3 / 2 - 0.05, 0, annex);
  box(RW + 0.08, 0.2, RD + 0.08, mat(0x4a3d33), 0, -0.03, 0, annex);
  for (const r of [0, 1]) for (const wx of [-1.9, 0, 1.9]) for (const lv of [-1.3, -3.8, -6.3]) {
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 1.2), new THREE.MeshBasicMaterial({ color: (wx + lv) % 2 ? 0xc99f5e : 0x1c2028 }));
    pane.position.set(wx, lv, RD / 2 + 0.01); if (r) annex.add(pane);
    if (r) box(0.84, 0.05, 0.08, mat(0x6e675c), wx, lv - 0.63, RD / 2 + 0.02, annex);
  }
  // Above the office the tower runs on up, out of the frame; the annex's old building stops one storey on.
  R3.upper = new THREE.Group(); office.add(R3.upper);
  box(RW, 44, 7.9, mat(0x12151b), 0, 4.6 + 22, -RD / 2 - 4.1, R3.upper);
  curtainWall(R3.upper, 0, -RD / 2 - 0.05, RW, 4.6, 48.6, 2.3, 0x83878e);
  // The shaft's top and sides, so the car stays out of sight from above.
  box(2.5, 0.1, 2.4, mat(0x2a2e36), 0, 4.6, -RD / 2 - 1.2, office);
  for (const sx of [-1, 1]) box(0.1, 4.6, 2.4, mat(0x171a20), sx * 1.2, 2.3, -RD / 2 - 1.2, office);
  box(2.5, 4.6, 0.1, mat(0x171a20), 0, 2.3, -RD / 2 - 2.4, office);
  R3.annexUp = new THREE.Group(); annex.add(R3.annexUp);
  box(RW, 2.8, 4, mat(0xffffff, { map: rep(brickTex, RW, 2.8), roughness: 0.95 }), 0, 6.0, -RD / 2 - 2.0, R3.annexUp);
  box(RW + 0.3, 0.2, 4.3, mat(0x4a3d33), 0, 7.5, -RD / 2 - 2.0, R3.annexUp);
  for (const dx of [-2, 2]) box(0.7, 1.2, 0.7, mat(0xffffff, { map: rep(brickTex, 0.7, 1.2), roughness: 0.95 }), dx, 8.1, -RD / 2 - 2.0, R3.annexUp);
  // The tower's street door under a canopy, the annex's plain one, and the annex's way down: floor 2's hoist.
  box(1.0, 1.9, 0.04, mat(0x3a3020), 0, G3 + 1.07, RD / 2 + 0.03, office);
  box(2.2, 0.1, 0.7, mat(0x20252e), 0, G3 + 2.3, RD / 2 + 0.35, office);
  sign("MERIDIAN WORDS", 2.0, 0.24, 0, G3 + 2.48, RD / 2 + 0.52, "#14171c", "#f0ad45", office).material.emissive = new THREE.Color(0x4a3210);
  box(0.9, 1.9, 0.04, mat(0x2a2320), -0.95, G3 + 1.07, RD / 2 + 0.03, annex);
  const davit = mat(0x2a2e36, { metalness: 0.5, roughness: 0.5 });
  box(0.1, 3.4, 0.1, davit, 0.85, 1.7, RD / 2 + 0.08, annex);
  box(0.95, 0.1, 0.1, davit, 0.42, 3.4, RD / 2 + 0.08, annex);
  box(0.1, 0.1, 0.6, davit, 0, 3.4, RD / 2 + 0.3, annex);
  R3.hoist = makeHoist(annex, 0, RD / 2 + 0.55, 3.35);
  R3.hoist.pallet.scale.set(1.25, 1, 2.1);

  // The main office: blue-grey carpet, plaster walls, a tube overhead, windows onto the city
  box(RW, 0.1, RD, mat(0xffffff, { map: rep(carpetTex, RW, RD), roughness: 1 }), 0, 0.05, 0, office);
  // The back wall, with the lift in its middle, level with the aisle.
  wallWithDoor(office, plaster, -RW / 2, RW / 2, 4.6, -RD / 2 + 0.06, 0, 0.12, 0.1);
  for (const [a, b] of [[-RW / 2, -0.77], [0.77, RW / 2]]) {
    box(b - a, 0.07, 0.05, trimM, (a + b) / 2, 1.05, -RD / 2 + 0.08, office);
    box(b - a, 0.14, 0.03, skirtM, (a + b) / 2, 0.17, -RD / 2 + 0.075, office);
  }
  for (const sx of [-1, 1]) sideWall(office, sx * RW / 2, plaster, 4.6, 1.3);
  R3.officeSign = sign("MAIN OFFICE", 2.4, 0.42, 0, 4.22, -RD / 2 + 0.07, undefined, undefined, office);
  R3.lift = makeLift(office, { x: 0, y: 0.1, z: -RD / 2 + 0.065, number: 3, name: "Musical Desks", signSide: 0 });
  R3.officeWin = new THREE.Group(); office.add(R3.officeWin);
  for (const sx of [-1, 1]) {
    const w = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.15), new THREE.MeshBasicMaterial({ map: windowView(sx > 0 ? 31 : 17, 3) }));
    w.position.set(sx * 1.75, 2.75, -RD / 2 + 0.07); R3.officeWin.add(w);
    for (const [fw, fh, fx, fy] of [[1.5, 0.07, 0, 0.6], [1.5, 0.1, 0, -0.6], [0.07, 1.2, -0.72, 0], [0.07, 1.2, 0.72, 0], [0.04, 1.15, 0, 0]]) box(fw, fh, 0.05, trimM, sx * 1.75 + fx, 2.75 + fy, -RD / 2 + 0.09, R3.officeWin);
  }
  makeClock(office, -2.4, 3.6, -RD / 2 + 0.09);
  // Hanging tube: emissive with a glow; the room's one real light sits under it.
  const tubeM = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.05, 0.16), new THREE.MeshBasicMaterial({ color: 0xe6efe9 }));
  tubeM.position.set(0, 3.75, -0.7); office.add(tubeM); tubes.push(tubeM);
  halo(0xdfe9e4, 1.6, 0, 3.7, -0.65, office, 0.16);
  for (const wx of [-1.0, 1.0]) box(0.01, 0.85, 0.01, steel3, wx, 4.2, -0.7, office);
  const cool = new THREE.PointLight(0xdfe9e4, 0.55, 10, 1.6); cool.position.set(0, 3.5, 0.2); office.add(cool);
  pool(office, 0xc8daf0, 2.6, 0, 0.104, -0.4, 0.1);
  // Water cooler, plant, filing cabinet, coat stand
  makeCooler(office, -2.55, -2.15);
  makePlant(office, 2.55, -2.2, 1.15);
  makeCabinet(office, -2.68, -0.6, 3);
  makeCoatStand(office, 2.7, 2.25);

  // The annex: floorboards, brick, every manual nobody visits, warm lamps
  box(RW, 0.1, RD, mat(0xffffff, { map: rep(boardsTex, RW, RD), roughness: 0.8 }), 0, 0.05, 0, annex);
  R3.annexBack = new THREE.Group(); annex.add(R3.annexBack);
  box(RW, 4.6, 0.12, mat(0xffffff, { map: rep(brickTex, RW, 4.6), roughness: 0.95 }), 0, 2.3, -RD / 2, R3.annexBack);
  makeLibrary(R3.annexBack, 0, 1.75, -RD / 2 + 0.07, RW - 0.7, 3.1, 5);
  makeLadder(R3.annexBack, 2.1, -RD / 2 + 0.35);
  sign("ANNEX", 1.4, 0.42, 0, 4.22, -RD / 2 + 0.07, "#d9c9a8", "#3a2a22", R3.annexBack);
  R3.annexSides = [-1, 1].map((sx) => sideWall(annex, sx * RW / 2, mat(0xffffff, { map: brickTex, roughness: 0.95 }), 4.6, 1.3));
  // On a tall screen the annex stands in front: a low parapet with a gap at the aisle replaces its back wall.
  R3.parapet = new THREE.Group(); annex.add(R3.parapet);
  for (const sx of [-1, 1]) box(RW / 2 - 0.62, 0.5, 0.16, mat(0xffffff, { map: rep(brickTex, 2.4, 0.5), roughness: 0.95 }), sx * (RW / 4 + 0.31), 0.25, -RD / 2, R3.parapet);
  sign("ANNEX", 1.1, 0.33, -1.6, 0.72, -RD / 2 + 0.09, "#d9c9a8", "#3a2a22", R3.parapet);
  makeRug(annex, 0, 0.104, -0.3, 1.3, 3.8);
  const warm = new THREE.PointLight(0xffc27a, 0.75, 10, 1.6); warm.position.set(0, 3.4, 0.4); annex.add(warm);
  // Books piled by the wall, and the annex cat asleep on them
  for (const [bx, bz, n] of [[2.65, 0.4, 5], [2.62, -0.25, 3], [-2.66, -0.5, 4]]) for (let k = 0; k < n; k++) {
    const b = box(0.34 - (k % 2) * 0.04, 0.07, 0.26, mat([0x6b3a32, 0x2f4a5c, 0x7a6a3a, 0x4f6a63][(k + n) % 4]), bx, 0.135 + k * 0.07, bz, annex); b.rotation.y = (k * 0.7) % 0.5 - 0.25;
  }
  R3.cat = makeCat(annex, 2.65, 0.47, 0.4);

  for (const [r, room] of rooms.entries()) for (const row of ROWZ) for (const sx of [-1, 1]) roomDesk(room, sx * COL, row + 1.2, r === 1);

  // A side table with a fax machine, at each room's alley side; layout3 places them.
  R3.fax = rooms.map((room, r) => {
    const g = new THREE.Group(); room.add(g);
    box(0.6, 0.06, 0.5, r ? oak : laminate, 0, 0.75, 0, g);
    for (const [dx, dz] of [[-0.25, -0.2], [0.25, -0.2], [-0.25, 0.2], [0.25, 0.2]]) box(0.04, 0.72, 0.04, r ? oak : steel3, dx, 0.36, dz, g);
    g.add(makeFax(0, 0.79, 0, r ? 0.5 : -0.5));
    return g;
  });

  // The alley's end wall, side by side only: it closes the gap between the buildings and carries the tally board.
  R3.alley = new THREE.Group(); scene.add(R3.alley);
  box(LANE, 4.6 - G3, 0.1, mat(0xffffff, { map: rep(brickTex, LANE, 4.6 - G3), roughness: 0.95, color: 0x8a8378 }), X3, (4.6 + G3) / 2, -RD / 2 + 0.2, R3.alley);

  // Two telephone poles up from the street and the line the fax travels on; one pole carries a street lamp.
  R3.poles = [0, 1].map(() => {
    const g = new THREE.Group(); scene.add(g);
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 3.7 - G3, 10), mat(0x3b3029)); p.position.y = (3.7 + G3) / 2; p.castShadow = true; g.add(p);
    g.arm = new THREE.Group(); g.add(g.arm);
    box(0.6, 0.06, 0.06, mat(0x3b3029), 0, 3.55, 0, g.arm);
    return g;
  });
  // The lamp's arm points along its local +z; layout3 turns it into the alley.
  R3.lamp = new THREE.Group(); R3.poles[0].add(R3.lamp);
  box(0.06, 0.05, 0.55, steel3, 0, 3.15, 0.27, R3.lamp);
  box(0.26, 0.08, 0.2, steel3, 0, 3.12, 0.55, R3.lamp);
  box(0.22, 0.03, 0.16, mat(0xffd59a, { emissive: 0xffb45c, emissiveIntensity: 1.3 }), 0, 3.07, 0.55, R3.lamp);
  halo(0xffb45c, 1.4, 0, 3.02, 0.6, R3.lamp, 0.42);
  R3.lampPool = pool(scene, 0xffa850, 1.6, 0, G3 + 0.16, 0, 0.32);
  R3.wire = new THREE.Line(new THREE.BufferGeometry(), wireMat); scene.add(R3.wire);

  // The board on its frame. Side by side it hangs on the alley's end wall; one behind the other it leans against the
  // office's back wall.
  R3.board = new THREE.Group(); scene.add(R3.board);
  R3.boardFace = new THREE.Group(); R3.board.add(R3.boardFace);
  const face = new THREE.Mesh(new THREE.PlaneGeometry(BW, BH), boardMat); R3.boardFace.add(face);
  for (const [w, h, x, y] of [[BW + 0.16, 0.08, 0, BH / 2 + 0.04], [BW + 0.16, 0.08, 0, -BH / 2 - 0.04], [0.08, BH, -BW / 2 - 0.04, 0], [0.08, BH, BW / 2 + 0.04, 0]]) box(w, h, 0.07, wood, x, y, 0, R3.boardFace);
  box(BW, 0.04, 0.12, woodDark, 0, -BH / 2 - 0.08, 0.06, R3.boardFace);

  R3.city = city(scene, X3, 3, { near: -19 });
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
  dressAdviser(s.A, s.hat, s.color);
  s.A.sit = 1;
  s.slot = i;
  s.box = packBox(s.color);
  // A ring on the floor: their colour while a question waits for them, amber while picked.
  s.ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.6, 32), new THREE.MeshBasicMaterial({ color: s.color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  s.ring.rotation.x = -Math.PI / 2; scene.add(s.ring);
}
// The interns who mind a desk while its adviser is away. The player's are given no notice and panic; the narrator's
// keep typing. Each has a tag, placed with the names in dom3.
const INTERNS = [[0x6b4a2e, 0xe0c3a8], [0x2a2420, 0x8d5a3b], [0xb98a44, 0xd9b9a0], [0x3b322b, 0xb98a64]].map(([hair, skin], i) => {
  const chair = makeChair(0, 0), A = makeAdviser({ suit: 0xd9d6cb, hair, skin, tie: 0xf2c230, nodPhase: i * 1.7 });
  A.sit = 1; chair.visible = A.group.visible = false;
  return { chair, A, on: false, slot: -1, tag: null };
});
const internAt = (slot) => INTERNS.find((n) => n.on && n.slot === slot);
function seatIntern(slot, player, watching) {
  const n = INTERNS.find((i) => !i.on);
  if (!n) return;
  const [x, z] = SLOTS[slot];
  n.on = true; n.slot = slot; n.tag.style.display = "";
  n.A.typing = n.A.busy = !player; n.A.cheer = player; n.A.watch = player ? watching : null; n.A.heading = 0; n.A.lift = 0;
  n.chair.position.set(x, 0, z); n.A.group.position.set(x, 0, z + 0.05); n.A.group.rotation.set(0, 0, 0);
  n.chair.visible = n.A.group.visible = true;
  puff(x, z);
  const size = (k) => { n.chair.scale.setScalar(k); n.A.group.scale.setScalar(k); };
  if (reduced) size(0.92); else tween(0.25, (u) => size(0.92 * Math.max(0.01, backOut(u))));
}
function dismissIntern(n) {
  if (!n.on) return;
  n.on = false; n.tag.style.display = "none";
  puff(n.chair.position.x, n.chair.position.z);
  const size = (k) => { n.chair.scale.setScalar(k); n.A.group.scale.setScalar(k); };
  const gone = () => { n.chair.visible = n.A.group.visible = false; };
  if (reduced) gone(); else tween(0.16, (u) => size(0.92 * Math.max(0.01, 1 - u))).then(gone);
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
  L3.aisle = [o[0], a[0]]; L3.o = o; L3.a = a;
  for (let i = 0; i < 8; i++) {
    const [rx, rz] = i < 4 ? o : a;
    SLOTS[i][0] = rx + (i % 2 ? COL : -COL); SLOTS[i][1] = rz + ROWZ[(i >> 1) & 1];
  }
  R3.annexBack.visible = R3.annexUp.visible = row; R3.parapet.visible = !row;
  for (const w of R3.annexSides) w.scale.y = row ? 1 : 0.3;
  R3.officeWin.visible = R3.officeSign.visible = row;
  R3.alley.visible = row;
  R3.road.position.z = row ? 0 : a[1];
  // The van's stops and runs, in the road's coordinates. One behind the other, the tower's door is round the corner.
  const s0 = row ? o[0] : X3 - 14, s1 = row ? a[0] : X3, at = (x, z) => new THREE.Vector3(x, 0, z);
  L3.stops = [s0, s1];
  L3.vanOut = new THREE.LineCurve3(at(s0, ROAD_NEAR), at(s1, ROAD_NEAR));
  L3.vanBack = row ? new THREE.CatmullRomCurve3([at(s1, ROAD_NEAR), at(s1 + 0.8, ROAD_NEAR), at(s1 + 1.8, ROAD_NEAR + 0.9), at(s1 + 0.8, ROAD_FAR), at(s0 - 0.8, ROAD_FAR), at(s0 - 1.8, ROAD_NEAR + 0.9), at(s0 - 0.8, ROAD_NEAR), at(s0, ROAD_NEAR)])
    : new THREE.LineCurve3(at(s1, ROAD_NEAR), at(X3 + 14, ROAD_NEAR));
  parkVan();
  // The fax tables sit at each room's alley side.
  R3.fax[0].position.set(RW / 2 - 0.45, 0, 1.95); R3.fax[0].rotation.y = 0;
  R3.fax[1].position.set(row ? -(RW / 2 - 0.45) : RW / 2 - 0.45, 0, row ? 1.95 : -1.95); R3.fax[1].rotation.y = 0;
  MFAX.set(o[0] + R3.fax[0].position.x, 1.05, o[1] + R3.fax[0].position.z);
  AFAX.set(a[0] + R3.fax[1].position.x, 1.05, a[1] + R3.fax[1].position.z);
  const P = row ? [[X3 - LANE / 2 + 0.12, 2.3], [X3 + LANE / 2 - 0.12, 2.3]] : [[X3 + RW / 2 + 0.3, -1.1], [X3 + RW / 2 + 0.3, 1.1]];
  // Cross-arms square to the wire; the lamp leans over the alley, or over the rooms' side.
  R3.poles.forEach((p, k) => { p.position.set(P[k][0], 0, P[k][1]); p.arm.rotation.y = row ? Math.PI / 2 : 0; });
  R3.lamp.rotation.y = row ? Math.PI / 2 : -Math.PI / 2;
  R3.lampPool.position.set(P[0][0] + (row ? 0.55 : -0.55), G3 + 0.16, P[0][1]);
  const p0 = new THREE.Vector3(P[0][0], 3.55, P[0][1]), p1 = new THREE.Vector3(P[1][0], 3.55, P[1][1]);
  const mid = p0.clone().lerp(p1, 0.5); mid.y = 3.15;
  R3.wire.geometry.dispose();
  R3.wire.geometry = new THREE.BufferGeometry().setFromPoints(new THREE.QuadraticBezierCurve3(p0, mid, p1).getPoints(30));
  const rise = (f, p) => new THREE.Vector3(lerp(f.x, p.x, 0.7), 2.9, lerp(f.z, p.z, 0.7));
  L3.faxRoute = new THREE.CatmullRomCurve3([MFAX.clone(), rise(MFAX, p0), p0.clone().setY(3.62), mid.clone().setY(3.3), p1.clone().setY(3.62), rise(AFAX, p1), AFAX.clone()]);
  // The board: on the alley's end wall, or leaning on the office's back wall.
  R3.board.position.set(X3, 0, row ? -RD / 2 + 0.35 : o[1] - RD / 2 + 0.22);
  R3.boardFace.position.y = row ? 3.0 : 3.5;
  R3.boardFace.rotation.x = row ? 0 : -0.16;
  L3.board.set(X3, R3.boardFace.position.y, R3.board.position.z + 0.08);
  L3.office.set(o[0], 4.5, o[1] - RD / 2 + 0.1);
  L3.annex.set(row ? a[0] : a[0] - 1.6, row ? 4.5 : 0.95, a[1] - RD / 2 + 0.1);
}
function parkVan() {
  const V = R3.van, p = L3.vanOut.getPointAt(0);
  V.group.position.set(p.x, G3 + 0.03, p.z); V.group.rotation.y = 0; V.body.rotation.set(0, 0, 0); V.body.position.y = 0;
  V.lights(false); V.brake(false);
}
function placeAt(s, i) {
  const [x, z] = SLOTS[i];
  s.slot = i;
  s.chair.visible = s.A.group.visible = true;
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
  for (const n of INTERNS) dismissIntern(n);
  fit3.wide = false; R3.hoist.set(HOIST_UP); R3.lift.show(3); parkVan();
  E3.group.visible = false; page3.visible = false; letter3.visible = false;
  L3.need = null; L3.letterMark = false; L3.hot.clear();
}

// ---------- Camera ----------
const camera3 = { look: new THREE.Vector3(X3, 1.3, 0.6), offset: true, fog: [20, 90], par: 0.7, fov: 36 };

// A trip pulls the camera back to the wide outline, which adds the street.
const FIT3 = { row: { pitch: 0.44, pts: [], wide: [] }, stack: { pitch: 0.82, pts: [], wide: [] } };
for (const [m, hx, z0, z1, hb, hf, zs] of [["row", 7.75, -2.6, RD / 2 + 0.3, 4.7, 1.8, ROAD_NEAR + 1], ["stack", RW / 2 + 0.4, -RD - 0.9, RD + 0.9, 4.7, 1.6, RD / 2 + 0.9 + ROAD_NEAR + 1]])
  for (const sx of [-1, 1]) {
    FIT3[m].pts.push(new THREE.Vector3(X3 + sx * hx, 0, z0), new THREE.Vector3(X3 + sx * hx, hb, z0), new THREE.Vector3(X3 + sx * hx, 0, z1), new THREE.Vector3(X3 + sx * hx, hf, z1));
    FIT3[m].wide.push(...FIT3[m].pts.slice(-4), new THREE.Vector3(X3 + sx * hx, G3, zs));
  }
const FOV3 = 36, outline3 = (m, wide) => ({ pitch: FIT3[m].pitch, pts: wide ? FIT3[m].wide : FIT3[m].pts });
const fit3 = makeFramer(camera3, FOV3, () => outline3(L3.mode, fit3.wide));
fit3.wide = false;
// One behind the other wins only where it shows the rooms clearly larger.
const pickLayout3 = () => (bestFit(outline3("stack"), FOV3).scale > bestFit(outline3("row"), FOV3).scale * 1.12 ? "stack" : "row");
camera3.place = fit3.place;

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
    const [rx, rz] = s.moving ? SLOTS[s.slot] : [s.chair.position.x, s.chair.position.z];
    s.ring.position.set(rx, 0.11, rz + 0.35);
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
  if (n) {
    const [bx, bz] = n.moving ? SLOTS[n.slot] : [n.A.group.position.x, n.A.group.position.z];
    beam3.position.set(bx, 1.75, bz + 0.1); beam3.material.color.setHex(n.color);
  }
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
      return { el: s.tag, x, w, h, y: front ? y : y - h };
    });
    for (const n of INTERNS) if (n.on) {
      const g = n.A.group.position, [x, y] = toScreen(v.set(g.x, 2.0, g.z)), [w, h] = sizeOf(n.tag);
      items.push({ el: n.tag, x, w, h, y: y - h });
    }
    placeTags(items);
  }
  placeLabelsApart(anchors3);
}

// The night: who each question needs, by STAFF index. The first twelve spread over the office; then a letter
// arrives and the questions turn legal.
const NIGHT = [0, 2, 1, 0, 2, 3, 0, 2, 6, 0, 2, 1, 4, 5, 4, 4, 5, 0, 4, 5, 2, 4, 5, 4, 0, 5, 4, 5];
const SHIFT_AT = 12;
// Seconds on the stopwatch. A desk answers in LOCAL; a fax goes out, is thought over in the annex, and comes back.
// The player's swap stops the office for SWAP; the narrator's round holds it for ROUND, its packing done behind.
const T3 = { LOCAL: 0.5, FAX_OUT: 0.35, FAX_THINK: 0.9, SWAP: 1.0, ROUND: 0.1 };
// With reduced motion the narrator's trip is a wait of ZIP, and the player's a wait of SWAP.
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
for (const n of INTERNS) {
  n.tag = document.createElement("span"); n.tag.className = "tag intern"; n.tag.textContent = "Intern"; n.tag.style.display = "none";
  tagsBox.appendChild(n.tag);
}
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

// ---------- The trip ----------
// A swap goes by road, as floor 2's crates do: office staff take the lift, annex staff the hoist, and the van runs
// between the two while an intern minds each desk. TRIP is seconds on screen; the stopwatch charges the player's as
// SWAP. The player's chair sprouts legs and scurries, its rider's arms up; the narrator's glides with the box on the lap.
const TRIP = 3, LEG3 = 0.78;
const eased = (t, a, b) => ease(Math.min(1, Math.max(0, (t - a) / (b - a))));
// A mover's route: legs in order, each lasting until t1 seconds into the trip, along pts [x, y, z], y being the level
// underfoot. `hide` is inside the lift, `van` inside the van.
function tripLegs(from, to, k) {
  const [x0, z0] = SLOTS[from], [x1, z1] = SLOTS[to], off = k * 0.3, [ox, oz] = L3.o, [ax, az] = L3.a, [s0, s1] = L3.stops;
  const door = oz - RD / 2 + 0.75, car = oz - RD / 2 - 0.55, gate = oz + RD / 2 + 0.12, edge = az + RD / 2 - 0.2;
  const px = ax + off, pz = az + RD / 2 + 0.55, kerb = R3.road.position.z + ROAD_NEAR - 0.6;
  const pave = G3 + 0.14, road = G3 + 0.03, up = HOIST_UP - 0.02, down = HOIST_DOWN - 0.02;
  if (from < 4) return [
    { t1: 0.5, run: true, pts: [[x0, 0, z0], [ox + off, 0, z0], [ox + off, 0, door]] },
    { t1: 0.62, run: true, pts: [[ox + off, 0, door], [ox + off, 0, car]] },
    { t1: 1.0, hide: true, pts: [[ox, 0, car], [ox, G3, car]] },
    { t1: 1.12, run: true, pts: [[s0 + off, pave, gate], [s0 + off, road, kerb]] },
    { t1: 1.62, van: true },
    { t1: 1.74, run: true, pts: [[s1 + off, road, kerb], [px, down, pz]] },
    { t1: 1.86, pts: [[px, down, pz]] },
    { t1: 2.35, pts: [[px, down, pz], [px, up, pz]] },
    { t1: 2.85, run: true, pts: [[px, up, pz], [ax + off, 0, edge], [ax + off, 0, z1], [x1, 0, z1]] },
  ];
  return [
    { t1: 0.45, run: true, pts: [[x0, 0, z0], [ax + off, 0, z0], [ax + off, 0, edge], [px, up, pz]] },
    { t1: 0.5, pts: [[px, up, pz]] },
    { t1: 1.05, pts: [[px, up, pz], [px, down, pz]] },
    { t1: 1.62, pts: [[px, down, pz]] },
    { t1: 1.74, run: true, pts: [[px, down, pz], [s1 + off, road, kerb]] },
    { t1: 2.4, van: true },
    { t1: 2.5, run: true, pts: [[s0 + off, road, kerb], [s0 + off, pave, gate]] },
    { t1: 2.72, hide: true, pts: [[ox, G3, car], [ox, 0, car]] },
    { t1: TRIP, run: true, pts: [[ox + off, 0, car], [ox + off, 0, door], [ox + off, 0, z1], [x1, 0, z1]] },
  ];
}
// The point a fraction f of the way along a list of points.
function along(pts, f) {
  let total = 0;
  for (let n = 1; n < pts.length; n++) total += Math.hypot(pts[n][0] - pts[n - 1][0], pts[n][1] - pts[n - 1][1], pts[n][2] - pts[n - 1][2]);
  let d = f * total;
  for (let n = 1; n < pts.length; n++) {
    const a = pts[n - 1], b = pts[n], len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    if (d <= len && len) { const g = d / len; return [lerp(a[0], b[0], g), lerp(a[1], b[1], g), lerp(a[2], b[2], g)]; }
    d -= len;
  }
  return pts[pts.length - 1];
}
function moveAlong(m, t, dt, player) {
  const { s } = m;
  let i = 0;
  while (i < m.legs.length - 1 && t >= m.legs[i].t1) i++;
  const leg = m.legs[i], hidden = leg.hide || leg.van;
  s.chair.visible = s.A.group.visible = !hidden;
  s.box.group.visible = !hidden && !player;
  let x, y, z;
  if (leg.van) { const g = R3.van.group.position; x = g.x; y = G3; z = g.z + R3.road.position.z; }
  else [x, y, z] = along(leg.pts, eased(t, i ? m.legs[i - 1].t1 : 0, leg.t1));
  const step = Math.hypot(x - m.x, z - m.z), want = step > 0.001 && !hidden ? Math.atan2(x - m.x, z - m.z) : 0;
  m.x = x; m.z = z;
  m.h += wrapAngle(want - m.h) * Math.min(1, dt * 16);
  m.ph += step * 8;
  let lift, lean;
  if (player) {
    m.len += ((leg.run ? LEG3 : 0) - m.len) * Math.min(1, dt * 14);
    const amp = m.len / LEG3;
    setChairLegs(s.legs, m.len);
    for (const l of s.legs) l.pivot.rotation.x = Math.sin(m.ph + l.phase) * 0.6 * amp;
    lift = Math.max(0, m.len - 0.44) + Math.abs(Math.sin(m.ph)) * 0.06 * amp;
    lean = 0.12 * amp;
    s.A.cheer = !!leg.run;
  } else {
    const glide = leg.run ? 1 : 0;
    lift = 0.08 * glide; lean = 0.2 * glide;
    m.streak.position.set(x - Math.sin(m.h) * 0.7, y + 0.75, z - Math.cos(m.h) * 0.7);
    m.streak.material.opacity = hidden ? 0 : Math.min(0.55, step / Math.max(dt, 0.001) / 40);
    const a = v.set(x, y + 0.75, z).project(camera), b2 = new THREE.Vector3(x - Math.sin(m.h), y + 0.75, z - Math.cos(m.h)).project(camera);
    m.streak.material.rotation = Math.atan2((a.y - b2.y) / camera.aspect, a.x - b2.x);
    s.box.group.position.set(x + Math.sin(m.h) * 0.25, y + 0.92 + lift, z + Math.cos(m.h) * 0.25);
    s.box.group.rotation.set(0, m.h, 0);
  }
  s.chair.position.set(x, y + lift, z); s.chair.rotation.set(lean, m.h, 0);
  s.A.group.position.set(x + Math.sin(m.h) * 0.05, 0, z + Math.cos(m.h) * 0.05);
  s.A.group.rotation.set(lean, m.h, 0); s.A.heading = m.h; s.A.lift = y + lift;
}
// The van's two runs: out along the near lane to the annex, and back round by the far lane.
function driveVan(t, dt) {
  const V = R3.van, out = eased(t, 1.12, 1.62), back = eased(t, 1.74, 2.4), running = back > 0 && back < 1 ? L3.vanBack : L3.vanOut;
  const u = running === L3.vanBack ? back : back >= 1 ? 0 : out, p = running.getPointAt(u), dir = running.getTangentAt(Math.min(0.999, Math.max(0.001, u)));
  const moved = Math.hypot(p.x - V.group.position.x, p.z - V.group.position.z);
  for (const w of V.wheels) w.rotation.z -= moved / 0.22;
  V.group.position.set(p.x, G3 + 0.03, p.z);
  V.group.rotation.y = -Math.atan2(dir.z, dir.x);
  const boarding = Math.sin(Math.PI * Math.min(1, Math.max(0, (t - 1.04) / 0.14))) + Math.sin(Math.PI * Math.min(1, Math.max(0, (t - 1.66) / 0.14)));
  V.body.rotation.x = -0.05 * boarding;
  V.body.rotation.z = moved > 0.001 ? (u < 0.5 ? 0.035 : -0.045) * Math.sin(u * Math.PI) : 0;
  V.lights(t > 1.0 && t < 2.5); V.brake(u > 0.7 && moved > 0.001);
}
async function trip(run, style, clockFn) {
  const player = style === "player";
  const movers = run.map(([s, from, to, k]) => {
    s.moving = true; s.ready = 0; s.A.busy = false; s.A.watch = null;
    if (player) puff(SLOTS[from][0], SLOTS[from][1]);
    return { s, from, to, legs: tripLegs(from, to, k), h: 0, len: 0, ph: 0, x: SLOTS[from][0], z: SLOTS[from][1], streak: streaks[STAFF.indexOf(s)] };
  });
  const seat = () => { for (const m of movers) seatIntern(m.from, player, movers[0].s.chair.position); };
  if (reduced) {
    for (const m of movers) { m.s.chair.visible = m.s.A.group.visible = m.s.box.group.visible = false; }
    seat();
    await rt(player ? T3.SWAP : ZIP, (u) => { if (clockFn) clockFn(u); });
  } else {
    parkCar({ lift: R3.lift, number: 3 });
    fit3.wide = true;
    let last = performance.now(), seated = false, left = false;
    await rt(TRIP, (u) => {
      const t = u * TRIP, now = performance.now(), dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (!seated && t > 0.12) { seated = true; seat(); }
      if (!left && t > 2.5) { left = true; fit3.wide = false; for (const n of INTERNS) dismissIntern(n); }
      R3.hoist.set(lerp(HOIST_UP, HOIST_DOWN, eased(t, 0.5, 1.05) - eased(t, 1.86, 2.35)));
      R3.lift.open(eased(t, 0.35, 0.55) - eased(t, 0.62, 0.8) + eased(t, 2.6, 2.8));
      R3.lift.show(3 - 3 * (eased(t, 0.62, 1.0) - eased(t, 2.5, 2.72)));
      driveVan(t, dt);
      for (const m of movers) moveAlong(m, t, dt, player);
      if (clockFn) clockFn(u);
    });
    tween(0.25, (u) => R3.lift.open(1 - u));
    R3.hoist.set(HOIST_UP); R3.lift.show(3); parkVan();
  }
  for (const n of INTERNS) dismissIntern(n);
  fit3.wide = false;
  for (const m of movers) {
    m.streak.material.opacity = 0;
    pack(m.s, false);
    placeAt(m.s, m.to);
    m.s.moving = false;
    if (player) puff(SLOTS[m.to][0], SLOTS[m.to][1]);
  }
  if (player && !reduced) view.shake = 0.03;
}
// The pairs' places swap the moment the swap starts, and their chairs catch up. One van: a trip waits for the last.
let tripLine = Promise.resolve();
function swapPairs(pairs, style, clockFn) {
  const run = pairs.flatMap(([a, b], k) => {
    const ia = a.slot, ib = b.slot;
    a.moves++; b.moves++;
    a.slot = ib; b.slot = ia;
    return [[a, ia, ib, k], [b, ib, ia, k]];
  });
  return (tripLine = tripLine.then(() => trip(run, style, clockFn)));
}
// While the player's swap stops the office, everyone else downs tools and watches the chairs go.
function hold3(on, movers) {
  document.querySelector(".watch").classList.toggle("hold", on);
  for (const s of STAFF) if (!movers.includes(s)) { s.A.watch = on ? movers[0].chair.position : null; if (on) s.A.busy = false; }
}

// One question, from the envelope's drop to its stamp. The clock counts st.mov and st.th.
async function answer(e, st) {
  const s = STAFF[e], [x, z] = SLOTS[s.slot], stand = s.moving && internAt(s.slot), who = stand ? stand.A : s.A;
  E3.stamp.material.color.setHex(s.color); E3.stamp.material.emissive.setHex(s.color);
  E3band.material.color.setHex(s.color); E3band.material.emissive.setHex(s.color);
  E3.stamp.visible = false; E3.group.visible = true;
  s.tag.classList.add("need"); L3.need = s;
  const m0 = st.mov, t0 = st.th;
  if (inOffice(s)) {
    const from = new THREE.Vector3(x - 0.3, 4.6, z + 1.25), to = new THREE.Vector3(x - 0.3, 1.04, z + 1.25);
    who.busy = true;
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
    who.busy = true;
    await rt(T3.FAX_THINK, (u) => { st.th = t0 + T3.FAX_THINK * u; setClock(st.mov, st.th); });
    who.busy = !!who.typing;
    wireMat.color.setHex(0xe8a33d); page3.visible = true;
    await rt(T3.FAX_OUT, (u) => { place(1 - u); st.mov = m0 + T3.FAX_OUT * (1 + u); setClock(st.mov, st.th); });
    page3.visible = false; wireMat.color.setHex(0x14171c);
    st.mov = m0 + 2 * T3.FAX_OUT; st.th = t0 + T3.FAX_THINK;
    E3.group.position.copy(to); E3.group.visible = true; E3.stamp.visible = true;
    await rt(0.12, () => {});
    st.faxes++; st.faxRun++; st.localRun = 0; st.faxTimes.push(performance.now() / 1000);
    if (e === 4 || e === 5) st.legalFaxes++;
  }
  who.busy = !!who.typing;
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
      await swapPairs([[a, b]], "player", (u) => { st.mov = m0 + T3.SWAP * u; setClock(st.mov, st.th); });
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
          // The boxes close and the advisers set off: the round holds the office for ROUND; interns mind the desks
          // through the next questions until they land.
          for (const s of moving.flat()) tween(0.12, (u) => { for (const f of s.box.flaps) f.rotation.x = -Math.sign(f.position.z) * lerp(1.2, Math.PI, u); });
          zips.push(swapPairs(moving, "engine"));
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
    ["Take the lift to floor 4", () => goFloor(4), true],
    [endings[3].has("beat") ? "Run the night again" : "Try to beat him", f3Again],
    ["Replay floor 3", () => { endings[3].clear(); goFloor(3); }],
    ["Back to floor 2", () => goFloor(2)],
  ]);
}

resetFloor3();
drawBoard();
closeFloor(3, opened);

export default {
  ...config, camera: camera3, lift: R3.lift, start: f3Intro,
  enter() { fit3.snap = true; fit3.key = ""; resetFloor3(); counts.fill(0); drawBoard(); },
  frame(t, dt) {
    for (const s of STAFF) poseAdviser(s.A, t, dt);
    for (const n of INTERNS) if (n.on) poseAdviser(n.A, t, dt);
    scene3(t, dt);
    R3.city(t);
    if (!reduced) R3.traffic(dt);
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
