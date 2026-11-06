// Floor 4, the briefing room, around x = X4, seen from the chair at the microphone. The CPU tells the card what to run,
// one job at a time. The player calls the names out by hand, then has them written on a list once, and the office
// works down it alone. The list holds names and their order, never an answer: every job runs again on each word's number.
import { $, box, camera, canvasTex, closeFloor, halo, mat, openFloor, pool, reduced, rep, scene, sign, toScreen, tubes, up, v, view } from "../world.js";
import { ease, lerp, rt, tween, until } from "../motion.js";
import { makeFramer, placeLabelsApart, placeTags, sizeOf, tagSize } from "../framing.js";
import { dressAdviser, makeAdviser, makeCabinet, makeChair, makeClock, makeCoatStand, makeCooler, makeEnvelope, makeFax, makePlant, oak, paperM, PEOPLE, poseAdviser, roomDesk, steel3 } from "../props.js";
import { brass, carpetTex, ceilingTex, city, curtainWall, cutawayWall, floorDrop, makeLift, plaster, skirtM, trimM, wallWithDoor, windowView } from "../tower.js";
import { skip, speak, speaking } from "../narration.js";
import { addHistory, choose, clearStage, endings, findEnding, goFloor, labels, revealButtons, setClock } from "../hud.js";
import { faxRing, flick, paCall, stampSound } from "../sound.js";

const config = {
  number: 4, name: "The Briefing", minutes: 4, scale: 9, keys: ["Waiting for you", "Working"],
  labels: ["aMic4", "aTeam4", "aBoard4", "aSlip4", "aFax4"],
  map: [
    ["Each adviser", "one small job on the graphics card; a word takes hundreds"],
    ["Your voice", "the computer's main processor, telling the card each job"],
    ["The list", "every job written down once, when c0ven starts, then run with one instruction"],
    ["Two on a line", "jobs that need the same number, so they can run together"],
    ["Whiteboard", "the numbers that change each word, which the jobs read from memory"],
    ["Fax", "an expert not on the card: the processor works it out, and the list waits for the answer"],
  ],
  lesson: "The work stays. Only the telling goes. Every job still runs on each word's new numbers, so the card gets only a little quicker. What the processor gets back is its hands, for the work only it can do.",
  next: "Floor 5 · coming next.",
};

// Two rooms for two screen shapes, one with four desks in each of two rows and one with two in each of four. A slot's
// number is its place on the route, which snakes down one side and back up the other. A desk sits 1.2 in front of its chair.
// eye is the chair's view; reach and least set how wide it must see. via is the walk from the lift, round the desk to the
// chair. The rest place the things on the desk, relative to its centre.
const X4 = 180, opened = openFloor();
const LAY = {
  wide: {
    back: -4.4, front: 6.6, half: 5.9, pitch: 0.54, mic: [0, 4.75], eye: [0, 1.66, 5.85], reach: 0.85, least: 0.52, boardX: 0,
    via: [[0, 1.66, 2.7], [1.6, 1.66, 4.3], [1.3, 1.66, 5.55]],
    clerk: [-1.05, -2.0], pad: [-0.56, 0.02], tray: [0.5, -0.06], stand: [0.04, -0.26], fax: [0.92, -0.26], mug: [-0.98, -0.2],
    slots: [[-4.05, -2.6], [-1.35, -2.6], [1.35, -2.6], [4.05, -2.6], [4.05, 0.35], [1.35, 0.35], [-1.35, 0.35], [-4.05, 0.35]],
    solid: [[-5.9, -5.5], [-2.6, 4.1], [5.5, 5.9]], windows: [[-5.5, -2.6], [4.1, 5.5]], lights: [[-2.7, -0.7], [2.7, -0.7], [0, 2.9]],
  },
  tall: {
    back: -5.6, front: 7.0, half: 3.4, pitch: 0.82, mic: [0, 5.5], eye: [0, 1.72, 6.5], reach: 0.42, least: 0.82, boardX: -0.4,
    via: [[0, 1.7, 3.6], [1.6, 1.7, 5.0], [1.35, 1.7, 6.2]],
    clerk: [-1.55, -1.0], pad: [-0.3, -0.04], tray: [0.36, -0.04], stand: [0, -0.34], fax: [0.5, -0.34], mug: [-0.62, -0.3],
    slots: [[-1.2, -3.9], [-1.2, -1.9], [-1.2, 0.1], [-1.2, 2.1], [1.2, 2.1], [1.2, 0.1], [1.2, -1.9], [1.2, -3.9]],
    solid: [[-3.4, -3.1], [-1.25, 3.4]], windows: [[-3.1, -1.25]], lights: [[0, -2.2], [0, 2.4]],
  },
};
const WALL = 4.4, BOARD = { x: 2.35, y: 2.5, w: 2.7, h: 1.76 };
const room = new THREE.Group(); room.position.x = X4; scene.add(room);
const shells = {}, upperWalls = {};
const L4 = { mode: "", mic: new THREE.Vector3(), tray: new THREE.Vector3(), pad: new THREE.Vector3(), board: new THREE.Vector3(), face: new THREE.Vector3(), fax: new THREE.Vector3() };

// The whiteboard: redrawn each word, lit from within so it reads at a distance.
const boardTex = canvasTex(1024, 668, () => {});
const boardMat = new THREE.MeshStandardMaterial({ map: boardTex, emissiveMap: boardTex, emissive: 0xffffff, emissiveIntensity: 0.3, roughness: 0.5 });
const hornGlow = [];
const airSign = { m: null };

// A horn speaker on a bracket, facing into the room.
function horn(parent, x, y, z, ry = 0) {
  const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = ry; parent.add(g);
  const grey = mat(0x7d8088, { metalness: 0.4, roughness: 0.5, side: THREE.DoubleSide });
  const bell = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.36, 16, 1, true), grey); bell.rotation.x = -Math.PI / 2; bell.position.z = 0.24; g.add(bell);
  box(0.09, 0.09, 0.14, mat(0x3a3f48), 0, 0, 0.04, g);
  hornGlow.push(halo(0xffb45c, 0.9, 0, 0, 0.5, g, 0));
}
for (const [mode, L] of Object.entries(LAY)) {
  const g = new THREE.Group(); room.add(g); shells[mode] = g;
  const w = L.half * 2, d = L.front - L.back, cz = (L.back + L.front) / 2, drop = floorDrop(4);
  box(w, 0.1, d, mat(0xffffff, { map: rep(carpetTex, w, d), roughness: 1 }), 0, 0.05, cz, g);
  box(w + 0.08, 0.2, d + 0.08, mat(0x4a4f58), 0, -0.03, cz, g);
  box(w, drop - 0.1, d, mat(0x12151b), 0, -(drop + 0.1) / 2, cz, g);
  curtainWall(g, 0, L.front + 0.01, w, -drop, -0.1, 2.5);
  for (const sx of [-1, 1]) {
    cutawayWall(g, sx * L.half, plaster, L.back, L.front, WALL, 1.3);
    box(0.03, 0.14, d, skirtM, sx * (L.half - 0.08), 0.17, cz, g);
  }
  // The back wall: solid runs, one holding the lift, and windows onto the city with the upper floors above.
  const z = L.back;
  for (const [a, b] of L.solid) {
    if (a < 0 && b > 0) wallWithDoor(g, plaster, a, b, WALL, z + 0.06, 0, 0.12, 0.1);
    else box(b - a, WALL, 0.12, plaster, (a + b) / 2, WALL / 2, z, g);
  }
  const Y0 = 1.1, Y1 = 3.3;
  for (const [a, b] of L.windows) {
    box(b - a, Y0, 0.12, plaster, (a + b) / 2, Y0 / 2, z, g);
    box(b - a, WALL - Y1, 0.12, plaster, (a + b) / 2, (WALL + Y1) / 2, z, g);
    const glass = new THREE.Mesh(new THREE.PlaneGeometry(b - a, Y1 - Y0), new THREE.MeshBasicMaterial({ color: 0x9fb4c8, transparent: true, opacity: 0.05, blending: THREE.AdditiveBlending, depthWrite: false }));
    glass.position.set((a + b) / 2, (Y0 + Y1) / 2, z); g.add(glass);
    box(b - a + 0.1, 0.05, 0.26, mat(0x56606f, { roughness: 0.6 }), (a + b) / 2, Y0, z + 0.04, g);
    const bays = Math.max(1, Math.round((b - a) / 1.3));
    for (let k = 1; k < bays; k++) box(0.05, Y1 - Y0, 0.08, trimM, a + (b - a) * k / bays, (Y0 + Y1) / 2, z + 0.02, g);
    box(0.1, Y1 - Y0, 0.14, trimM, a, (Y0 + Y1) / 2, z + 0.02, g); box(0.1, Y1 - Y0, 0.14, trimM, b, (Y0 + Y1) / 2, z + 0.02, g);
  }
  for (const [a, b] of [[-L.half, -0.77], [0.77, L.half]]) box(b - a, 0.14, 0.03, skirtM, (a + b) / 2, 0.17, z + 0.075, g);
  curtainWall(g, 0, z - 0.05, w, WALL, 40, 2.3, 0x83878e);
  sign("THE BRIEFING ROOM", 2.9, 0.4, 0, 3.94, z + 0.07, undefined, undefined, g);
  for (const sx of [-1, 1]) horn(g, sx * 1.75, 3.9, z + 0.07);
  // Hanging tubes, each with its pool on the carpet.
  for (const [lx, lz] of L.lights) {
    const t = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.05, 0.16), new THREE.MeshBasicMaterial({ color: 0xe6efe9 }));
    t.position.set(lx, 3.75, lz); g.add(t); tubes.push(t);
    halo(0xdfe9e4, 1.6, lx, 3.7, lz + 0.05, g, 0.16);
    for (const wx of [-1.0, 1.0]) box(0.01, 0.85, 0.01, steel3, lx + wx, 4.2, lz, g);
    pool(g, 0xc8daf0, 2.6, lx, 0.104, lz + 0.3, 0.09);
  }
  // From the chair the room is whole. These faces show only from inside, so the view from above looks in through them.
  const ceil = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat(0xffffff, { map: rep(ceilingTex, w / 1.2, d / 1.2), emissive: 0xffffff, emissiveMap: rep(ceilingTex, w / 1.2, d / 1.2), emissiveIntensity: 0.3, roughness: 0.95 }));
  ceil.rotation.x = Math.PI / 2; ceil.position.set(0, WALL - 0.01, cz); g.add(ceil);
  const behind = new THREE.Mesh(new THREE.PlaneGeometry(w, WALL), plaster);
  behind.rotation.y = Math.PI; behind.position.set(0, WALL / 2, L.front - 0.03); g.add(behind);
  for (let x = -L.half + 1.6; x < L.half - 1; x += 3) {
    const view1 = new THREE.Mesh(new THREE.PlaneGeometry(2.3, 2.1), new THREE.MeshBasicMaterial({ map: windowView(23 + Math.round(x), 4) }));
    view1.rotation.y = Math.PI; view1.position.set(x, 2.2, L.front - 0.05); g.add(view1);
  }
  const upper = upperWalls[mode] = new THREE.Group(); g.add(upper);
  for (const sx of [-1, 1]) {
    const side = new THREE.Mesh(new THREE.PlaneGeometry(d, WALL), plaster);
    side.rotation.y = -sx * Math.PI / 2; side.position.set(sx * (L.half - 0.065), WALL / 2, cz); upper.add(side);
  }
  const cool = new THREE.PointLight(0xdfe9e4, 0.75, 14, 1.5); cool.position.set(0, 3.6, cz - 0.6); g.add(cool);
  const warm = new THREE.PointLight(0xffc27a, 0.5, 7, 1.6); warm.position.set(L.mic[0], 2.2, L.mic[1]); g.add(warm);
}
// Each shape's own furniture along its walls.
{
  const W = shells.wide, T = shells.tall;
  makeCooler(W, 5.35, -3.6); makePlant(W, -5.35, -3.7, 1.15); makeCabinet(W, -5.45, 1.7, 3); makeCabinet(W, -5.45, 2.3, 4); makeCoatStand(W, 5.4, 4.9); makePlant(W, -5.3, 5.1, 1.0);
  makeCabinet(W, 5.45, 1.7, 3);
  makeCooler(T, -3.0, -5.0); makePlant(T, 3.0, 5.9, 1.0);
  // The house rules, on the wall by the window.
  const note = canvasTex(360, 280, (g, w, h) => {
    g.fillStyle = "#e9e4d2"; g.fillRect(0, 0, w, h);
    g.fillStyle = "#c2453d"; g.fillRect(0, 0, w, 40);
    g.fillStyle = "#f4efe0"; g.textAlign = "center"; g.font = "700 28px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("STAFF NOTICE", w / 2, 29);
    g.fillStyle = "#23272f"; g.font = "700 36px 'Barlow Condensed', Arial Narrow, sans-serif";
    ["PLEASE WAIT", "TO BE TOLD.", "", "THEN WAIT", "A LITTLE MORE."].forEach((s, i) => g.fillText(s, w / 2, 86 + i * 40));
  });
  const n = new THREE.Mesh(new THREE.PlaneGeometry(1.15, 0.9), new THREE.MeshStandardMaterial({ map: note, roughness: 0.7 }));
  n.position.set(-1.8, 2.15, LAY.wide.back + 0.075); W.add(n);
}

// What stands on whichever wall and floor the layout gives it: the whiteboard, the lift, the desks.
const backGear = new THREE.Group(), boardGear = new THREE.Group(); room.add(backGear); backGear.add(boardGear);
{
  const face = new THREE.Mesh(new THREE.PlaneGeometry(BOARD.w, BOARD.h), boardMat); face.position.set(BOARD.x, BOARD.y, 0.03); boardGear.add(face);
  for (const [w, h, x, y] of [[BOARD.w + 0.1, 0.05, 0, BOARD.h / 2 + 0.025], [BOARD.w + 0.1, 0.05, 0, -BOARD.h / 2 - 0.025], [0.05, BOARD.h, -BOARD.w / 2 - 0.025, 0], [0.05, BOARD.h, BOARD.w / 2 + 0.025, 0]]) box(w, h, 0.05, steel3, BOARD.x + x, BOARD.y + y, 0.03, boardGear);
  box(BOARD.w * 0.7, 0.03, 0.09, steel3, BOARD.x, BOARD.y - BOARD.h / 2 - 0.06, 0.07, boardGear);
  for (const [dx, c] of [[-0.3, 0x2f5fd0], [-0.12, 0xc2453d]]) { const pen = box(0.12, 0.02, 0.02, mat(c), BOARD.x + dx, BOARD.y - BOARD.h / 2 - 0.035, 0.07, boardGear); pen.rotation.y = 0.1; }
  makeClock(boardGear, 3.5, 3.86, 0.03, 0.2);
  airSign.m = sign("ON AIR", 0.74, 0.25, -0.02, 3.36, 0.03, "#2a1212", "#ff5a44", backGear).material;
  airSign.m.emissive = new THREE.Color(0xff3a2a); airSign.m.emissiveMap = airSign.m.map; airSign.m.emissiveIntensity = 0.05;
}
const lift = makeLift(backGear, { x: 0, y: 0.1, z: 0.005, number: 4, name: "The Briefing", signSide: 0 });

const STAFF = PEOPLE.map((p) => ({ ...p }));
const JOBS = ["copies the board", "doubles it", "adds 9", "adds those two", "adds the fax", "halves it", "takes 7", "stamps it"];
const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const desks = STAFF.map(() => { const g = new THREE.Group(); room.add(g); roomDesk(g, 0, 1.2, false); return g; });
// The last desk's tube, which takes a stamped word up to the next floor.
// It stands at the desk's outer corner, away from the aisle the chair looks down.
const TUBE = new THREE.Vector3(0, 0, 1.5), tubeGear = new THREE.Group(); desks[7].add(tubeGear);
{
  const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 3.3, 12), brass); pipe.position.set(0, 2.75, TUBE.z); pipe.castShadow = true; tubeGear.add(pipe);
  box(0.16, 0.2, 0.16, brass, 0, 1.12, TUBE.z, tubeGear);
}
const ringM = (c) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
for (const [i, s] of STAFF.entries()) {
  s.chair = makeChair(0, 0);
  s.chair.traverse((o) => { if (o.material && o.material.color && o.material.color.getHex() === 0x2f3644) o.material.color.setHex(s.color).lerp(new THREE.Color(0x1c2027), 0.45); });
  s.A = makeAdviser({ suit: s.suit, hair: s.hair, tie: s.color, skin: s.skin, nodPhase: i * 0.9, old: !!s.old });
  s.A.group.rotation.order = "YXZ";
  dressAdviser(s.A, s.hat, s.color);
  s.A.sit = 1;
  s.ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.6, 32), ringM(s.color));
  s.ring.rotation.x = -Math.PI / 2; scene.add(s.ring);
}
// A column of light on whoever is working.
const beamTex = canvasTex(16, 128, (g, w, h) => {
  const l = g.createLinearGradient(0, 0, 0, h); l.addColorStop(0, "rgba(255,255,255,0)"); l.addColorStop(1, "rgba(255,255,255,0.9)");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
});
const beams = [0, 1].map(() => {
  const b = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.62, 3.4, 24, 1, true), new THREE.MeshBasicMaterial({ map: beamTex, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
  scene.add(b); return b;
});

// The microphone's desk: yours, tonight. On it the pad, the mic, the tray, the fax and a mug; the clerk sits off its near corner.
const micDesk = new THREE.Group(); room.add(micDesk);
const R4 = {};
{
  const D = micDesk;
  box(2.4, 0.07, 0.95, oak, 0, 0.985, 0, D);
  box(2.3, 0.8, 0.05, oak, 0, 0.53, -0.42, D);
  for (const sx of [-1.12, 1.12]) box(0.07, 0.95, 0.9, oak, sx, 0.475, 0, D);
  box(1.5, 0.006, 0.6, mat(0x2d4a39, { roughness: 0.7 }), 0, 1.022, 0.02, D);
  const stand = R4.stand = new THREE.Group(); D.add(stand);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.085, 0.03, 18), mat(0x23272f, { metalness: 0.4, roughness: 0.4 })); base.position.set(0, 1.035, 0); stand.add(base);
  const stem = new THREE.CatmullRomCurve3([new THREE.Vector3(0, 1.04, 0), new THREE.Vector3(0, 1.2, 0), new THREE.Vector3(0, 1.27, 0.08), new THREE.Vector3(0, 1.28, 0.16)]);
  const neck = new THREE.Mesh(new THREE.TubeGeometry(stem, 16, 0.008, 8), mat(0x8a9099, { metalness: 0.7, roughness: 0.35 })); neck.castShadow = true; stand.add(neck);
  const headM = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.016, 0.12, 12), mat(0x23272f, { metalness: 0.4, roughness: 0.6 })); headM.rotation.x = Math.PI / 2 - 0.15; headM.position.set(0, 1.285, 0.21); stand.add(headM);
  R4.micLamp = new THREE.Mesh(new THREE.SphereGeometry(0.018, 10, 8), new THREE.MeshBasicMaterial({ color: 0x40100c })); R4.micLamp.position.set(0.05, 1.058, 0.03); stand.add(R4.micLamp);
  // The tray the words arrive in.
  const tray = R4.tray = new THREE.Group(); D.add(tray);
  box(0.52, 0.02, 0.38, mat(0x3a3f48), 0, 1.035, 0, tray);
  for (const [w, d, x, z] of [[0.52, 0.015, 0, -0.18], [0.52, 0.015, 0, 0.18], [0.015, 0.38, -0.25, 0], [0.015, 0.38, 0.25, 0]]) box(w, 0.05, d, mat(0x565c66), x, 1.06, z, tray);
  R4.trayRing = new THREE.Mesh(new THREE.RingGeometry(0.32, 0.4, 32), ringM(0xe8a33d)); R4.trayRing.rotation.x = -Math.PI / 2; R4.trayRing.position.y = 1.03; tray.add(R4.trayRing);
  // The pad, in yellow, and its glow once the narrator has offered it.
  const pad = R4.padG = new THREE.Group(); D.add(pad);
  R4.pad = box(0.3, 0.05, 0.22, mat(0xe9d98a, { roughness: 0.75 }), 0, 1.05, 0, pad); R4.pad.rotation.y = -0.15;
  box(0.13, 0.012, 0.02, mat(0x2f5fd0), 0.18, 1.08, 0.06, pad).rotation.y = 0.6;
  R4.padRing = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.3, 28), ringM(0xe9d98a)); R4.padRing.rotation.x = -Math.PI / 2; R4.padRing.position.y = 1.03; pad.add(R4.padRing);
  R4.mug = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.042, 0.11, 12), mat(0xd9d2c0, { roughness: 0.5 })); R4.mug.position.y = 1.08; R4.mug.castShadow = true; D.add(R4.mug);
  const plateTex = canvasTex(256, 64, (g, w, h) => { g.fillStyle = "#22262d"; g.fillRect(0, 0, w, h); g.fillStyle = "#d2ab62"; g.textAlign = "center"; g.textBaseline = "middle"; g.font = "600 36px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("PUBLIC ADDRESS", w / 2, h / 2 + 2); });
  const plate = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 0.07), new THREE.MeshStandardMaterial({ map: plateTex, roughness: 0.5 })); plate.position.set(0, 1.06, 0.16); plate.rotation.x = -0.5; stand.add(plate);
  pool(D, 0xffb060, 1.0, 0, 1.027, 0.02, 0.26);
  // The fax, its lamp, the sheet it holds out, and its ring on the desk.
  const F = R4.faxG = new THREE.Group(); F.position.y = 1.02; F.scale.setScalar(0.8); D.add(F);
  F.add(makeFax(0, 0, 0, -0.35));
  R4.faxLamp = halo(0xffb45c, 1.0, 0, 0.34, 0, F, 0);
  R4.faxSheet = box(0.22, 0.006, 0.3, paperM, -0.05, 0.3, -0.12, F); R4.faxSheet.rotation.x = -0.5; R4.faxSheet.visible = false;
  R4.faxRing = new THREE.Mesh(new THREE.RingGeometry(0.34, 0.46, 32), ringM(0xffb45c)); R4.faxRing.rotation.x = -Math.PI / 2; R4.faxRing.position.y = 0.012; F.add(R4.faxRing);
}
// The clerk who writes the list, with a small table of their own to take the pad.
const clerkChair = makeChair(0, 0), clerk = makeAdviser({ suit: 0x6a6456, hair: 0x8a7a62, tie: 0xe9d98a, skin: 0xd9b9a0, nodPhase: 2.2 });
clerk.sit = 1;
const clerkTable = new THREE.Group(); room.add(clerkTable);
box(0.56, 0.03, 0.44, oak, 0, 0.86, 0, clerkTable);
for (const [dx, dz] of [[-0.24, -0.18], [0.24, -0.18], [-0.24, 0.18], [0.24, 0.18]]) box(0.03, 0.85, 0.03, oak, dx, 0.425, dz, clerkTable);

// The envelope, and its copy for two people on one line. The list is stapled on once it exists.
const makeE = () => {
  const E = makeEnvelope(); E.group.visible = false; E.group.scale.setScalar(1.6);
  E.slip = box(0.14, 0.006, 0.2, mat(0xe9d98a, { emissive: 0x6a5c1c, emissiveIntensity: 0.6 }), 0.12, 0.012, -0.02, E.group); E.slip.visible = false;
  return E;
};
const E1 = makeE(), E2 = makeE();
// The fax's answer, on its way to the desk that waits for it.
const answerPage = box(0.3, 0.008, 0.38, new THREE.MeshStandardMaterial({ color: 0xf2f2ec, emissive: 0xe8a33d, emissiveIntensity: 0.35 }), 0, 0, 0); answerPage.visible = false;

R4.city = city(scene, X4, 4, { near: -19 });
{
  const key = new THREE.SpotLight(0xe8f0ea, 1.0, 50, 0.85, 1, 1);
  key.position.set(X4 + 1.5, 15, 10); key.target.position.set(X4, 0, 0);
  key.castShadow = true; key.shadow.mapSize.set(1024, 1024); key.shadow.bias = -0.0005;
  scene.add(key, key.target);
}

// The list's route on the carpet, dashes from desk to desk: one line to two people, and two lines into the next.
const ROUTE = [[0, 1, 0], [0, 2, 1], [1, 3, 1], [2, 3, 0], [3, 4, 0], [4, 5, 0], [5, 6, 0], [6, 7, 0]];
const routeM = new THREE.MeshBasicMaterial({ color: 0xe8a33d, transparent: true, opacity: 0 });
for (const [mode, L] of Object.entries(LAY)) {
  // In the wide room it runs just in front of each desk; in the tall one, down the middle aisle.
  const node = ([x, z]) => (mode === "wide" ? new THREE.Vector2(x, z + 1.85) : new THREE.Vector2(Math.sign(x) * 0.2, z + 1.2));
  const dashes = [];
  for (const [a, b, bulge] of ROUTE) {
    const p = node(L.slots[a]), q = node(L.slots[b]), n = new THREE.Vector2(q.y - p.y, p.x - q.x).normalize().multiplyScalar(bulge * 0.34);
    const mid = p.clone().add(q).multiplyScalar(0.5).add(n), steps = Math.round(p.distanceTo(q) / 0.26);
    for (let k = 1; k < steps; k++) {
      const u = k / steps, f = (u2) => p.clone().multiplyScalar((1 - u2) * (1 - u2)).add(mid.clone().multiplyScalar(2 * u2 * (1 - u2))).add(q.clone().multiplyScalar(u2 * u2));
      const at = f(u), to = f(u + 0.01).sub(at);
      dashes.push([at.x, at.y, Math.atan2(to.y, to.x)]);
    }
  }
  const lots = new THREE.InstancedMesh(new THREE.BoxGeometry(0.15, 0.004, 0.035), routeM, dashes.length), o = new THREE.Object3D();
  dashes.forEach(([x, z, r], k) => { o.position.set(x, 0.108, z); o.rotation.y = -r; o.updateMatrix(); lots.setMatrixAt(k, o.matrix); });
  shells[mode].add(lots);
}
const showRoute = (on) => { if (reduced) routeM.opacity = on ? 0.8 : 0; else tween(0.5, (u) => { routeM.opacity = (on ? u : 1 - u) * 0.8; }); };

const spot = (i) => { const [x, z] = LAY[L4.mode].slots[i]; return new THREE.Vector3(X4 + x - 0.3, 1.06, z + 1.25); };
const tubeMouth = () => desks[7].position.clone().add(TUBE).setY(1.24);

function layout4(mode) {
  L4.mode = mode;
  const L = LAY[mode], [mx, mz] = L.mic, at = (p) => new THREE.Vector3(X4 + mx + p[0], 0, mz + p[1]);
  for (const m of Object.keys(LAY)) shells[m].visible = m === mode;
  backGear.position.z = L.back + 0.07; boardGear.position.x = L.boardX;
  micDesk.position.set(mx, 0, mz);
  R4.stand.position.set(L.stand[0], 0, L.stand[1]);
  R4.tray.position.set(L.tray[0], 0, L.tray[1]);
  R4.padG.position.set(L.pad[0], 0, L.pad[1]);
  R4.mug.position.set(L.mug[0], 1.08, L.mug[1]);
  R4.faxG.position.set(L.fax[0], 1.02, L.fax[1]);
  L4.mic.copy(at(L.stand)).setY(1.4);
  L4.tray.copy(at(L.tray)).setY(1.1);
  L4.pad.copy(at(L.pad)).setY(1.08);
  L4.fax.copy(at(L.fax)).setY(1.2);
  L4.board.set(X4 + L.boardX + BOARD.x + 1.35, BOARD.y + BOARD.h / 2 - 0.2, L.back + 0.12);
  L4.face.set(X4 + L.boardX + BOARD.x, BOARD.y - 0.25, L.back + 0.12);
  clerkChair.position.copy(at(L.clerk));
  clerk.group.position.copy(at(L.clerk)).setZ(mz + L.clerk[1] + 0.05);
  clerkTable.position.set(mx + L.clerk[0], 0, mz + L.clerk[1] + 0.62);
  lift.arrive = { via: L.via.map(([x, y, z]) => new THREE.Vector3(X4 + x, y, z)), walk: 3.6, turn: 0.5 };
  STAFF.forEach((s, i) => {
    const [x, z] = L.slots[i];
    desks[i].position.set(x, 0, z);
    s.chair.position.set(X4 + x, 0, z);
    s.A.group.position.set(X4 + x, 0, z + 0.05);
    s.ring.position.set(X4 + x, 0.11, z + 0.35);
    HOT[i].p.set(X4 + x, 1.05, z + 0.3);
  });
  TUBE.x = Math.sign(L.slots[7][0]) * 1.02; tubeGear.position.x = TUBE.x;
  HOT.pad.p.copy(L4.pad); HOT.fax.p.copy(L4.fax); HOT.tray.p.copy(L4.tray);
  frameHead();
}

// ---------- Camera ----------
// In the chair the camera looks where the head is pointed. For the reveal it rises to a framing of the room from above.
const camera4 = { look: new THREE.Vector3(X4, 1, 0), offset: false, fog: [20, 90], par: 0.1, fov: 52 };
const FOV4 = 36, FIT4 = {}, DEG = 180 / Math.PI;
for (const [m, L] of Object.entries(LAY)) {
  const pts = [];
  for (const sx of [-1, 1]) pts.push(new THREE.Vector3(X4 + sx * L.half, 0, L.back), new THREE.Vector3(X4 + sx * L.half, WALL, L.back), new THREE.Vector3(X4 + sx * L.half, 0, L.front), new THREE.Vector3(X4 + sx * L.half, 1.3, L.front));
  FIT4[m] = { pitch: L.pitch, pts };
}
const over = { look: new THREE.Vector3() }, fit4 = makeFramer(over, FOV4, () => FIT4[L4.mode]);
const H4 = { eye: new THREE.Vector3(), rest: new THREE.Vector3(), look: new THREE.Vector3(), want: new THREE.Vector3(), pos: new THREE.Vector3(), aim: null, aimFov: 0, fov: 52, f: 52, up: 0, px: 0, py: 0 };
// The narrowest view that takes in the room's width, tipped down so the board's top clears what covers the screen's top:
// the stopwatch on a narrow screen or in the tall room, allowing for its rows to come, or else the top bar.
function frameHead() {
  const L = LAY[L4.mode], T = Math.max(L.least, L.reach / (innerWidth / innerHeight));
  H4.fov = 2 * Math.atan(T) * DEG;
  H4.eye.set(X4 + L.eye[0], L.eye[1], L.eye[2]);
  const under = innerWidth <= 720 || L4.mode === "tall", clear = document.querySelector(under ? ".watch" : ".topbar").getBoundingClientRect().bottom + (under ? 34 : 10);
  const top = Math.atan2(BOARD.y + BOARD.h / 2 - L.eye[1], L.eye[2] - L.back), tip = Math.atan((1 - 2 * clear / innerHeight) * T) - top;
  H4.rest.set(H4.eye.x, H4.eye.y - Math.sin(tip) * 8, H4.eye.z - Math.cos(tip) * 8);
}
const aim = (p, fov = 0) => { H4.aim = p; H4.aimFov = fov; };
// A view that holds half-width w and half-height h around p, never wider than the chair's own.
const zoomOn = (p, w, h) => {
  const d = p.distanceTo(H4.eye), T = Math.max(h / d, w / d / (innerWidth / innerHeight));
  return Math.min(H4.fov, 2 * Math.atan(T) * DEG);
};
camera4.place = (dt) => {
  const k = reduced ? 1 : 1 - Math.exp(-dt * (H4.aim ? 2.8 : 5));
  if (H4.aim) H4.want.copy(H4.aim); else H4.want.set(H4.rest.x + H4.px * 2.6, H4.rest.y - H4.py * 1.1, H4.rest.z);
  H4.look.lerp(H4.want, k);
  H4.f += ((H4.aimFov || H4.fov) - H4.f) * k;
  if (H4.up <= 0) {
    camera4.offset = false; camera4.look.copy(H4.look); camera4.fov = H4.f;
    scene.fog.near = camera4.fog[0]; scene.fog.far = camera4.fog[1];
    return H4.eye;
  }
  const u = H4.up, p = fit4.place(dt);
  camera.setViewOffset(innerWidth, innerHeight, fit4.off[0] * u, fit4.off[1] * u, innerWidth, innerHeight);
  camera4.offset = true; camera4.look.lerpVectors(H4.look, over.look, u); camera4.fov = lerp(H4.f, FOV4, u);
  upperWalls[L4.mode].visible = u < 0.55;
  return H4.pos.lerpVectors(H4.eye, p, u);
};
const rise = () => { fit4.snap = true; fit4.key = ""; if (reduced) H4.up = 1; else return tween(1.6, (u) => { H4.up = ease(u); }); };
const toDesk = async () => {
  if (!H4.up) return;
  if (!reduced) await tween(1.2, (u) => { H4.up = 1 - ease(u); });
  H4.up = 0; upperWalls[L4.mode].visible = true;
};

const anchors4 = [
  [$("aMic4"), () => L4.mic.clone().setY(1.75)],
  [$("aTeam4"), () => up(STAFF[3].A.group, 2.15)],
  [$("aBoard4"), () => L4.board],
  [$("aSlip4"), () => E1.group.position.clone().setY(1.25)],
  [$("aFax4"), () => L4.fax.clone().setY(1.5)],
];

// ---------- The shift ----------
// In proportion to docs/measurements/MEASUREMENTS.md#2026-10-11-launch-cost: reading out a name takes longer than the job
// it starts, so by hand the office waits on the CPU, and with the list only the office's own pace counts. Two on a line
// save what #2026-10-16-decode-launches-together saves, and DROP is the CPU's whole share of a word on the list.
const T4 = { CALL: 0.32, JOB: 0.48, PAIR: 0.78, DROP: 0.3, ANSWER: 0.9 };
const ROUNDS = 2;       // words read out by hand before the narrator steps in
const BY_HAND = 5;      // words by hand before the voice gives out
const SLIP_WORDS = 2;   // words on the list that end the shift
const SLIP = [[0], [1, 2], [3], [4], [5], [6], [7]];   // lines of STAFF indices
const NEEDS = [[], [0], [0], [1, 2], [3], [4], [5], [6]];   // whose number each job reads
const NUMBERS = [17, 42, 8, 63, 25, 91, 36, 54, 12, 77, 29, 48, 5, 84, 33, 70];
const SENTENCE = "PLEASE REMAIN AT YOUR DESK UNTIL THE BUILDING SAYS OTHERWISE THANK YOU".split(" ");
// S is the shift, W the word in hand. The stopwatch runs on W.lane: "f" while the office waits, "t" while it works.
const S = { on: false, live: false, stopped: false, stepped: false, refused: false, quiet: false, trayLit: false, mode: "hand", want: false, pad: true, calls: 0, words: 0, faxes: 0, slipWords: 0, total: 0, best: {}, free: true, wake: null, buffered: false, next: null, said: "", note: "", stepAt: 0, faxWaiting: false, hover: null };
let W = { run: false, lane: "f", f: 0, t: 0, n: 0, fax: "none" };
let react = null, tagsLive = false;

// The board keeps the numbers already used, so the only thing that changes from word to word can be seen changing.
const font = (wt, px) => `${wt} ${px}px 'Barlow Condensed', 'Arial Narrow', sans-serif`;
function wipeBoard() {
  const c = boardTex.image, g = c.getContext("2d"), w = c.width, h = c.height;
  g.fillStyle = "#e6e9e3"; g.fillRect(0, 0, w, h);
  g.fillStyle = "rgba(120,130,140,0.07)";
  for (let k = 0; k < 7; k++) { g.beginPath(); g.ellipse((k * 233) % w, 90 + (k * 151) % (h - 120), 190, 34, (k % 3) * 0.2, 0, Math.PI * 2); g.fill(); }
  g.textBaseline = "alphabetic"; g.textAlign = "left";
  return g;
}
function drawBoard() {
  const g = wipeBoard(), w = boardTex.image.width;
  g.fillStyle = "#2f5fd0"; g.font = font(700, 64); g.fillText(`WORD ${Math.max(1, S.total)}`, 34, 78);
  g.fillStyle = "#23272f"; g.textAlign = "center"; g.font = font(700, 320); g.fillText(String(W.n || "--"), w / 2, 370);
  g.textAlign = "left"; g.font = font(600, 64);
  let x = 34;
  for (const n of NUMBERS.slice(0, Math.max(0, S.total - 1)).slice(-6)) {
    const t = String(n), tw = g.measureText(t).width;
    g.fillStyle = "rgba(35,39,47,0.55)"; g.fillText(t, x, 470);
    g.fillStyle = "#c2453d"; g.fillRect(x - 6, 446, tw + 12, 6);
    x += tw + 44;
  }
  const said = SENTENCE.slice(0, S.words).join(" ");
  g.fillStyle = "rgba(35,39,47,0.9)"; g.font = font(600, 44);
  const lines = [""];
  for (const word of said.split(" ")) { const t = lines[lines.length - 1] + (lines[lines.length - 1] ? " " : "") + word; if (g.measureText(t).width > w - 68) lines.push(word); else lines[lines.length - 1] = t; }
  lines.slice(-2).forEach((l, i) => g.fillText(l, 34, 562 + i * 48));
  boardTex.needsUpdate = true;
}
// The board after a shift on the list: where a word's time went. Each bar grows as the narrator names it. Orange and blue
// match the stopwatch.
const TALLY = { hand: 0, slip: 0 };
function drawTally() {
  // From the tall room's chair, the lift door and a column stand in front of the board's edges below the legend.
  const [left, right] = L4.mode === "tall" ? [104, 880] : [34, boardTex.image.width - 34], scale = (right - left) / 6.8;
  const g = wipeBoard(), ink = "#23272f", orange = "#e8a33d", blue = "#6fa3c4";
  g.fillStyle = "#2f5fd0"; g.font = font(700, 64); g.fillText("WHERE THE TIME WENT", 34, 78);
  g.font = font(600, 44);
  g.fillStyle = orange; g.fillRect(34, 104, 34, 34); g.fillStyle = ink; g.fillText("WAITING FOR YOU", 80, 136);
  g.fillStyle = blue; g.fillRect(400, 104, 34, 34); g.fillStyle = ink; g.fillText("WORKING", 446, 136);
  for (const [name, mode, y] of [["READ OUT", "hand", 232], ["THE LIST", "slip", 410]]) {
    const run = S.best[mode], u = TALLY[mode];
    if (!run || !u) continue;
    g.fillStyle = ink; g.font = font(700, 60); g.fillText(`${name}  ${(run.f + run.t).toFixed(1)} s`, left, y);
    g.fillStyle = orange; g.fillRect(left, y + 14, run.f * scale * u, 92);
    g.fillStyle = blue; g.fillRect(left + run.f * scale * u, y + 14, run.t * scale * u, 92);
  }
  boardTex.needsUpdate = true;
}
const growBar = (mode) => tween(0.8, (u) => { TALLY[mode] = u; drawTally(); });

// ---------- What can be pressed ----------
// Each person and thing has a tag, so a tap need not find it in the room and a keyboard can reach it. The tags of things
// stay out of sight until the thing is called for.
const tagsBox = $("tags4"), hintEl = $("hint4"), linesEl = $("lines4");
STAFF.forEach((s, i) => {
  const b = document.createElement("button");
  b.type = "button"; b.className = "tag";
  b.style.setProperty("--c", hex(s.color));
  b.innerHTML = `<i></i>${s.name}<b class="v"></b>`;
  b.setAttribute("aria-label", `${s.name}, ${JOBS[i]}`);
  b.addEventListener("click", () => callOut(i));
  s.tag = b; s.val = b.querySelector(".v"); tagsBox.appendChild(b);
});
const thingTag = (text, what) => {
  const b = document.createElement("button");
  b.type = "button"; b.className = "tag q";
  b.style.setProperty("--c", "#e8a33d");
  b.textContent = text;
  b.addEventListener("click", () => press(what));
  tagsBox.appendChild(b);
  return b;
};
const HOT = STAFF.map((s, i) => ({ p: new THREE.Vector3(), r: 0.72, tap: () => callOut(i) }));
HOT.pad = { p: new THREE.Vector3(), r: 0.26, tap: () => press("pad"), tag: thingTag("The pad", "pad") };
HOT.fax = { p: new THREE.Vector3(), r: 0.32, tap: () => press("fax"), tag: thingTag("Answer the fax", "fax") };
HOT.tray = { p: new THREE.Vector3(), r: 0.34, tap: () => press("main"), tag: thingTag("Send the word", "main") };
const THINGS = [HOT.pad, HOT.fax, HOT.tray];
const v2 = new THREE.Vector3();
// The person or thing nearest the point on the screen, measured in its own size so a small one close by still wins.
function pick(cx, cy) {
  let best = null, score = 1;
  for (const h of [...HOT, ...THINGS]) {
    const [x, y] = toScreen(h.p), [, y2] = toScreen(v2.copy(h.p).setY(h.p.y + h.r));
    const d = Math.hypot(cx - x, cy - y) / Math.max(22, Math.abs(y - y2));
    if (d < score) { score = d; best = h; }
  }
  return best;
}
// The narrator offers the pad while still speaking, so it can be taken from then on.
const inPlay = () => view.floor === 4 && S.on && (!S.stopped || R4.padLit);
addEventListener("pointermove", (e) => {
  if (view.floor !== 4 || view.rig) return;
  if (e.pointerType !== "touch") { H4.px = e.clientX / innerWidth - 0.5; H4.py = e.clientY / innerHeight - 0.5; }
  const free = inPlay() && !(e.target.closest && e.target.closest("button, .panel, .caption"));
  S.hover = free ? pick(e.clientX, e.clientY) : null;
  document.body.style.cursor = S.hover ? "pointer" : "";
});
addEventListener("click", (e) => {
  if (!inPlay() || (e.target.closest && e.target.closest("button, .panel, .caption"))) return;
  const h = pick(e.clientX, e.clientY);
  if (h) h.tap();
});
addEventListener("keydown", (e) => {
  if (!inPlay() || e.repeat) return;
  const k = e.key.toLowerCase(), onButton = document.activeElement && document.activeElement.tagName === "BUTTON";
  if (k === "enter" && !onButton) press("main");
  if (k === "f") press("fax");
  if (k === "p") press("pad");
});
function setLive(on) {
  tagsLive = on;
  tagsBox.classList.toggle("live", on);
}
function hint(text, said = false) {
  hintEl.hidden = !text;
  hintEl.classList.toggle("said", said);
  if (text) hintEl.textContent = text;
}
function paintHint() {
  if (!S.on || S.stopped) return hint("");
  if (S.note) return hint(S.note);
  if (W.held) return hint("Margaret can't finish until you answer the fax");
  if (W.fax === "ringing") return hint("The fax is ringing");
  if (S.mode === "slip") return hint(W.sent ? "" : "Send the word: it's in your tray");
  if (S.quiet) return hint("Pick up the pad");
  if (S.mode === "hand" && S.want && W.called) return hint("Finish this word. The clerk takes the next one down");
  if (S.mode === "hand" && !S.live) return hint("Click the lit name to call it out");
  hint(S.said, true);
}
function setNext(i) {
  S.next = i;
  S.quiet = i === 0 && S.stepped && !S.refused && !S.want;
  STAFF.forEach((s, k) => s.tag.classList.toggle("need", k === i && !S.quiet));
  tagSize.clear();
}
const CALLS = ["copy the board", "double it", "add 9", "add those two", "add the fax", "halve it", "take 7", "stamp it"];
function sayName(i) {
  const line = `${STAFF[i].name}, ${CALLS[i]}.`;
  // The voice goes as the shift wears on.
  S.said = S.calls > 32 ? `…${line.toLowerCase().slice(0, 5)}…` : S.calls > 20 ? line.toLowerCase() : line;
  paintHint();
}
// A name called out of turn: that person says why not, and looks at whoever is next.
let noteTimer = 0;
function callOut(i) {
  if (!inPlay() || S.mode !== "hand" || S.next === null) return;
  if (S.next === i) return press("main");
  const s = STAFF[i], need = NEEDS[i].filter((k) => k >= S.next).map((k) => STAFF[k].name).join(" and ");
  S.note = i < S.next ? `${s.name} has done this word` : need ? `${s.name} needs ${need}'s number first` : `One at a time. ${STAFF[S.next].name} first`;
  s.A.watch = STAFF[S.next].A.group.position;
  flick();
  clearTimeout(noteTimer);
  noteTimer = setTimeout(() => { S.note = ""; if (!s.working) s.A.watch = camera.position; paintHint(); }, 1800);
  paintHint();
}

// What a press does depends on what the hands are at: one made while they are busy is kept for when they are free.
function press(what) {
  if (what === "pad" && S.stopped && R4.padLit) { S.want = true; R4.padLit = false; slidePad(); skip(); return; }
  if (!S.on || S.stopped) return;
  if (what === "pad") {
    if (S.mode !== "hand" || S.want || !S.pad) return;
    S.want = true; R4.padLit = false; S.quiet = false;
    slidePad();
    if (!S.stepped && react) react.say("f4found", "now");
    paintHint();
    // The pad counts from the first name of a word, so a press before that one records this word.
    if (!W.called && S.wake) { const r = S.wake; S.wake = null; r("pad"); }
    return;
  }
  if (what === "fax") {
    if (W.fax !== "ringing") return;
    if (S.free) answerFax(); else S.faxWaiting = true;
    return;
  }
  if (W.held || (S.mode === "slip" && W.sent) || S.mode === "record") return;
  S.live = true;
  if (S.free && S.wake) { const r = S.wake; S.wake = null; r(true); } else S.buffered = true;
}
// Waits for the player's press. Resolves to true, or "pad" if the pad was picked up instead.
function waitPress() {
  if (S.buffered && S.free) { S.buffered = false; return Promise.resolve(true); }
  S.buffered = false;
  return new Promise((r) => { S.wake = r; });
}
// The pad crosses the desk to the clerk.
function slidePad() {
  const L = LAY[L4.mode], from = R4.padG.position.clone(), to = new THREE.Vector3(L.clerk[0], -0.15, L.clerk[1] + 0.62);
  flick();
  tween(0.5, (u) => { const k = ease(u); R4.padG.position.lerpVectors(from, to, k); R4.padG.position.y += Math.sin(k * Math.PI) * 0.25; });
}

function ringFax() {
  W.fax = "ringing";
  W.answered = new Promise((r) => { W.settle = r; });
  faxRing();
  R4.faxSheet.visible = true;
  paintHint();
}
async function answerFax() {
  const word = W;
  word.fax = "answering";
  S.free = false;
  aim(L4.fax);
  await rt(T4.ANSWER, () => {});
  aim(null);
  S.free = true;
  if (S.buffered && S.wake) { const r = S.wake; S.wake = null; S.buffered = false; r(true); }
  S.faxes++;
  R4.faxSheet.visible = false;
  word.fax = "answered"; word.value = 3 + word.n % 5;
  const wasHeld = word.held;
  word.settle();
  if (react && S.mode === "slip" && !wasHeld) react.say("f4s4", true);
  paintHint();
  // The answer flies to the desk that waits for it, off the clock either way.
  answerPage.visible = true;
  const from = L4.fax.clone(), to = spot(4).add(new THREE.Vector3(0.5, 0.02, -0.1));
  await rt(0.28, (u) => { const k = reduced ? 1 : ease(u); answerPage.position.lerpVectors(from, to, k); answerPage.position.y += Math.sin(k * Math.PI) * 1.2; answerPage.rotation.z = (1 - k) * 2; });
  answerPage.visible = false;
  flick();
}
// Holds at the line that waits for the fax. The office is idle, so this counts as waiting.
async function holdForFax() {
  if (W.fax === "none" || W.fax === "answered") return;
  W.lane = "f"; W.held = true;
  // Nothing else can move, so nothing glows and an early tap is not kept for later.
  S.next = null; S.buffered = false;
  STAFF.forEach((s) => s.tag.classList.remove("need"));
  STAFF[4].tag.classList.add("wait");
  STAFF[4].A.watch = L4.fax;
  paintHint();
  await W.answered;
  W.held = false;
  STAFF[4].tag.classList.remove("wait");
  STAFF[4].A.watch = camera.position;
  paintHint();
}

function hop(E, to, seconds) {
  const from = E.group.position.clone();
  E.group.visible = true; E.group.scale.setScalar(1.6); flick();
  return rt(seconds, (u) => { const k = reduced ? 1 : ease(u); E.group.position.lerpVectors(from, to, k); E.group.position.y += Math.sin(k * Math.PI) * 0.7; });
}
const VALUE = [
  (w) => w.n, (w) => 2 * w.n, (w) => w.n + 9, (w) => 3 * w.n + 9, (w) => 3 * w.n + 9 + (w.value || 0),
  (w) => Math.floor((3 * w.n + 9 + (w.value || 0)) / 2), (w) => Math.floor((3 * w.n + 9 + (w.value || 0)) / 2) - 7, (w) => Math.floor((3 * w.n + 9 + (w.value || 0)) / 2) - 7,
];
// One adviser's job: the envelope comes to their desk and they work. Their result goes up on their tag.
async function doJob(i, seconds, E) {
  const s = STAFF[i], word = W, travel = Math.min(0.16, seconds * 0.3);
  s.A.busy = true; s.working = true; s.tag.classList.add("work"); s.A.watch = null;
  await hop(E, spot(i), travel);
  await rt(seconds - travel, () => {});
  s.A.busy = false; s.working = false; s.tag.classList.remove("work");
  if (i !== 4) { s.val.textContent = VALUE[i](word); tagSize.delete(s.tag); }
}
const showValue4 = () => { STAFF[4].val.textContent = VALUE[4](W); tagSize.delete(STAFF[4].tag); };
const setChip = (i, cls) => { const c = linesEl.children[i * 2]; if (c) { c.className = cls; } };
// The clerk's list, a line at a time. Line k is SLIP[k]; two names on one line come when the second is called.
function writeSlip(i) {
  const k = SLIP.findIndex((l) => l.includes(i));
  const names = SLIP[k].slice(0, SLIP[k].indexOf(i) + 1).map((n) => STAFF[n].name).join(" + ");
  let chip = linesEl.children[k * 2];
  if (!chip) {
    if (k) { const a = document.createElement("i"); a.textContent = "›"; linesEl.appendChild(a); }
    chip = document.createElement("span"); linesEl.appendChild(chip);
  }
  chip.innerHTML = i === 4 ? `${names}<em> + fax</em>` : names;
}

async function newWord() {
  W = { run: false, lane: "f", f: 0, t: 0, n: NUMBERS[S.total++ % NUMBERS.length], fax: "none", value: 0, sent: false, held: false, called: false };
  for (const s of STAFF) { s.val.textContent = ""; s.tag.classList.remove("work", "wait"); }
  tagSize.clear();
  for (const E of [E1, E2]) { E.stamp.visible = false; E.stamp.scale.set(1, 1, 1); E.slip.visible = false; }
  E2.group.visible = false;
  E1.group.position.copy(L4.tray); E1.group.visible = true; E1.group.scale.setScalar(1);
  drawBoard();
  tween(0.4, (u) => { boardMat.emissiveIntensity = 0.3 + 0.5 * (1 - u); });
}
// The stamp, and the tube. The word's time stops at the stamp.
async function finishWord(E, mode) {
  W.run = false; W.lane = "f";
  setClock(W.f, W.t);
  stampSound();
  E.stamp.visible = true;
  if (!reduced) tween(0.25, (u) => { const k = 1 + 0.6 * Math.sin(u * Math.PI); E.stamp.scale.set(k, 1, k); });
  S.words++;
  if (mode === "slip") S.slipWords++;
  const label2 = mode === "slip" ? "With the list" : "By hand", best = S.best[mode];
  if (!best || W.f + W.t < best.f + best.t) { S.best[mode] = { f: W.f, t: W.t }; addHistory({ label: label2, fetch: W.f, think: W.t }); }
  for (let k = 0; k < 8; k++) if (k !== 7) STAFF[k].A.watch = S.mode === "slip" ? null : camera.position;
  drawBoard();
  const from = E.group.position.clone(), to = tubeMouth(), top = to.clone().setY(4.4);
  await rt(0.14, () => {});
  await rt(0.22, (u) => { if (reduced) return; if (u < 0.4) E.group.position.lerpVectors(from, to, u / 0.4); else E.group.position.lerpVectors(to, top, (u - 0.4) / 0.6); });
  E.group.visible = false;
}

// By hand: one press for each name, each a launch. The clock waits out the narrator's offer of the pad.
async function handWord() {
  for (let i = 0; i < STAFF.length; i++) {
    setNext(i);
    W.lane = "f";
    W.run = !S.quiet;
    paintHint();
    const go = await waitPress();
    if (go === "pad") return "pad";
    W.run = true;
    if (!i) {
      W.called = true;
      if (S.stepped && !S.want && !S.refused) { S.refused = true; if (react) react.say("f4refuse", "now"); }
    }
    S.calls++;
    S.note = "";
    // The next name goes up at once: a tap during this call is kept for when it ends.
    setNext(i < STAFF.length - 1 ? i + 1 : null);
    S.free = false;
    sayName(i);
    airOn(true);
    paCall(T4.CALL + T4.JOB, Math.min(1, S.calls / 40));
    await rt(T4.CALL, () => {});
    W.lane = "t";
    await doJob(i, T4.JOB, E1);
    airOn(false);
    S.free = true;
    if (S.faxWaiting && W.fax === "ringing") { S.faxWaiting = false; W.lane = "f"; await answerFax(); }
    if (i === 2 && faxWord()) ringFax();
    if (i === 4) { await holdForFax(); showValue4(); if (i < STAFF.length - 1) setNext(i + 1); }
    for (const p of STAFF) if (!p.working) p.A.watch = camera.position;
  }
  S.said = "";
  await finishWord(E1, "hand");
  return true;
}
// The last word read out loud, at reading pace, with no presses: the clerk writes down each name as it is called.
async function recordWord() {
  S.mode = "record"; S.live = true; W.called = true; W.run = true; W.f = 0; W.t = 0;
  setNext(null);
  clerk.busy = true;
  $("slip4").hidden = false;
  if (react) react.say("f4rec1", true);
  for (let i = 0; i < STAFF.length; i++) {
    S.calls++;
    sayName(i);
    airOn(true);
    paCall(T4.CALL + T4.JOB, Math.min(1, S.calls / 40));
    W.lane = "f";
    await rt(T4.CALL, () => {});
    W.lane = "t";
    await doJob(i, T4.JOB, E1);
    airOn(false);
    writeSlip(i);
    if (i === 4) showValue4();
  }
  clerk.busy = false; S.said = "";
  hint("");
  await finishWord(E1, "hand");
}
// With the list: one press, then the office works down it. The player's hands are free for the fax.
async function slipWord() {
  setNext(null);
  W.lane = "f";
  // The narrator's account of the list ends on "let go", so the tray lights as it does.
  if (!S.slipWords && react) await until(() => !react.talking);
  S.trayLit = true;
  paintHint();
  await waitPress();
  S.trayLit = false; W.sent = true; W.run = true;
  S.free = false;
  E1.slip.visible = true;
  flick();
  if (react) react.sent();
  await rt(T4.DROP, () => {});
  S.free = true;
  W.lane = "t";
  paintHint();
  for (const c of linesEl.querySelectorAll("span")) c.className = "";
  for (let k = 0; k < SLIP.length; k++) {
    setChip(k, "on");
    const line = SLIP[k];
    if (line.length === 2) {
      E2.group.position.copy(E1.group.position); E2.slip.visible = true;
      // The question goes out as these two start, so an answer in good time holds nobody up.
      if (faxWord()) ringFax();
      await Promise.all([doJob(line[0], T4.PAIR, E1), doJob(line[1], T4.PAIR, E2)]);
    } else {
      if (k === 2) hop(E2, spot(line[0]), 0.16).then(() => { E2.group.visible = false; });
      await doJob(line[0], T4.JOB, E1);
    }
    if (line[0] === 4) {
      await holdForFax();
      W.lane = "t";
      showValue4();
    }
    setChip(k, "done");
  }
  await finishWord(E1, "slip");
}
// By hand the fax rings from the second word on; on the list, the first word runs in silence.
const faxWord = () => (S.mode === "slip" ? S.slipWords % 2 === 1 : S.words >= 1);
function airOn(on) {
  airSign.m.emissiveIntensity = on ? 1.5 : 0.05;
  R4.micLamp.material.color.setHex(on ? 0xff3a2a : 0x40100c);
  for (const h of hornGlow) h.material.opacity = on ? 0.5 : 0;
}

// The narrator stops the shift, turns the player's head to the board, and then to the pad.
async function stepIn() {
  const R = react;
  S.stopped = true; S.stepped = true; S.next = null; S.note = "";
  STAFF.forEach((s) => s.tag.classList.remove("need"));
  paintHint();
  R.hold = true; R.queue.length = 0;
  if (R.talking) skip();
  await until(() => !R.talking);
  await speak("f4stop", (line) => {
    if (line === "f4stop2") aim(L4.face, zoomOn(L4.face, 1.75, 1.45));
    if (line === "f4stop3") { aim(L4.pad, zoomOn(L4.pad, 0.62, 0.46)); R4.padLit = true; }
  });
  $("caption").hidden = true;
  aim(null);
  R.hold = false; S.stopped = false; S.stepAt = performance.now() / 1000;
}

function resetFloor4() {
  const want = innerWidth / innerHeight < 1.15 ? "tall" : "wide";
  if (want !== L4.mode) layout4(want);
  S.on = false; S.live = false; S.stopped = false; S.stepped = false; S.refused = false; S.quiet = false; S.trayLit = false; S.mode = "hand"; S.want = false; S.pad = true; S.calls = 0; S.words = 0; S.faxes = 0; S.total = 0; S.slipWords = 0; S.next = null; S.wake = null; S.buffered = false; S.faxWaiting = false; S.free = true; S.note = ""; S.said = ""; S.hover = null;
  W = { run: false, lane: "f", f: 0, t: 0, n: NUMBERS[0], fax: "none" };
  for (const s of STAFF) { s.A.busy = false; s.A.watch = camera.position; s.A.sit = 1; s.A.hand = false; s.working = false; s.val.textContent = ""; s.tag.classList.remove("need", "work", "wait"); }
  clerk.busy = false;
  for (const E of [E1, E2]) { E.group.visible = false; E.slip.visible = false; E.stamp.visible = false; }
  R4.faxSheet.visible = false; answerPage.visible = false; R4.padLit = false; R4.padG.visible = true;
  const L = LAY[L4.mode];
  R4.padG.position.set(L.pad[0], 0, L.pad[1]);
  airOn(false); routeM.opacity = 0;
  linesEl.innerHTML = ""; $("slip4").hidden = true;
  aim(null);
  drawBoard();
}

// The narrator's remarks during the shift: each at most once, one at most every 5.5 s.
function makeReact() {
  const now = () => performance.now() / 1000;
  const R = { used: new Set(), queue: [], last: -1e9, talking: false, hold: false, t0: now() };
  const play = (id) => {
    R.talking = true; R.last = now();
    speak(id).then(() => {
      R.talking = false; R.t0 = now();
      const more = R.queue.shift();
      if (more) return play(more);
      setTimeout(() => { if (!R.talking && !speaking() && view.floor === 4) $("caption").hidden = true; }, 900);
    });
  };
  // how: true waits its turn after the line in progress, "now" cuts it short, and without it a line is dropped if the
  // narrator is busy or spoke in the last 5.5 s.
  R.say = (id, how) => {
    if (R.used.has(id) || R.hold) return false;
    if (R.talking) {
      if (!how) return false;
      R.used.add(id); R.queue.push(id);
      if (how === "now") skip();
      return true;
    }
    if (!how && now() - R.last < 5.5) return false;
    R.used.add(id); play(id);
    return true;
  };
  R.any = (ids, force) => { const free = ids.filter((x) => !R.used.has(x)); return free.length ? R.say(free[0], force) : false; };
  R.sent = () => { R.any(["f4s1", "f4s3"], true); };
  R.tick = () => {
    if (!S.on || S.stopped || R.talking) return;
    // Said only while still true: the fax may be answered before the narrator is free.
    if (W.held && R.say(S.mode === "slip" ? "f4s6" : "f4fax2", true)) return;
    if (W.fax === "ringing" && S.mode !== "slip" && R.say("f4fax1", true)) return;
    if (!S.live) { if (now() - R.t0 > 10) R.say("f4idle"); return; }
    if (S.quiet && now() - S.stepAt > 7) R.say("f4nudge");
    if (S.mode === "hand" && S.calls >= 24 && R.say("f4hoarse")) return;
    if (S.mode === "hand" && S.words >= 1 && !S.stepped) R.any(["f4same1", "f4same2"]);
    if (S.mode === "slip" && W.sent && W.fax !== "ringing" && S.slipWords >= 1) R.say("f4s5");
  };
  return R;
}

// The opening looks at each thing as it is named. The shift starts on the instructions, so the first click counts.
async function f4Intro() {
  await speak("f4intro", (line) => {
    if (line === "f4a3") tagsBox.hidden = false;
    if (line === "f4a4") aim(L4.face, zoomOn(L4.face, 1.75, 1.45));
    if (line === "f4a5") aim(null);
    if (line === "f4a6") aim(L4.mic, zoomOn(L4.mic, 0.7, 0.55));
  });
  await f4Shift(true, "f4how");
}
// One shift. pad: whether the pad is on the desk. opening: a chunk to speak as it starts.
async function f4Shift(pad = true, opening = null) {
  clearStage();
  resetFloor4();
  setClock(0, 0);
  S.pad = pad; R4.padG.visible = pad;
  tagsBox.hidden = false;
  setLive(true);
  const R = react = makeReact();
  const timer = setInterval(() => R.tick(), 250);
  S.on = true;
  if (opening) R.say(opening, true);
  const over4 = () => S.slipWords >= SLIP_WORDS || (S.mode === "hand" && !S.want && S.words >= BY_HAND);
  while (!over4()) {
    await newWord();
    if (S.mode === "hand" && S.pad && !S.want && !S.stepped && S.words >= ROUNDS) await stepIn();
    if (S.mode === "slip") await slipWord();
    else if (S.want || (await handWord()) === "pad") {
      await recordWord();
      S.mode = "slip";
      showRoute(true);
      R.say("f4rec3", true);
    }
  }
  const won = S.slipWords > 0;
  S.on = false;
  clearInterval(timer);
  W.run = false;
  R.hold = true;
  setNext(null);
  setLive(false);
  document.body.style.cursor = "";
  hint("");
  airOn(false);
  await until(() => !R.talking);
  react = null;
  await speak(won ? "f4win" : pad ? "f4lose" : "f4voice", (line) => {
    if (line === "f4w1") { showTally(); growBar("hand"); }
    if (line === "f4w2") growBar("slip");
  });
  if (won) findEnding("slip", "Standing Orders");
  else findEnding("hoarse", "Hoarse");
  choose(won ? [["What just happened?", reveal4], ["Run it again", again("f4again", true)]] : [["What just happened?", reveal4], ["Use the pad this time", again("f4again", true)]]);
}
// The head turns to the board. In the wide room the board sits left of centre, clear of the stopwatch.
function showTally() {
  const at = L4.face.clone();
  if (L4.mode === "wide") at.x += 0.7;
  aim(at, zoomOn(at, 1.75, 1.45));
}
const backToRoom = () => { aim(null); TALLY.hand = TALLY.slip = 0; drawBoard(); };
const again = (id, pad) => async () => { clearStage(); hint(""); backToRoom(); await toDesk(); await speak(id); $("caption").hidden = true; await f4Shift(pad); };
// Each label appears as the narrator names what it stands for.
const NAMED = { f4x2: "aTeam4", f4x3: "aMic4", f4x4: "aSlip4", f4x5: "aBoard4", f4x6: "aFax4" };
async function reveal4() {
  clearStage();
  hint("");
  backToRoom();
  tagsBox.hidden = true; $("slip4").hidden = true;
  // The word on the desk, its list on it, for the labels.
  E1.group.visible = true; E1.slip.visible = true; E1.stamp.visible = false; E1.group.position.copy(L4.tray); E1.group.scale.setScalar(1);
  labels(true);
  for (const id of config.labels) $(id).classList.remove("on");
  rise();
  if (!routeM.opacity) showRoute(true);
  const more = !(endings[4].has("slip") && endings[4].has("hoarse"));
  $("nextNote").textContent = config.next + (more ? " One ending on this floor is still hidden." : " Both endings found.");
  revealButtons([]);
  await speak("f4reveal", (line) => {
    if (NAMED[line]) $(NAMED[line]).classList.add("on");
    if (line === "f4x7") $("reveal").hidden = false;
  });
  labels(true);
  $("reveal").hidden = false;
  $("caption").hidden = true;
  revealButtons([
    [endings[4].has("slip") ? "Read it all out by hand" : "Use the pad this time", endings[4].has("slip") ? again("f4byhand", false) : again("f4again", true), true],
    ["Replay floor 4", () => { endings[4].clear(); goFloor(4); }],
    ["Back to floor 3", () => goFloor(3)],
  ]);
}

// Floor 4, before each render: the word's clock, rings and beams, hands up, and heads following the word.
let last4 = performance.now();
function scene4(t) {
  const now = performance.now(), dt = (now - last4) / 1000; last4 = now;
  if (W.run && S.live) { W[W.lane] += dt; setClock(W.f, W.t); }
  const pulse = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(t * 7);
  let b = 0;
  for (const [i, s] of STAFF.entries()) {
    const next = S.on && S.next === i && S.mode === "hand" && !S.quiet;
    s.A.hand = next;
    s.ring.material.opacity = next && S.free ? 0.55 + 0.35 * pulse : s.working ? 0.5 : S.hover === HOT[i] ? 0.3 : 0;
    if (s.working && b < 2) { const bm = beams[b++]; bm.position.set(s.A.group.position.x, 1.75, s.A.group.position.z + 0.1); bm.material.color.setHex(s.color); bm.material.opacity = 0.3; }
    if (S.mode === "slip" && S.on && !s.working && !s.A.busy && !s.tag.classList.contains("wait")) s.A.watch = E1.group.position;
  }
  for (; b < 2; b++) beams[b].material.opacity = 0;
  const ring = W.fax === "ringing" && S.on;
  R4.faxRing.material.opacity = ring ? 0.5 + 0.4 * pulse : S.hover === HOT.fax ? 0.3 : 0;
  R4.faxLamp.material.opacity = ring ? 0.5 + 0.4 * pulse : 0;
  R4.padRing.material.opacity = R4.padLit && S.on ? 0.45 + 0.4 * pulse : S.hover === HOT.pad && S.mode === "hand" && !S.want ? 0.35 : 0;
  R4.trayRing.material.opacity = S.trayLit && S.on ? 0.45 + 0.4 * pulse : 0;
  R4.pad.position.y = R4.padLit && !reduced ? 1.05 + pulse * 0.012 : 1.05;
  HOT.pad.tag.classList.toggle("q", !(R4.padLit && S.on)); HOT.pad.tag.classList.toggle("need", !!R4.padLit && S.on);
  HOT.fax.tag.classList.toggle("q", !ring); HOT.fax.tag.classList.toggle("need", ring);
  HOT.tray.tag.classList.toggle("q", !(S.trayLit && S.on)); HOT.tray.tag.classList.toggle("need", S.trayLit && S.on);
}
// Floor 4, after each render: the name tags, and the reveal's labels.
function dom4() {
  if (!tagsBox.hidden) {
    const items = STAFF.map((s) => {
      const g = s.A.group.position, [x, y] = toScreen(v.set(g.x, 1.72, g.z)), [w, h] = sizeOf(s.tag);
      return { el: s.tag, x, w, h, y: y - h };
    });
    for (const h of THINGS) {
      if (h.tag.classList.contains("q")) continue;
      const [x, y] = toScreen(v.copy(h.p).setY(h.p.y + 0.05)), [w, ht] = sizeOf(h.tag);
      items.push({ el: h.tag, x, w, h: ht, y: y - ht });
    }
    placeTags(items);
    // Someone out of frame has no tag: held at the screen's edge, the tags would pile up there. A lit name always shows.
    for (const it of items) it.el.style.visibility = (it.x < 0 || it.x > innerWidth) && !it.el.classList.contains("need") ? "hidden" : "";
  }
  placeLabelsApart(anchors4);
}

resetFloor4();
closeFloor(4, opened);

export default {
  ...config, camera: camera4, lift, start: f4Intro,
  enter() { S.best = {}; resetFloor4(); H4.up = 0; upperWalls[L4.mode].visible = true; H4.look.copy(H4.rest); H4.f = H4.fov; camera.clearViewOffset(); },
  leave() { tagsBox.hidden = true; $("slip4").hidden = true; hint(""); setLive(false); S.on = false; document.body.style.cursor = ""; },
  frame(t, dt) {
    for (const s of STAFF) poseAdviser(s.A, t, dt);
    poseAdviser(clerk, t, dt);
    scene4(t);
    R4.city(t);
  },
  afterRender: dom4,
  onLabels(on) { tagSize.clear(); document.body.classList.toggle("reveal4", on); },
  // Between shifts a turned or resized screen may suit the other room.
  resized() {
    tagSize.clear(); fit4.key = "";
    if (view.floor === 4 && !S.on) {
      const want = innerWidth / innerHeight < 1.15 ? "tall" : "wide";
      if (want !== L4.mode) { layout4(want); fit4.snap = true; }
    }
    frameHead();
  },
};
