// Floor 1, the shelf room, around x = 0. Reading the model's weights is the slow part, so move them closer.
import { $, box, keyLight, mat, reduced, scene, sign, tube, up, view, windowsTex } from "../world.js";
import { backOut, ease, sleep, tween } from "../motion.js";
import { makeAdviser, makeChair, makeDesk, makeEnvelope, poseAdviser, sitAndThink, wood, woodDark } from "../props.js";
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

(() => {
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 14), mat(0x2f3540, { roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(30, 6), mat(0x323b4b));
  wall.position.set(0, 3, -2.6); wall.receiveShadow = true; scene.add(wall);
  box(30, 0.12, 0.06, mat(0x262c37), 0, 0.06, -2.56);
  const win = new THREE.Mesh(new THREE.PlaneGeometry(13, 1.6), new THREE.MeshBasicMaterial({ map: windowsTex }));
  win.position.set(2.2, 2.85, -2.58); scene.add(win);
  for (let i = 0; i <= 8; i++) box(0.06, 1.7, 0.06, mat(0x262c37), -4.3 + i * 13 / 8, 2.85, -2.55);
  box(13.2, 0.08, 0.1, mat(0x262c37), 2.2, 2.02, -2.55);
  box(13.2, 0.08, 0.1, mat(0x262c37), 2.2, 3.68, -2.55);
  sign("FLOOR 1", 0.9, 0.34, -2.9, 2.85, -2.56);
  keyLight(2.5, 7, 4, 2.5, Math.PI / 4.5, 26);
  for (const x of [-0.6, 3.2, 7]) tube(x, 4.4, 0.4);
  for (const x of [5.2, 9.5]) { box(1.8, 0.08, 0.9, wood, x, 0.98, -1.6); box(0.5, 0.9, 0.8, woodDark, x + 0.55, 0.47, -1.6); }
  const plant = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 0), mat(0x3f5a4b, { flatShading: true }));
  plant.position.set(-3.4, 1.0, -1.9); plant.castShadow = true; scene.add(plant);
  box(0.36, 0.5, 0.36, mat(0x55473a), -3.4, 0.25, -1.9);
})();
makeDesk(0.35, 1.35);
// Just behind where the adviser stands, so sitting down lands on the seat.
makeChair(0.35, 0.15);

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
let k = 0;
for (let s = 0; s < 5; s++) {
  const y = 0.06 + s * 0.56;
  box(1.2, 0.05, 0.52, shelfWood, 0, y, 0, shelfBody);
  if (s === 4) break;
  let x = -0.54;
  while (x < 0.5) {
    const w = 0.05 + ((k * 7) % 5) * 0.012, h = 0.34 + ((k * 5) % 7) * 0.025;
    if (x + w > 0.56) break;
    box(w, h, 0.36, mat(spineCols[k % spineCols.length]), x + w / 2, y + 0.025 + h / 2, 0.02, shelfBody);
    x += w + 0.008; k++;
  }
}
shelf.position.set(FAR, 0, 0);
shelf.rotation.y = -0.18;

const HOME = 0.35;
const A1 = makeAdviser({ x: HOME, z: 0.45 });
const adv = A1.group;
// The manual, carried flat across both hands; rotated back out of the forearm's tilt in the carry pose.
const handBook = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.07, 0.24), mat(0x4f6a63));
handBook.position.set(-0.25, 0.02, -0.04); handBook.rotation.x = 1.95; handBook.visible = false; A1.armR.hand.add(handBook);

const E1 = makeEnvelope();
const ENV_IN = new THREE.Vector3(0.15, 1.03, 1.1), ENV_OUT = new THREE.Vector3(-4.5, 1.6, 0.8);
E1.group.position.copy(ENV_OUT);
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

const camera1 = { look: new THREE.Vector3(2.4, 1.05, 0), wide: new THREE.Vector3(2.6, 3.0, 8.2), narrow: new THREE.Vector3(2.2, 3.4, 10.5), fog: [11, 24], par: 1 };
camera1.place = () => innerWidth / innerHeight < 0.9 ? camera1.narrow : camera1.wide;
const anchors = [
  [$("aAdv"), () => up(adv, 2.2)],
  [$("aShelf"), () => up(shelf, 2.65)],
  [$("aBook"), () => up(deskBook, 0.2)],
  [$("aPath"), () => new THREE.Vector3((HOME + shelf.position.x - 0.8) / 2, 0.1, 0.6)],
];

const squash = (u, kk) => shelfBody.scale.set(1 + 0.6 * kk * Math.sin(u * Math.PI), 1 - kk * Math.sin(u * Math.PI), 1 + 0.6 * kk * Math.sin(u * Math.PI));

// The shelf crouches, pops its legs out, looks both ways, waddles to `to` in `seconds`, and sets itself down.
async function moveShelf(to, seconds) {
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
}

// Floor 1: one word delivered and stamped. The clock counts only the adviser's fetching and thinking.
async function deliver(run) {
  setClock(0, 0);
  E1.stamp.visible = false; deskBook.visible = false;
  await tween(0.7, (u) => E1.group.position.lerpVectors(ENV_OUT, ENV_IN, ease(u)));
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
    handBook.visible = A1.carry = !there;
    setClock(run.fetch * u, 0);
  });
  A1.walking = 0;
  adv.position.x = HOME;
  handBook.visible = A1.carry = false; deskBook.visible = true;
  await sitAndThink(A1, run.fetch, run.think);
  E1.stamp.visible = true;
  addHistory(run);
  await sleep(0.4);
}

function resetFloor1() {
  shelf.position.x = FAR; setPath();
  adv.position.set(HOME, 0, 0.45); adv.rotation.y = A1.heading = 0; A1.sit = 0; A1.reach = false;
  E1.group.position.copy(ENV_OUT); E1.stamp.visible = false; deskBook.visible = false;
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

export default {
  ...config, anchors, camera: camera1, enter: resetFloor1, start: intro,
  frame(t, dt) { poseAdviser(A1, t, dt); },
  onLabels(on) { pathLine.material.opacity = on ? 0.95 : 0; },
};
