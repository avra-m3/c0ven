// The street, the outside of the tower and its lobby, around x = X0: what the start screen shows, and where the first
// lift ride sets off. It has no narration and no stopwatch.
import { box, camera, canvasTex, closeFloor, halo, mat, openFloor, pool, rep, scene, sign } from "../world.js";
import { makeAdviser, makeChair, makePlant, poseAdviser } from "../props.js";
import { asphaltTex, block, city, curtainWall, makeLift, plaster, slabTex, traffic, wallWithDoor } from "../tower.js";

const X0 = -140, opened = openFloor();
const TW = 20, TH = 110, LOBBY_H = 5, LOBBY_D = 12;
const doorsGlass = [];

const blink = city(scene, X0, 0, { near: -30 });
const cars = traffic(scene, 0, 12, X0 - 90, X0 + 90, 8);
// The street.
const road = new THREE.Mesh(new THREE.PlaneGeometry(320, 22), mat(0xffffff, { map: rep(asphaltTex, 80, 6), roughness: 0.55, metalness: 0.15 }));
road.rotation.x = -Math.PI / 2; road.position.set(X0, 0, 17); road.receiveShadow = true; scene.add(road);
box(320, 0.14, 6.2, mat(0xffffff, { map: rep(slabTex, 200, 4), roughness: 0.9 }), X0, 0.07, 3.1);
box(320, 0.14, 8, mat(0xffffff, { map: rep(slabTex, 200, 5), roughness: 0.9 }), X0, 0.07, 32);
for (let x = -150; x <= 150; x += 5) box(2.4, 0.02, 0.16, mat(0xd8d2b8, { roughness: 0.8 }), X0 + x, 0.02, 17);
// The tower: glass all the way up, its top lost in the night, with a sign running up one corner.
const tower = new THREE.Group(); tower.position.x = X0; scene.add(tower);
curtainWall(tower, 0, 0, TW, LOBBY_H, TH, 2.3, 0xffffff);
const side = curtainWall(tower, TW / 2, -9, 18, LOBBY_H, TH, 2.3, 0x8f97a4); side.rotation.y = Math.PI / 2;
const side2 = curtainWall(tower, -TW / 2, -9, 18, LOBBY_H, TH, 2.3, 0x8f97a4); side2.rotation.y = -Math.PI / 2;
box(TW + 0.6, 0.5, 0.6, mat(0x20252e), 0, LOBBY_H + 0.1, 0.05, tower);
box(TW - 3, 7, 12, mat(0x171b22), 0, TH + 3.5, -9, tower);
const crown = sign("MERIDIAN WORDS", 15, 2, 0, TH + 4, -2.9, "#16120c", "#f0ad45", tower); crown.material = new THREE.MeshBasicMaterial({ map: crown.material.map, fog: false });
const beacon = halo(0xff4a3a, 3, 0, TH + 9.5, -9, tower, 0.9); beacon.material.fog = false;
const blade = canvasTex(128, 1024, (g, w, h) => {
  g.fillStyle = "#16120c"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#f0ad45"; g.lineWidth = 5; g.strokeRect(6, 6, w - 12, h - 12);
  g.fillStyle = "#ffc266"; g.textAlign = "center"; g.textBaseline = "middle"; g.font = "700 104px 'Barlow Condensed', Arial Narrow, sans-serif";
  [..."MERIDIAN"].forEach((c, i) => g.fillText(c, w / 2, 76 + i * 124));
});
const bladeM = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 19), new THREE.MeshBasicMaterial({ map: blade, fog: false }));
bladeM.position.set(TW / 2 + 1.4, 18, 0.6); tower.add(bladeM);
halo(0xffb45c, 18, TW / 2 + 1.4, 18, 0.4, tower, 0.16);
// Next door: lower, older, brick.
block(scene, X0 + 27, -6, 28, 14, 0, 16, 2, 0xc9a58a);
block(scene, X0 - 28, -8, 30, 14, 0, 24, 1);
// Street lamps with their light on the pavement, and a wet shine on the road under the lobby.
for (let x = -44; x <= 44; x += 22) {
  const p = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.1, 5.2, 8), mat(0x2c3038)); p.position.set(X0 + x, 2.6, 6.1); scene.add(p);
  box(1.1, 0.07, 0.1, mat(0x2c3038), X0 + x, 5.2, 6.4).rotation.y = Math.PI / 2;
  halo(0xffb45c, 2.4, X0 + x, 5.05, 6.8, scene, 0.8);
  pool(scene, 0xffa850, 4.5, X0 + x, 0.16, 6.2, 0.3);
  pool(scene, 0xffa850, 2.2, X0 + x, 0.03, 10, 0.1);
}
pool(scene, 0xffd9a0, 11, X0, 0.16, 1.4, 0.38).scale.set(1.2, 0.5, 1);
pool(scene, 0xffd9a0, 11, X0, 0.03, 8.5, 0.14).scale.set(1, 0.42, 1);

// The lobby, seen through its glass front.
const terrazzo = canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = "#5c5a54"; g.fillRect(0, 0, w, h);
  for (let k = 0; k < 1300; k++) { g.fillStyle = ["#7c7a72", "#3e3c38", "#918b7c", "#4c5058"][k % 4]; g.fillRect((k * 67) % w, (k * 131 + (k >> 2)) % h, 3, 2); }
  g.fillStyle = "#2a2926"; g.fillRect(0, 0, w, 3); g.fillRect(0, 0, 3, h);
});
terrazzo.wrapS = terrazzo.wrapT = THREE.RepeatWrapping; terrazzo.repeat.set(10, 6);
const L = new THREE.Group(); L.position.x = X0; scene.add(L);
box(TW, 0.12, LOBBY_D, mat(0xffffff, { map: terrazzo, roughness: 0.35, metalness: 0.08 }), 0, 0.06, -LOBBY_D / 2, L);
for (const s of [-1, 1]) box(0.2, LOBBY_H, LOBBY_D, plaster, s * TW / 2, LOBBY_H / 2, -LOBBY_D / 2, L);
box(TW, 0.2, LOBBY_D, mat(0x2c2a26), 0, LOBBY_H, -LOBBY_D / 2, L);
// Dark polished marble with pale veins, in book-matched slabs.
const marbleTex = canvasTex(512, 512, (g, w, h) => {
  g.fillStyle = "#2a2f30"; g.fillRect(0, 0, w, h);
  for (let k = 0; k < 42; k++) {
    let x = (k * 137) % w, y = (k * 211) % h;
    g.strokeStyle = k % 5 ? "rgba(168,172,160,0.16)" : "rgba(214,206,178,0.42)"; g.lineWidth = k % 5 ? 1 : 1.6;
    g.beginPath(); g.moveTo(x, y);
    for (let s = 0; s < 8; s++) { x += 18 + (k * s * 7) % 40; y += ((k * 13 + s * 29) % 54) - 18; g.lineTo(x, y); }
    g.stroke();
  }
  g.fillStyle = "rgba(0,0,0,0.5)"; g.fillRect(0, 0, 3, h); g.fillRect(0, 0, w, 3);
});
marbleTex.wrapS = marbleTex.wrapT = THREE.RepeatWrapping; marbleTex.repeat.set(TW / 2.5, LOBBY_H / 2.5);
const back = -LOBBY_D, marble = mat(0xffffff, { map: marbleTex, roughness: 0.3, metalness: 0.1 });
wallWithDoor(L, marble, -TW / 2, TW / 2, LOBBY_H, back, 0, 0.2, 0.12);
const lift = makeLift(L, { x: 0, y: 0.12, z: back + 0.005, number: 0, name: "Lobby", signSide: 0 });
// A second lift, with a sign that explains itself.
box(1.54, 2.39, 0.1, mat(0xb08a4a, { metalness: 0.7, roughness: 0.35 }), -4.2, 1.31, back + 0.02, L);
box(1.3, 2.25, 0.1, mat(0x7c8088, { metalness: 0.45, roughness: 0.42 }), -4.2, 1.25, back + 0.05, L);
sign("OUT OF ORDER", 1.05, 0.28, -4.2, 1.5, back + 0.11, "#e0c94f", "#1a1a12", L).rotation.z = -0.06;
// The building's name in brass, and the directory.
sign("MERIDIAN WORDS LTD.", 7, 0.72, 0, 3.75, back + 0.02, "#2c2d30", "#d2ab62", L);
const directory = canvasTex(360, 512, (g, w, h) => {
  g.fillStyle = "#14171c"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#b08a4a"; g.lineWidth = 8; g.strokeRect(5, 5, w - 10, h - 10);
  g.fillStyle = "#e6dfc9"; g.font = "700 36px 'Barlow Condensed', Arial Narrow, sans-serif"; g.textAlign = "center"; g.fillText("DIRECTORY", w / 2, 54);
  g.textAlign = "left"; g.font = "500 28px 'Barlow Condensed', Arial Narrow, sans-serif";
  [["01", "THE SHELF ROOM"], ["02", "THE ANNEX"], ["03", "MUSICAL DESKS"], ["04", "THE BRIEFING"], ["05", "CLOSED"], ["...", ""], ["48", "THE ROOF"]].forEach(([n, t], i) => {
    g.fillStyle = "#e8a33d"; g.fillText(n, 34, 112 + i * 52); g.fillStyle = i < 4 ? "#e6dfc9" : "#7d8088"; g.fillText(t, 96, 112 + i * 52);
  });
});
const dirM = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 2.1), new THREE.MeshStandardMaterial({ map: directory, roughness: 0.4, emissive: 0x332a18, emissiveMap: directory, emissiveIntensity: 0.9 }));
dirM.position.set(3.5, 1.65, back + 0.02); L.add(dirM);
// The reception desk, a night guard who has seen enough, and plants.
const desk = mat(0x4a3324, { roughness: 0.45 });
box(4.2, 1.1, 0.8, desk, 6, 0.67, -6.2, L);
box(4.4, 0.06, 1, mat(0x6e6a60, { roughness: 0.25, metalness: 0.2 }), 6, 1.25, -6.1, L);
sign("RECEPTION", 1.8, 0.28, 6, 0.8, -5.78, "#2c2d30", "#d2ab62", L);
makeChair(X0 + 6, -7.3);
const guard = makeAdviser({ x: X0 + 6, z: -7.1, suit: 0x2f3b52, hair: 0x4a3b2c, tie: 0x1c2027, old: true });
guard.sit = 1;
const lampG = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.2, 0.2, 14), mat(0x2f7a4c, { emissive: 0x2f7a4c, emissiveIntensity: 0.6 })); lampG.position.set(4.6, 1.6, -6.1); L.add(lampG);
box(0.03, 0.3, 0.03, mat(0xb08a4a), 4.6, 1.4, -6.1, L);
halo(0xffd9a0, 1.4, 4.6, 1.55, -6.1, L, 0.4);
for (const x of [-8.6, 8.6]) makePlant(L, x, -LOBBY_D + 1, 1.4);
makePlant(L, -1.6, -LOBBY_D + 0.9, 0.9); makePlant(L, 1.6, -LOBBY_D + 0.9, 0.9);
// Ceiling lights: bright glow from outside, one real light inside.
for (let i = -1; i <= 1; i++) for (const z of [-3, -8]) {
  const panel = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.03, 0.8), new THREE.MeshBasicMaterial({ color: 0xffefd0 }));
  panel.position.set(i * 6, LOBBY_H - 0.1, z); L.add(panel);
  pool(L, 0xffdcae, 2.6, i * 6, 0.135, z, 0.16);
}
const warm = new THREE.PointLight(0xffe0b0, 1.1, 26, 1.3); warm.position.set(0, 4, -5); L.add(warm);
const warm2 = new THREE.PointLight(0xffd9a0, 0.5, 18, 1.5); warm2.position.set(0, 3, 2); L.add(warm2);

// The glass front, with sliding doors that open for the player.
const glassM = new THREE.MeshStandardMaterial({ color: 0x8fa3b5, transparent: true, opacity: 0.14, roughness: 0.08, metalness: 0.5, depthWrite: false, side: THREE.DoubleSide });
const frameM = mat(0x20252e);
for (const [a, b] of [[-TW / 2, -2], [2, TW / 2]]) {
  const g = new THREE.Mesh(new THREE.PlaneGeometry(b - a, LOBBY_H), glassM); g.position.set((a + b) / 2, LOBBY_H / 2, 0); L.add(g);
  for (let x = a; x <= b + 0.01; x += (b - a) / 3) box(0.12, LOBBY_H, 0.14, frameM, x, LOBBY_H / 2, 0, L);
}
box(4, 0.3, 0.16, frameM, 0, 3.1, 0, L);
for (const s of [-1, 1]) {
  const leaf = new THREE.Group(); leaf.position.set(s, 0, 0); L.add(leaf);
  const g = new THREE.Mesh(new THREE.PlaneGeometry(2, 2.95), glassM); g.position.y = 1.55; leaf.add(g);
  for (const dx of [-1, 1]) box(0.06, 2.95, 0.06, frameM, dx, 1.55, 0, leaf);
  doorsGlass.push({ leaf, s });
}
const canopy = box(9, 0.22, 3.4, mat(0x20252e), 0, 3.5, 1.7, L);
canopy.castShadow = false;
sign("MERIDIAN WORDS", 6, 0.58, 0, 3.86, 3.42, "#14171c", "#f0ad45", L).material.fog = false;
for (const x of [-3.2, 0, 3.2]) halo(0xffe0b0, 0.9, x, 3.34, 2.4, L, 0.6);
pool(L, 0xffdcae, 3.6, 0, 0.16, 2.4, 0.22);

// ---------- Camera ----------
const camera0 = {
  look: new THREE.Vector3(), offset: true, par: 1.2, fog: [80, 300],
  pos: new THREE.Vector3(),
  place() {
    const t = performance.now() / 1000, tall = innerWidth / innerHeight < 0.9;
    camera0.fov = tall ? 66 : 58;
    if (camera.view && camera.view.enabled) camera.clearViewOffset();
    camera0.pos.set(X0 + (tall ? 8 : -14) + Math.sin(t * 0.09) * 1.2, 2.6, tall ? 76 : 66);
    camera0.look.set(X0 + (tall ? 0 : -11), tall ? 44 : 27, 0);
    return camera0.pos;
  },
};

const route = () => [new THREE.Vector3(X0, 5.2, 34), new THREE.Vector3(X0, 2.2, 9), new THREE.Vector3(X0, 1.8, 1), new THREE.Vector3(X0, 1.75, -5)];
Object.assign(lift, {
  get route() { return route(); }, walk: 4.2, doorAt: 2.7,
  // The front doors slide apart just before the camera reaches them.
  onWalk(u) { const o = Math.min(1, Math.max(0, (u - 0.3) / 0.22)); for (const { leaf, s } of doorsGlass) leaf.position.x = s * (1 + o * 1.9); },
});

closeFloor(0, opened);

export default {
  number: 0, name: "Lobby", minutes: 0, scale: 1, keys: ["", ""], labels: [], map: [], lesson: "",
  camera: camera0, lift, enter() {}, start() {},
  frame(t, dt) {
    poseAdviser(guard, t, dt);
    guard.head.rotation.x = 0.45;
    blink(t); cars(dt);
    beacon.material.opacity = (t % 2.2) < 0.9 ? 0.9 : 0.1;
  },
};
