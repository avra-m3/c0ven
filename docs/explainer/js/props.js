// Furniture and people, used by more than one floor, and how an adviser walks and thinks.
import { $, box, canvasTex, halo, mat, pool, reduced, scene, view } from "./world.js";
import { ease, lerp, tween } from "./motion.js";
import { setClock } from "./hud.js";

// Furniture builders
export const wood = mat(0x6b5845, { roughness: 0.7 }), woodDark = mat(0x4e4135), paperM = mat(0xeceee8, { roughness: 0.6 });
// A desk centred at (cx, cz), facing the camera, with drawers on its right, papers on its left and a lamp at the back left.
export function makeDesk(cx, cz, lamp = true) {
  box(2.0, 0.08, 1.0, wood, cx, 0.98, cz);
  box(2.0, 0.03, 0.06, woodDark, cx, 0.925, cz + 0.47);
  for (const [dx, dz] of [[-0.9, -0.4], [0.9, -0.4], [-0.9, 0.4], [0.9, 0.4]]) box(0.07, 0.94, 0.07, woodDark, cx + dx, 0.47, cz + dz);
  box(0.6, 0.6, 0.9, woodDark, cx + 0.6, 0.64, cz);
  for (const dy of [0.45, 0.72]) box(0.14, 0.02, 0.02, mat(0x8a9099, { metalness: 0.6, roughness: 0.4 }), cx + 0.6, dy, cz + 0.46);
  const stack = box(0.3, 0.035, 0.22, paperM, cx - 0.55, 1.04, cz + 0.2); stack.rotation.y = 0.15;
  const mug = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.04, 0.1, 12), mat(0xe8a33d, { roughness: 0.5 }));
  mug.position.set(cx + 0.78, 1.07, cz - 0.3); mug.castShadow = true; scene.add(mug);
  if (!lamp) return;
  const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.5, 8), mat(0x8a9099, { metalness: 0.6, roughness: 0.4 }));
  arm.position.set(cx - 0.7, 1.25, cz - 0.25); arm.rotation.z = 0.25; scene.add(arm);
  box(0.16, 0.02, 0.16, mat(0x8a9099, { metalness: 0.6, roughness: 0.4 }), cx - 0.76, 1.03, cz - 0.25);
  const shade = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.2, 18, 1, true), mat(0xe8a33d, { side: THREE.DoubleSide, emissive: 0x6a4210, emissiveIntensity: 0.6 }));
  shade.position.set(cx - 0.63, 1.52, cz - 0.25); scene.add(shade);
  halo(0xffb45c, 0.5, cx - 0.63, 1.46, cz - 0.2, scene, 0.35);
  const l = new THREE.PointLight(0xffb45c, 0.9, 3.2, 2); l.position.set(cx - 0.6, 1.4, cz - 0.2); scene.add(l);
}
// A desk centred at (x, z), the adviser behind it facing out. The office's: pale laminate, a slim lamp, a laptop. The
// annex's: dark oak, a green banker's lamp, books. Lamps are emissive with a halo and a pool of light.
const laminate = mat(0x9ba1a9, { roughness: 0.5 }), steel3 = mat(0x6d737c, { metalness: 0.5, roughness: 0.45 }), plastic = mat(0x23272f, { roughness: 0.5 });
const oak = mat(0x4a3324, { roughness: 0.6 }), leather = mat(0x2d4a39, { roughness: 0.7 }), brassM = mat(0xb08a4a, { metalness: 0.7, roughness: 0.35 });
export { laminate, oak, steel3 };
export function roomDesk(room, x, z, annex, { books = true } = {}) {
  const top = box(1.8, 0.07, 0.9, annex ? oak : laminate, x, 0.985, z, room);
  top.castShadow = true;
  if (annex) {
    box(1.4, 0.006, 0.55, leather, x - 0.1, 1.022, z + 0.05, room);
    box(1.7, 0.8, 0.05, oak, x, 0.53, z + 0.4, room);
    for (const sx of [-0.85, 0.85]) box(0.07, 0.95, 0.86, oak, x + sx, 0.475, z, room);
    if (books) for (const [w, h, c, dy] of [[0.3, 0.07, 0x6b3a32, 0], [0.27, 0.06, 0x2f4a5c, 0.065], [0.29, 0.05, 0x7a6a3a, 0.12]]) {
      const b = box(w, h, 0.21, mat(c), x + 0.62, 1.055 + dy, z - 0.2, room); b.rotation.y = 0.1 - dy;
    }
    // Banker's lamp: brass stem, green half-shade lit from under.
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.09, 0.03, 16), brassM); base.position.set(x - 0.62, 1.035, z - 0.22); room.add(base);
    box(0.02, 0.26, 0.02, brassM, x - 0.62, 1.17, z - 0.22, room);
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
// A wall clock, facing +z. Returns its minute and hour hands, each a pivot to turn.
export function makeClock(parent, x, y, z, r = 0.22) {
  const dark = mat(0x1a1f27), face = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.04, 28), mat(0xe9ebe6, { emissive: 0x30343a, emissiveIntensity: 0.3 }));
  face.rotation.x = Math.PI / 2; face.position.set(x, y, z); parent.add(face);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(r, r * 0.11, 6, 28), dark); rim.position.set(x, y, z + 0.02); parent.add(rim);
  for (let k = 0; k < 12; k++) box(0.015, k % 3 ? 0.03 : 0.05, 0.01, dark, x + Math.sin(k * Math.PI / 6) * r * 0.77, y + Math.cos(k * Math.PI / 6) * r * 0.77, z + 0.025, parent).rotation.z = -k * Math.PI / 6;
  const hand = (len, rot) => { const p = new THREE.Group(); p.position.set(x, y, z + 0.03); p.rotation.z = -rot; parent.add(p); box(0.018, len, 0.01, dark, 0, len / 2, 0, p); return p; };
  return { minute: hand(r * 0.73, 1.05), hour: hand(r * 0.45, 2.15) };
}
// A water cooler. bubble is a sprite the floor can send up the bottle.
export function makeCooler(parent, x, z) {
  box(0.34, 0.9, 0.34, mat(0xc9ccc4, { roughness: 0.5 }), x, 0.55, z, parent);
  box(0.06, 0.03, 0.02, mat(0x4f8fd0), x - 0.07, 0.8, z + 0.18, parent); box(0.06, 0.03, 0.02, mat(0xc2453d), x + 0.07, 0.8, z + 0.18, parent);
  const bottle = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.42, 16), mat(0x7fb3d9, { transparent: true, opacity: 0.75, roughness: 0.15, emissive: 0x1d3a52, emissiveIntensity: 0.5 }));
  bottle.position.set(x, 1.21, z); parent.add(bottle);
  const bubble = halo(0xdff0ff, 0.09, x, 1.02, z + 0.05, parent, 0);
  return { bubble, x, z };
}
// A potted plant: a fan of long leaves.
export function makePlant(parent, x, z, size = 1) {
  const pot = new THREE.Mesh(new THREE.CylinderGeometry(0.2 * size, 0.15 * size, 0.36 * size, 14), mat(0x5a4a3c)); pot.position.set(x, 0.18 * size + 0.1, z); pot.castShadow = true; parent.add(pot);
  const leafM = mat(0x3f5a4b, { flatShading: true }), leafN = mat(0x4f6e58, { flatShading: true });
  for (let k = 0; k < 11; k++) {
    const a = k * 2.4, up = 0.45 + (k % 4) * 0.2, L = (0.55 + (k % 3) * 0.14) * size;
    const pivot = new THREE.Group(); pivot.position.set(x, 0.42 * size + 0.1, z); pivot.rotation.set(0, a, 0); parent.add(pivot);
    const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.5, 8, 6), k % 2 ? leafM : leafN); leaf.scale.set(0.2 * size, L * 1.2, 0.05); leaf.position.set(Math.cos(up) * L * 0.55, Math.sin(up) * L * 0.6, 0); leaf.rotation.z = up - Math.PI / 2; leaf.castShadow = true; pivot.add(leaf);
  }
}
// A filing cabinet with labelled drawers, its front towards the camera.
export function makeCabinet(parent, x, z, drawers = 3) {
  const h = 0.38 * drawers + 0.04, m = mat(0x4d535c, { roughness: 0.5 });
  box(0.5, h, 0.5, m, x, h / 2 + 0.1, z, parent);
  for (let d = 0; d < drawers; d++) {
    const y = 0.1 + 0.2 + d * 0.38;
    box(0.44, 0.34, 0.012, mat(0x565c66, { roughness: 0.45 }), x, y, z + 0.255, parent);
    box(0.16, 0.02, 0.02, steel3, x, y - 0.03, z + 0.27, parent);
    box(0.12, 0.05, 0.004, paperM, x, y + 0.08, z + 0.264, parent);
  }
}
export function makeCoatStand(parent, x, z, coat = 0x6a4a3a) {
  const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, 1.8, 8), mat(0x3b3029)); stand.position.set(x, 1.0, z); stand.castShadow = true; parent.add(stand);
  const c = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.17, 0.8, 10), mat(coat)); c.position.set(x - 0.08, 1.35, z); c.castShadow = true; parent.add(c);
  for (const a of [0.7, 2.8, 4.6]) { const peg = box(0.14, 0.02, 0.02, mat(0x3b3029), x + Math.cos(a) * 0.06, 1.82, z + Math.sin(a) * 0.06, parent); peg.rotation.y = -a; }
}
// The annex's cat, asleep. Returns the body, whose y scale is its breathing.
export function makeCat(parent, x, y, z, rotY = -0.6) {
  const fur = mat(0x9a7b5a, { roughness: 1 });
  const cat = new THREE.Group(); cat.position.set(x, y, z); cat.rotation.y = rotY; parent.add(cat);
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.15, 14, 10), fur); body.scale.set(1.3, 0.62, 1); body.castShadow = true; cat.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 10), fur); head.position.set(0.17, 0.0, 0.05); cat.add(head);
  for (const ez of [-0.035, 0.035]) { const ear = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.05, 6), fur); ear.position.set(0.18, 0.07, 0.05 + ez); cat.add(ear); }
  const tail = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.022, 6, 14, Math.PI * 1.1), fur); tail.rotation.x = Math.PI / 2; tail.position.set(-0.02, -0.04, 0.02); cat.add(tail);
  return body;
}
// A wall of books, painted on a plane; rows of shelves in w by h units.
export function makeLibrary(parent, x, y, z, w, h, rows = 5) {
  const tex = canvasTex(1024, Math.round(1024 * h / w), (g, cw, ch) => {
    g.fillStyle = "#2c2520"; g.fillRect(0, 0, cw, ch);
    const cols = ["#5d6b7f", "#7a6a55", "#4f6a63", "#6d7486", "#8a5a48", "#3f4e66", "#6b5a3a"], rh = ch / rows;
    let n = 0;
    for (let row = 0; row < rows; row++) {
      const y1 = (row + 1) * rh - 12;
      let bx = 10;
      while (bx < cw - 16) {
        const bw = 9 + ((n * 7) % 6) * 3, bh = (rh - 32) * (0.78 + ((n * 5) % 7) * 0.035);
        g.fillStyle = cols[n % cols.length]; g.fillRect(bx, y1 - bh, bw, bh);
        bx += bw + 2; n++;
      }
      g.fillStyle = "#4a3d31"; g.fillRect(0, y1, cw, 12);
    }
  });
  const lib = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9 }));
  lib.position.set(x, y, z); parent.add(lib); return lib;
}
// A library ladder leaning on the shelves at (x, z), rails along +z of the wall.
export function makeLadder(parent, x, z, h = 3.3) {
  const m = mat(0x6b5845);
  for (const lx of [-1, 1]) { const rail = box(0.05, h, 0.05, m, x + lx * 0.2, h / 2, z, parent); rail.rotation.x = -0.1; }
  for (let k = 0; k < 7; k++) box(0.45, 0.03, 0.04, m, x, 0.3 + k * 0.45, z - 0.04 * k + 0.12, parent);
}
// A red rug with a brass border, for the annex.
export function makeRug(parent, x, y, z, w, d) {
  const tex = canvasTex(128, 256, (g, cw, ch) => {
    g.fillStyle = "#5a2a2a"; g.fillRect(0, 0, cw, ch);
    g.strokeStyle = "#b08a4a"; g.lineWidth = 5; g.strokeRect(10, 10, cw - 20, ch - 20);
    g.strokeStyle = "#3a1c1e"; g.lineWidth = 3; g.strokeRect(22, 22, cw - 44, ch - 44);
    g.fillStyle = "#7a3a36"; for (let k = 0; k < 5; k++) { g.beginPath(); g.ellipse(64, 48 + k * 40, 22, 12, 0, 0, Math.PI * 2); g.fill(); }
  });
  const r = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshStandardMaterial({ map: tex, roughness: 1 }));
  r.rotation.x = -Math.PI / 2; r.position.set(x, y, z); r.receiveShadow = true; parent.add(r); return r;
}
// A typewriter on a desk. sheet sticks up from the platen; carriage shifts with each line typed.
export function makeTypewriter(parent, x, y, z, rotY = 0) {
  const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = rotY; parent.add(g);
  const body = mat(0x2b2f36, { roughness: 0.4, metalness: 0.2 }), key = mat(0xc9ccc4, { roughness: 0.5 });
  const bed = box(0.46, 0.09, 0.32, body, 0, 0.045, 0, g); bed.rotation.x = 0.06;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 7; c++) box(0.04, 0.012, 0.03, key, -0.15 + c * 0.05 + r * 0.008, 0.1 - r * 0.012, 0.1 - r * 0.05 + 0.02, g);
  const carriage = new THREE.Group(); carriage.position.set(0, 0.13, -0.11); g.add(carriage);
  const platen = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.5, 12), body); platen.rotation.z = Math.PI / 2; platen.castShadow = true; carriage.add(platen);
  const lever = box(0.018, 0.018, 0.1, mat(0xb9bcb5, { metalness: 0.6, roughness: 0.3 }), -0.27, 0.02, 0.03, carriage); lever.rotation.y = 0.5;
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(0.24, 0.26), new THREE.MeshStandardMaterial({ map: canvasTex(64, 70, (c, w, h) => {
    c.fillStyle = "#f0f0e8"; c.fillRect(0, 0, w, h); c.fillStyle = "#555b66"; for (let k = 0; k < 7; k++) c.fillRect(6, 7 + k * 8, 26 + ((k * 23) % 28), 2);
  }), side: THREE.DoubleSide }));
  sheet.position.set(0, 0.15, -0.02); sheet.rotation.x = -0.15; carriage.add(sheet);
  return { group: g, carriage, sheet };
}
// An abacus: a frame, five rods and beads that the floor can slide. Returns the beads with their rest x.
export function makeAbacus(parent, x, y, z, rotY = 0) {
  const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = rotY; parent.add(g);
  const frame = mat(0x3a2519, { roughness: 0.6 }), rod = mat(0x8a8a80, { metalness: 0.5, roughness: 0.4 });
  for (const sx of [-0.17, 0.17]) box(0.02, 0.2, 0.02, frame, sx, 0.1, 0, g);
  for (const dy of [0.01, 0.19]) box(0.36, 0.02, 0.02, frame, 0, dy, 0, g);
  const beads = [];
  for (let r = 0; r < 5; r++) {
    const ry = 0.04 + r * 0.03;
    box(0.33, 0.005, 0.005, rod, 0, ry, 0, g);
    for (let k = 0; k < 5; k++) {
      const b = new THREE.Mesh(new THREE.SphereGeometry(0.0135, 8, 6), mat([0xc2453d, 0xe8a33d, 0x4f6a63][r % 3], { roughness: 0.4 })); b.scale.x = 0.7;
      const rest = -0.14 + k * 0.024; b.position.set(rest, ry, 0); g.add(b); beads.push({ b, rest, r, k });
    }
  }
  return beads;
}
// A desk phone, handset across the top.
export function makePhone(x, z, rotY) {
  const g = new THREE.Group(); g.position.set(x, 1.02, z); g.rotation.y = rotY; scene.add(g);
  const dark = mat(0x2a2e36, { roughness: 0.5 });
  const base = box(0.22, 0.07, 0.16, dark, 0, 0.035, 0, g); base.rotation.x = 0.15;
  for (let i = 0; i < 6; i++) box(0.03, 0.01, 0.025, mat(0xb9bcb5), 0.03 + (i % 3) * 0.04, 0.075, -0.02 + Math.floor(i / 3) * 0.04, g);
  box(0.05, 0.035, 0.2, dark, -0.07, 0.1, 0, g);
  for (const dz of [-0.085, 0.085]) box(0.06, 0.03, 0.05, dark, -0.07, 0.085, dz, g);
  return g;
}
// An office chair facing the camera.
export function makeChair(x, z) {
  const chair = new THREE.Group(); scene.add(chair);
  const fabric = mat(0x2f3644, { roughness: 0.95 }), steel = mat(0x8a9099, { metalness: 0.7, roughness: 0.35 }), black = mat(0x1c2027, { roughness: 0.6 });
  for (let i = 0; i < 5; i++) {
    const arm = new THREE.Group(); arm.rotation.y = i * Math.PI * 2 / 5; chair.add(arm);
    box(0.3, 0.035, 0.05, steel, 0.15, 0.09, 0, arm);
    const wheel = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), black);
    wheel.position.set(0.29, 0.035, 0); wheel.castShadow = true; arm.add(wheel);
  }
  const column = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.035, 0.36, 12), steel);
  column.position.y = 0.28; column.castShadow = true; chair.add(column);
  box(0.5, 0.06, 0.48, black, 0, 0.47, 0, chair);
  const seat = box(0.48, 0.08, 0.46, fabric, 0, 0.54, 0.01, chair); seat.rotation.x = 0.04;
  box(0.05, 0.36, 0.04, black, 0, 0.66, -0.25, chair);
  const back = new THREE.Group(); back.position.set(0, 0.98, -0.27); back.rotation.x = -0.14; chair.add(back);
  box(0.46, 0.52, 0.07, fabric, 0, 0, 0, back);
  box(0.4, 0.1, 0.05, fabric, 0, -0.14, 0.04, back);
  box(0.44, 0.06, 0.03, fabric, 0, 0.29, 0.01, back);
  for (const sx of [-0.27, 0.27]) {
    box(0.035, 0.2, 0.035, black, sx, 0.64, -0.02, chair);
    box(0.06, 0.03, 0.28, black, sx, 0.75, 0.02, chair);
    box(0.07, 0.025, 0.2, fabric, sx, 0.775, 0.04, chair);
  }
  chair.position.set(x, 0, z);
  return chair;
}
// An adviser. The frame loop poses them from walking, walkPhase, sit (0 standing, 1 seated), carry, reach, busy
// (typing at a desk), lift (extra height) and heading (the yaw they turn towards). `old` gives a stoop and a shuffle.
// Optional: cheer (seated, both arms up) and watch (a point whose x and z the head turns to).
export function makeAdviser({ x = 0, z = 0, suit = 0x56657d, hair = 0x3b322b, tie = 0xe8a33d, skin = 0xc9b39d, nodPhase = 0, old = false } = {}) {
  const g = new THREE.Group(); scene.add(g); g.position.set(x, 0, z);
  const suitM = mat(suit), trousers = mat(0x2b303a), shoe = mat(0x23272f), skinM = mat(skin), hairM = mat(hair), dark = mat(0x1a1f27);
  const cyl = (rt, rb, h, m, parent, y) => { const c = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, 14), m); c.position.y = y; c.castShadow = true; parent.add(c); return c; };
  const ball = (r, m, parent, px, py, pz) => { const s = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 10), m); s.position.set(px, py, pz); s.castShadow = true; parent.add(s); return s; };
  // The upper body pivots at the waist, so a lean carries the arms and head with it.
  const body = new THREE.Group(); body.position.y = 0.92; g.add(body);
  box(0.34, 0.16, 0.24, trousers, 0, 0.86, 0, g);
  const torso = cyl(0.19, 0.22, 0.66, suitM, body, 0.33);
  box(0.1, 0.3, 0.02, mat(0xe6e8e3), 0, 0.5, 0.21, body);
  if (tie !== null) box(0.05, 0.26, 0.02, mat(tie), 0, 0.45, 0.225, body);
  for (const sx of [-1, 1]) ball(0.075, suitM, body, sx * 0.22, 0.6, 0);
  cyl(0.055, 0.06, 0.12, skinM, body, 0.7);
  const head = new THREE.Group(); head.position.y = 0.76; body.add(head);
  ball(0.16, skinM, head, 0, 0.14, 0);
  const cap = new THREE.Mesh(new THREE.SphereGeometry(0.166, 20, 10, 0, Math.PI * 2, 0, Math.PI / (old ? 3.4 : 2.2)), hairM);
  cap.position.y = 0.15; head.add(cap);
  for (const sx of [-1, 1]) {
    ball(0.02, dark, head, sx * 0.058, 0.165, 0.14);
    ball(0.025, skinM, head, sx * 0.155, 0.13, 0);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.045, 0.007, 6, 16), dark); ring.position.set(sx * 0.062, 0.165, 0.162); head.add(ring);
    if (old) box(0.06, 0.02, 0.02, hairM, sx * 0.06, 0.218, 0.15, head);
  }
  box(0.03, 0.006, 0.006, dark, 0, 0.165, 0.162, head);
  if (old) box(0.1, 0.028, 0.03, hairM, 0, 0.085, 0.15, head);
  else box(0.05, 0.008, 0.01, mat(0x9a6e60), 0, 0.08, 0.152, head);
  // Arms: shoulder pivot, elbow pivot, hand. Legs: hip pivot, knee pivot, shoe.
  function arm(sx) {
    const sh = new THREE.Group(); sh.position.set(sx * 0.27, 0.6, 0); body.add(sh);
    cyl(0.055, 0.05, 0.3, suitM, sh, -0.15);
    const elbow = new THREE.Group(); elbow.position.y = -0.3; sh.add(elbow);
    cyl(0.05, 0.042, 0.28, suitM, elbow, -0.14);
    const hand = new THREE.Group(); hand.position.y = -0.3; elbow.add(hand);
    ball(0.045, skinM, hand, 0, 0, 0);
    sh.elbow = elbow; sh.hand = hand; return sh;
  }
  function leg(sx) {
    const hip = new THREE.Group(); hip.position.set(sx * 0.1, 0.86, 0); g.add(hip);
    cyl(0.075, 0.065, 0.44, trousers, hip, -0.22);
    const knee = new THREE.Group(); knee.position.y = -0.43; hip.add(knee);
    cyl(0.065, 0.06, 0.4, trousers, knee, -0.2);
    box(0.12, 0.07, 0.24, shoe, 0, -0.395, 0.05, knee);
    hip.knee = knee; return hip;
  }
  return {
    group: g, body, torso, head, legL: leg(-1), legR: leg(1), armL: arm(-1), armR: arm(1),
    walking: 0, walkAmt: 0, walkPhase: 0, sit: 0, carry: false, reach: false, busy: false, lift: 0, nodPhase, old, heading: 0,
  };
}
export const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp1 = (v, m) => Math.max(-m, Math.min(m, v));
export function poseAdviser(A, t, dt) {
  const g = A.group, k = reduced ? 1 : Math.min(1, dt * 10);
  A.walkAmt += (A.walking - A.walkAmt) * k;
  const turn = wrapAngle(A.heading - g.rotation.y);
  g.rotation.y += reduced ? turn : turn * Math.min(1, dt * 11);
  const w = A.walkAmt, s = A.sit, ph = A.walkPhase, blend = (a, b) => a + (b - a) * s;
  const swing = Math.sin(ph) * (A.old ? 0.36 : 0.55) * w;
  // Limbs hang along -y and the body faces +z, so a negative x rotation swings a limb forward.
  // Legs: thighs forward and shins down when seated; a knee bend on whichever leg is swinging.
  const seatHip = 1.35 * s;
  A.legL.rotation.x = -(swing + seatHip); A.legR.rotation.x = -(-swing + seatHip);
  A.legL.knee.rotation.x = seatHip + Math.max(0, Math.cos(ph)) * 0.9 * w;
  A.legR.knee.rotation.x = seatHip + Math.max(0, -Math.cos(ph)) * 0.9 * w;
  // Arms standing: a swing opposite the legs; the old hand rests in the small of the back.
  let aL = -swing * 0.8, aR = swing * 0.8, eL = 0.25 + 0.25 * w, eR = eL, zL = 0, zR = 0, ezL = 0, ezR = 0;
  if (A.old) { aL = -0.41; zL = 0.45; eL = 0.6; }
  if (A.carry) { aL = aR = 0.8; eL = eR = 1.15; zL = zR = 0; ezL = 0.25; ezR = -0.25; }
  else if (A.reach) { aR = 1.35; eR = 0.1; }
  const nod = s > 0.9 && !reduced ? Math.sin(t * 9 + A.nodPhase) : 0;
  // Arms seated: both at the keyboard when busy, else one on the desk and one at the chin.
  let sL = 1.15, sR = 1.25, seL = 0.45, seR = 2.1 + nod * 0.05, sezR = -0.7;
  if (A.busy) { sL = sR = 1.05; seL = 0.55 + Math.sin(t * 13 + A.nodPhase) * 0.07; seR = 0.55 + Math.cos(t * 13 + A.nodPhase) * 0.07; sezR = 0; }
  if (A.cheer) { sL = sR = 2.75; seL = seR = 0.3 + (reduced ? 0 : Math.sin(t * 20 + A.nodPhase) * 0.15); sezR = 0; }
  A.armL.rotation.set(-blend(aL, sL), 0, blend(zL, 0)); A.armL.elbow.rotation.set(-blend(eL, seL), 0, blend(ezL, 0));
  A.armR.rotation.set(-blend(aR, sR), 0, blend(zR, 0)); A.armR.elbow.rotation.set(-blend(eR, seR), 0, blend(ezR, sezR));
  const stoop = A.old ? 0.22 * (1 - 0.5 * s) : 0;
  A.body.rotation.x = 0.1 * w + Math.sin(s * Math.PI) * 0.4 + stoop;
  A.body.rotation.z = (reduced ? 0 : Math.sin(t * 0.8 + A.nodPhase) * 0.015) * (1 - w) + Math.sin(ph) * (A.old ? 0.07 : 0.035) * w - clamp1(turn, 0.5) * 0.3;
  A.head.rotation.x = nod * 0.12 - stoop * 0.8 - 0.05 * w;
  A.head.rotation.y = A.watch ? clamp1(wrapAngle(Math.atan2(A.watch.x - g.position.x, A.watch.z - g.position.z) - g.rotation.y), 1.3)
    : A.busy ? Math.sin(t * 0.45 + A.nodPhase) * 0.35 : reduced ? 0 : Math.sin(t * 0.35 + A.nodPhase) * 0.12 * (1 - w);
  const b = reduced ? 0 : Math.sin(t * 1.6 + A.nodPhase) * 0.012;
  A.torso.scale.set(1 + b, 1 + b * 0.5, 1 + b);
  g.position.y = w * (0.5 - 0.5 * Math.cos(2 * ph)) * 0.045 - 0.29 * s + A.lift;
}
// The company's van: cab at +x, cargo box behind, a roof rack. body carries the suspension's squat and dip; the wheels
// sit on the group and turn with its travel. The caller adds the group to its parent.
export function makeVan() {
  const van = new THREE.Group(), body = new THREE.Group(); van.add(body);
  const paint = mat(0xc9ccc4, { roughness: 0.55 }), glass = mat(0x1b2230, { roughness: 0.2, metalness: 0.4 }), trim = mat(0x2a2e36, { roughness: 0.6 });
  box(1.9, 1.0, 0.95, paint, -0.25, 0.78, 0, body);
  box(0.78, 0.42, 0.95, paint, 1.06, 0.49, 0, body);
  box(0.55, 0.5, 0.95, paint, 0.95, 0.95, 0, body);
  const screen = box(0.03, 0.42, 0.85, glass, 1.21, 0.93, 0, body); screen.rotation.z = 0.28;
  for (const sz of [-1, 1]) {
    box(0.42, 0.3, 0.02, glass, 0.98, 0.98, sz * 0.475, body);
    box(0.05, 0.1, 0.06, trim, 1.2, 0.98, sz * 0.52, body);
    box(0.02, 0.5, 0.02, trim, 0.4, 0.78, sz * 0.475, body);
    box(0.02, 0.5, 0.02, trim, -0.5, 0.78, sz * 0.475, body);
    box(0.9, 0.02, 0.02, trim, -0.05, 1.03, sz * 0.475, body);
    const label = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.26), new THREE.MeshStandardMaterial({ transparent: true, roughness: 0.6, map: canvasTex(512, 112, (g, w, h) => {
      g.fillStyle = "#56657d"; g.font = "700 72px 'Barlow Condensed', Arial Narrow, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText("MERIDIAN WORDS", w / 2, h / 2);
    }) }));
    label.position.set(-0.25, 0.88, sz * 0.478); label.rotation.y = sz > 0 ? 0 : Math.PI; body.add(label);
  }
  box(1.9, 0.12, 0.96, mat(0xe8a33d), -0.25, 0.62, 0, body);
  box(0.08, 0.14, 0.98, trim, 1.44, 0.3, 0, body);
  box(0.08, 0.14, 0.98, trim, -1.22, 0.3, 0, body);
  box(0.02, 0.18, 0.5, trim, 1.46, 0.52, 0, body);
  for (const rz of [-0.4, 0.4]) box(1.7, 0.04, 0.04, trim, -0.3, 1.32, rz, body);
  for (const rx of [-1.0, -0.3, 0.4]) box(0.04, 0.04, 0.86, trim, rx, 1.32, 0, body);
  const tailM = mat(0xd2412f, { emissive: 0xd2412f, emissiveIntensity: 0.4 });
  const beams = [];
  for (const hz of [-0.32, 0.32]) {
    box(0.03, 0.1, 0.16, mat(0xfff2c9, { emissive: 0xfff2c9, emissiveIntensity: 1 }), 1.46, 0.52, hz, body);
    box(0.03, 0.1, 0.1, tailM, -1.21, 0.62, hz, body);
    const beam = new THREE.Mesh(new THREE.ConeGeometry(0.5, 2.4, 14, 1, true), new THREE.MeshBasicMaterial({ color: 0xfff2c9, transparent: true, opacity: 0.1, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
    beam.position.set(2.66, 0.45, hz); beam.rotation.z = Math.PI / 2; beam.visible = false; body.add(beam); beams.push(beam);
  }
  const headLight = new THREE.SpotLight(0xfff2c9, 1.2, 7, 0.55, 0.6, 1.2);
  headLight.position.set(1.4, 0.55, 0); headLight.target.position.set(6, -0.5, 0); body.add(headLight, headLight.target);
  headLight.visible = false;
  const wheels = [], hub = mat(0x8a9099, { metalness: 0.6, roughness: 0.4 });
  for (const [wx, wz] of [[-0.75, -0.48], [-0.75, 0.48], [0.95, -0.48], [0.95, 0.48]]) {
    const p = new THREE.Group(); p.position.set(wx, 0.22, wz); van.add(p); wheels.push(p);
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.14, 16), mat(0x1c2027)); w.rotation.x = Math.PI / 2; w.castShadow = true; p.add(w);
    const h = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.15, 12), hub); h.rotation.x = Math.PI / 2; p.add(h);
    box(0.17, 0.03, 0.16, trim, 0, 0, 0, p);
  }
  return {
    group: van, body, wheels,
    lights(on) { headLight.visible = on; for (const b of beams) b.visible = on; },
    brake(on) { tailM.emissiveIntensity = on ? 1.6 : 0.4; },
  };
}
export function makeEnvelope() {
  const g = new THREE.Group(); scene.add(g);
  const paper = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.012, 0.28), paperM);
  paper.castShadow = true; g.add(paper);
  const flap = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.004, 0.02), mat(0xb9bdb6)); flap.position.set(0, 0.008, -0.06); g.add(flap);
  for (const [w, z] of [[0.16, 0.02], [0.2, 0.05], [0.12, 0.08]]) box(w, 0.002, 0.01, mat(0x6d7486), -0.08, 0.008, z, g);
  const stamp = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.006, 24), mat(0xe8a33d, { emissive: 0x7a4a10, emissiveIntensity: 0.5 }));
  stamp.position.set(0.12, 0.01, 0.05); stamp.visible = false; g.add(stamp);
  return { group: g, stamp };
}

// A fax machine: sloped keypad, handset on the right, a sheet waiting in the out tray.
export function makeFax(x, y, z, rotY) {
  const g = new THREE.Group(); g.position.set(x, y, z); g.rotation.y = rotY; scene.add(g);
  const shell = mat(0xb9bcb5, { roughness: 0.5 }), dark = mat(0x2a2e36, { roughness: 0.5 });
  box(0.44, 0.12, 0.34, shell, 0, 0.06, 0, g);
  const panel = box(0.3, 0.07, 0.2, shell, -0.05, 0.145, 0.03, g); panel.rotation.x = 0.22;
  for (let i = 0; i < 12; i++) {
    const key = box(0.035, 0.012, 0.028, mat(i === 11 ? 0xe8a33d : 0x3a3f48), -0.13 + (i % 4) * 0.05, 0.19 - Math.floor(i / 4) * 0.012, 0.1 - Math.floor(i / 4) * 0.05, g);
    key.rotation.x = 0.22;
  }
  box(0.3, 0.025, 0.03, dark, -0.05, 0.13, -0.13, g);
  const tray = box(0.3, 0.01, 0.2, paperM, -0.05, 0.22, -0.2, g); tray.rotation.x = 0.5;
  box(0.06, 0.03, 0.26, dark, 0.17, 0.15, 0, g);
  for (const dz of [-0.11, 0.11]) box(0.07, 0.035, 0.06, dark, 0.17, 0.135, dz, g);
  box(0.04, 0.02, 0.02, mat(0x4fb36b, { emissive: 0x4fb36b, emissiveIntensity: 0.8 }), 0.16, 0.12, 0.16, g);
  return g;
}

// A fax page hanging from a clip on the line. sheet ripples through geo.
export function makeFaxPage() {
  const group = new THREE.Group(); group.visible = false; scene.add(group);
  box(0.08, 0.07, 0.04, mat(0xe8a33d, { emissive: 0xe8a33d, emissiveIntensity: 0.8 }), 0, 0, 0, group);
  const geo = new THREE.PlaneGeometry(0.5, 0.68, 6, 8);
  const pageTex = canvasTex(128, 176, (g, w, h) => {
    g.fillStyle = "#f2f2ec"; g.fillRect(0, 0, w, h);
    g.fillStyle = "#1a1f27"; g.font = "700 22px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText("FAX", 14, 30);
    g.fillStyle = "#6d7486";
    for (let i = 0; i < 9; i++) g.fillRect(14, 50 + i * 13, 40 + ((i * 37) % 60), 4);
  });
  const sheet = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: pageTex, side: THREE.DoubleSide, emissive: 0xe8a33d, emissiveIntensity: 0.25 }));
  sheet.position.y = -0.3; group.add(sheet);
  return { group, sheet, geo };
}

// Walks an adviser in a straight line, turning to face the way they go. Older advisers take shorter steps.
export async function walkTo(A, x, z, seconds) {
  const g = A.group, x0 = g.position.x, z0 = g.position.z, dist = Math.hypot(x - x0, z - z0);
  if (dist < 0.01) return;
  A.heading = Math.atan2(x - x0, z - z0);
  A.walking = 1;
  let last = 0;
  await tween(seconds, (u) => {
    g.position.x = lerp(x0, x, u); g.position.z = lerp(z0, z, u);
    A.walkPhase += (u - last) * dist * (A.old ? 11 : 7.5); last = u;
  });
  A.walking = 0;
}
// Sits into the chair behind, thinks while the clock counts `think` seconds after `moved`, stands up again.
export async function sitAndThink(A, moved, think, z0 = 0.45) {
  A.heading = 0;
  await tween(0.45, (u) => { A.sit = ease(u); A.group.position.z = z0 - 0.25 * A.sit; });
  view.thinker = A;
  $("aThink").classList.add("on");
  await tween(think, (u) => setClock(moved, think * u));
  $("aThink").classList.remove("on");
  await tween(0.4, (u) => { A.sit = 1 - ease(u); A.group.position.z = z0 - 0.25 * A.sit; });
  A.sit = 0;
}
