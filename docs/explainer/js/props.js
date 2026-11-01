// Furniture and people, used by more than one floor, and how an adviser walks and thinks.
import { $, box, canvasTex, halo, mat, reduced, scene, view } from "./world.js";
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
// A desk phone, handset across the top.
export function makePhone(x, z, rotY) {
  const g = new THREE.Group(); g.position.set(x, 1.02, z); g.rotation.y = rotY; scene.add(g);
  const dark = mat(0x2a2e36, { roughness: 0.5 });
  const base = box(0.22, 0.07, 0.16, dark, 0, 0.035, 0, g); base.rotation.x = 0.15;
  for (let i = 0; i < 6; i++) box(0.03, 0.01, 0.025, mat(0xb9bcb5), 0.03 + (i % 3) * 0.04, 0.075, -0.02 + Math.floor(i / 3) * 0.04, g);
  box(0.05, 0.035, 0.2, dark, -0.07, 0.1, 0, g);
  for (const dz of [-0.085, 0.085]) box(0.06, 0.03, 0.05, dark, -0.07, 0.085, dz, g);
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
