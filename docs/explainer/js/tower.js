// The building every floor belongs to: its surfaces, the city outside at a floor's height, the outer walls, and the
// lift doors. Floors stand at y = 0 and the street lies below them, at floorDrop(n).
import { box, canvasTex, halo, mat, pool, rep, tile } from "./world.js";

// How far below floor n the street is. The city outside falls faster than the storeys do, so one floor shows.
export const floorDrop = (n) => 2.5 * n;
const cityDrop = (n) => (n ? 5 + 3 * n : 2.5);

export const carpetTex = tile((g) => {
  for (let i = 0; i < 4; i++) { g.fillStyle = i === 1 || i === 2 ? "#2a3240" : "#2f3846"; g.fillRect((i % 2) * 128, (i >> 1) * 128, 128, 128); }
  for (let k = 0; k < 1600; k++) { g.fillStyle = k % 3 ? "rgba(70,84,104,0.5)" : "rgba(14,18,24,0.5)"; g.fillRect((k * 53) % 256, (k * 97 + (k >> 3)) % 256, 2, 2); }
  g.fillStyle = "rgba(10,13,18,0.55)"; g.fillRect(0, 0, 256, 2); g.fillRect(0, 128, 256, 2); g.fillRect(0, 0, 2, 256); g.fillRect(128, 0, 2, 256);
});
export const boardsTex = tile((g) => {
  for (let r = 0; r < 5; r++) {
    const y = Math.round(r * 51.2);
    g.fillStyle = ["#4c3828", "#563f2c", "#45332a", "#513b29", "#4a3729"][r]; g.fillRect(0, y, 256, 52);
    g.strokeStyle = "rgba(255,226,190,0.05)"; g.lineWidth = 2;
    for (let k = 0; k < 4; k++) { g.beginPath(); g.moveTo(0, y + 10 + k * 10); g.bezierCurveTo(80, y + 5 + k * 11, 170, y + 15 + k * 9, 256, y + 10 + k * 10); g.stroke(); }
    g.fillStyle = "rgba(18,11,7,0.7)"; g.fillRect(0, y, 256, 2); g.fillRect((r * 89 + 40) % 256, y, 2, 52);
  }
});
export const brickTex = tile((g) => {
  g.fillStyle = "#33231e"; g.fillRect(0, 0, 256, 256);
  for (let r = 0; r < 12; r++) for (let c = -1; c < 4; c++) {
    g.fillStyle = ["#5e3d33", "#664236", "#563a30", "#6b4639"][(r * 3 + c * 5 + 8) % 4];
    g.fillRect(c * 64 + (r % 2) * 32 + 2, Math.round(r * 21.33) + 2, 60, 18);
  }
});
export const slabTex = tile((g) => {
  g.fillStyle = "#3a3f47"; g.fillRect(0, 0, 256, 256);
  g.fillStyle = "#30353c"; g.fillRect(0, 0, 256, 3); g.fillRect(0, 0, 3, 256);
  for (let k = 0; k < 500; k++) { g.fillStyle = k % 2 ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.08)"; g.fillRect((k * 71) % 256, (k * 113) % 256, 3, 3); }
});
export const asphaltTex = tile((g) => {
  g.fillStyle = "#1a1d23"; g.fillRect(0, 0, 256, 256);
  for (let k = 0; k < 1400; k++) { g.fillStyle = k % 3 ? "rgba(255,255,255,0.025)" : "rgba(0,0,0,0.2)"; g.fillRect((k * 83) % 256, (k * 131 + (k >> 2)) % 256, 2, 2); }
});
// A suspended ceiling: pale tiles on a dark grid. One repeat is one tile.
export const ceilingTex = tile((g) => {
  g.fillStyle = "#59606b"; g.fillRect(0, 0, 256, 256);
  for (let k = 0; k < 900; k++) { g.fillStyle = "rgba(20,24,30,0.28)"; g.fillRect((k * 61) % 256, (k * 149 + (k >> 3)) % 256, 2, 2); }
  g.fillStyle = "#2c3038"; g.fillRect(0, 0, 256, 7); g.fillRect(0, 0, 7, 256);
});
export const plaster = mat(0x3e495a), trimM = mat(0x2a313d), skirtM = mat(0x1d222b);

// A pseudo-random number in 0..1 for an integer k, so the skyline comes out the same each time.
const rnd = (k) => { const s = Math.sin(k * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };

// Lit windows on a repeating grid. One repeat is 4 by 4 windows.
function windowTile(seed, lit, warm = "#d8b46a", cool = "#9fb5c9") {
  const t = canvasTex(128, 128, (g) => {
    g.fillStyle = "#0d1118"; g.fillRect(0, 0, 128, 128);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const k = seed + r * 4 + c, x = c * 32 + 7, y = r * 32 + 8;
      g.fillStyle = "#171c25"; g.fillRect(x, y, 18, 16);
      if (rnd(k) < lit) { g.fillStyle = rnd(k * 3) < 0.7 ? warm : cool; g.globalAlpha = 0.45 + rnd(k * 7) * 0.55; g.fillRect(x, y, 18, 16); g.globalAlpha = 1; }
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping; return t;
}
const cityTiles = [windowTile(3, 0.34), windowTile(40, 0.22), windowTile(90, 0.46)];
// Scales the uvs of one face of a box geometry (0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z), so a tile keeps its size.
function faceRepeat(geo, face, rx, ry) {
  const uv = geo.attributes.uv;
  for (let i = face * 4; i < face * 4 + 4; i++) uv.setXY(i, uv.getX(i) * rx, uv.getY(i) * ry);
}
const roofM = mat(0x161a21), sideM = mat(0x12151b);
// A building with lit windows on its front. base and top are heights.
export function block(parent, x, z, w, d, base, top, tileIndex = 0, tint = 0xffffff) {
  const geo = new THREE.BoxGeometry(w, top - base, d);
  faceRepeat(geo, 4, w / 4.2, (top - base) / 4.4);
  const face = new THREE.MeshBasicMaterial({ map: cityTiles[tileIndex % 3], color: tint });
  const b = new THREE.Mesh(geo, [sideM, sideM, roofM, roofM, face, sideM]);
  b.position.set(x, (base + top) / 2, z); parent.add(b); return b;
}
const skyTex = canvasTex(32, 512, (g, w, h) => {
  const l = g.createLinearGradient(0, 0, 0, h);
  l.addColorStop(0, "#090c14"); l.addColorStop(0.45, "#111826"); l.addColorStop(0.72, "#27303f"); l.addColorStop(0.86, "#4b4238"); l.addColorStop(1, "#1a1a1e");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
});
const starTex = canvasTex(1024, 512, (g, w, h) => {
  for (let k = 0; k < 170; k++) { g.fillStyle = `rgba(226,232,244,${0.2 + rnd(k) * 0.6})`; g.fillRect(rnd(k * 3) * w, rnd(k * 5) * h, 1 + (k % 9 === 0), 1 + (k % 9 === 0)); }
});

// The city outside floor n, around x = cx: the sky, three rows of buildings, roof tanks and blinking masts. The same
// skyline on every floor, set lower the higher the floor. Returns its update(t), which blinks the masts.
export function city(parent, cx, n, { near = -18 } = {}) {
  const G = new THREE.Group(); G.position.set(cx, -cityDrop(n), 0); parent.add(G);
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(420, 150), new THREE.MeshBasicMaterial({ map: skyTex, fog: false }));
  sky.position.set(0, 58 + cityDrop(n) * 0.45, near - 62); G.add(sky);
  const stars = new THREE.Mesh(new THREE.PlaneGeometry(420, 70), new THREE.MeshBasicMaterial({ map: starTex, transparent: true, fog: false, blending: THREE.AdditiveBlending, depthWrite: false }));
  stars.position.set(0, 84 + cityDrop(n) * 0.45, near - 61); G.add(stars);
  // Above the painted sky, for a camera looking up.
  const night = new THREE.Mesh(new THREE.PlaneGeometry(900, 500), new THREE.MeshBasicMaterial({ color: new THREE.Color(0x090c14).convertSRGBToLinear(), fog: false }));
  night.position.set(0, 58 + cityDrop(n) * 0.45 + 75 + 250, near - 62); G.add(night);
  const moon = new THREE.Mesh(new THREE.CircleGeometry(2.3, 32), new THREE.MeshBasicMaterial({ color: 0xdfe3e6, fog: false }));
  moon.position.set(-16, 34 + cityDrop(n) * 0.45, near - 60); G.add(moon);
  const glow = halo(0xbcc8d8, 15, -16, 34 + cityDrop(n) * 0.45, near - 60.5, G, 0.32); glow.material.fog = false;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(320, 110), new THREE.MeshBasicMaterial({ color: 0x0c0f14 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(0, 0, near - 25); G.add(ground);
  const masts = [];
  [[near, 3, 12, 7], [near - 15, 7, 21, 8.5], [near - 33, 12, 34, 10]].forEach(([z, lo, hi, step], row) => {
    let i = 0;
    for (let x = -75; x < 75; i++) {
      const k = row * 100 + i, w = step * (0.55 + rnd(k) * 0.55), top = lo + rnd(k * 2) * (hi - lo);
      block(G, x + w / 2, z, w, 7, 0, top, k, row === 0 ? 0xffffff : 0xb9c0cc);
      if (row < 2 && rnd(k * 5) < 0.3) {
        const tank = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.55, 1.2, 10), roofM);
        tank.position.set(x + w / 2 - 0.6, top + 0.95, z + 1); G.add(tank);
        const cap = new THREE.Mesh(new THREE.ConeGeometry(0.65, 0.5, 10), roofM);
        cap.position.set(x + w / 2 - 0.6, top + 1.8, z + 1); G.add(cap);
      } else if (rnd(k * 7) < 0.3) {
        const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.07, 3, 6), roofM);
        mast.position.set(x + w / 2, top + 1.5, z); G.add(mast);
        const s = halo(0xff4a3a, 1.1, x + w / 2, top + 3.05, z, G, 0.8); s.material.fog = false;
        masts.push({ s, ph: k });
      }
      x += w + rnd(k * 3) * step * 0.35;
    }
  });
  return (t) => { for (const m of masts) m.s.material.opacity = ((t + m.ph * 0.37) % 2.6) < 1.1 ? 0.85 : 0.08; };
}

// Headlights and tail lights crawling along a street below. Returns its update(dt).
export function traffic(parent, y, z, x0, x1, count = 7) {
  const cars = Array.from({ length: count }, (_, i) => {
    const way = i % 2 ? 1 : -1;
    const s = halo(way > 0 ? 0xfff0c8 : 0xff5a44, 0.8, x0 + rnd(i * 11) * (x1 - x0), y + 0.4, z + way * 0.9, parent, 0.9);
    return { s, way, v: 2.4 + rnd(i * 13) * 2 };
  });
  return (dt) => { for (const c of cars) { let x = c.s.position.x + c.way * c.v * dt; if (x > x1) x = x0; if (x < x0) x = x1; c.s.position.x = x; } };
}

// A painted view for a small window, seen from a height: dusk sky, then roofs lower down the higher the floor.
export function windowView(seed, n) {
  return canvasTex(256, 192, (g, w, h) => {
    const l = g.createLinearGradient(0, 0, 0, h);
    l.addColorStop(0, "#10172a"); l.addColorStop(0.65, "#3d3a3e"); l.addColorStop(1, "#6a5240");
    g.fillStyle = l; g.fillRect(0, 0, w, h);
    for (let row = 0; row < 2; row++) {
      let x = -10, i = 0;
      while (x < w) {
        const k = seed + row * 50 + i++, bw = 22 + rnd(k) * 42, top = h * (0.2 + row * 0.26 + n * 0.09) + rnd(k * 3) * 50;
        g.fillStyle = row ? "#0e1219" : "#1a2130"; g.fillRect(x, top, bw, h - top);
        for (let wy = top + 6; wy < h; wy += 12) for (let wx = x + 5; wx < x + bw - 6; wx += 10) {
          if (rnd(wx * 3 + wy * 7 + k) < (row ? 0.34 : 0.2)) { g.fillStyle = rnd(wx + wy) < 0.7 ? "#d8b46a" : "#9fb5c9"; g.fillRect(wx, wy, 5, 6); }
        }
        x += bw + 3;
      }
    }
  });
}

// The tower's outer wall: glass panes in a dark frame, some rooms lit, in runs as an office's lights are. One repeat is
// 12 by 12 panes.
const facadeTex = (() => {
  const t = canvasTex(768, 768, (g) => {
    g.fillStyle = "#14181f"; g.fillRect(0, 0, 768, 768);
    for (let r = 0; r < 12; r++) for (let c = 0; c < 12; c++) {
      const k = r * 12 + c + 7, x = c * 64 + 6, y = r * 64 + 8;
      const l = g.createLinearGradient(0, y, 0, y + 48);
      if (rnd(Math.floor(c / 3) * 13 + r * 29) + rnd(k) * 0.5 > 0.92) { l.addColorStop(0, "#e4c98c"); l.addColorStop(1, "#8f7a4e"); } else { l.addColorStop(0, "#232b38"); l.addColorStop(1, "#171c25"); }
      g.fillStyle = l; g.fillRect(x, y, 52, 48);
      g.fillStyle = "#0c0f14"; g.fillRect(x + 25, y, 2, 48);
    }
    g.fillStyle = "#3a414c"; for (let r = 0; r < 12; r++) g.fillRect(0, r * 64 + 58, 768, 4);
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping; return t;
})();
// A flat run of the tower's outer wall, facing +z, from y0 to y1. One storey is storey units tall.
export function curtainWall(parent, x, z, w, y0, y1, storey = 2.5, tint = 0xc4c8cf) {
  const t = rep(facadeTex, w / (storey * 3.15), (y1 - y0) / (storey * 12));
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, y1 - y0), new THREE.MeshBasicMaterial({ map: t, color: tint }));
  m.position.set(x, (y0 + y1) / 2, z); parent.add(m); return m;
}

// ---------- Lift doors ----------
// Sizes in units. A door plane faces +z; the car stands behind it. CAR is shared with lift.js.
export const CAR = { doorW: 1.3, doorH: 2.25, w: 2.0, d: 1.9, h: 2.6, eye: 1.62 };
const steelCanvas = (w, h, base, streak) => canvasTex(w, h, (g) => {
  g.fillStyle = base; g.fillRect(0, 0, w, h);
  for (let k = 0; k < w * 3; k++) { g.fillStyle = k % 2 ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.07)"; g.fillRect((k * 37) % w, 0, 1 + (k % 3 === 0), h); }
  const l = g.createLinearGradient(0, 0, 0, h); l.addColorStop(0, streak); l.addColorStop(1, "rgba(0,0,0,0.25)");
  g.fillStyle = l; g.fillRect(0, 0, w, h);
});
const leafOut = mat(0xffffff, { map: steelCanvas(128, 256, "#8d939b", "rgba(255,255,255,0.08)"), metalness: 0.45, roughness: 0.42 });
export const leafIn = new THREE.MeshBasicMaterial({ map: steelCanvas(128, 256, "#a7a59d", "rgba(255,236,200,0.2)"), fog: false });
export const brass = mat(0xb08a4a, { metalness: 0.7, roughness: 0.35 });
// What the small door windows show from inside the car while it moves: shaft wall, a beam and one lit landing per repeat.
export const shaftTex = canvasTex(64, 256, (g, w, h) => {
  g.fillStyle = "#14161a"; g.fillRect(0, 0, w, h);
  for (let k = 0; k < 400; k++) { g.fillStyle = "rgba(255,255,255,0.03)"; g.fillRect((k * 31) % w, (k * 79) % h, 2, 2); }
  g.fillStyle = "#2a2e35"; g.fillRect(0, 60, w, 14);
  g.fillStyle = "#3a3f47"; g.fillRect(14, 0, 3, h); g.fillRect(46, 0, 3, h);
  const l = g.createLinearGradient(0, 160, 0, 230); l.addColorStop(0, "#1a1a1a"); l.addColorStop(0.5, "#f0d9a4"); l.addColorStop(1, "#1a1a1a");
  g.fillStyle = l; g.fillRect(0, 170, w, 50);
});
shaftTex.wrapT = THREE.RepeatWrapping;
const shaftM = new THREE.MeshBasicMaterial({ map: shaftTex, fog: false });
const glassOut = mat(0x10141b, { metalness: 0.3, roughness: 0.15 });

// The floor marks on a lift's dial: G and the open floors are spaced wide, the closed floors run up to 48 in what is left.
export const dialAngle = (level) => {
  const stops = [-1.32, -0.96, -0.6, -0.24, 0.12], last = stops.length - 1;
  if (level <= last) { const i = Math.floor(level), f = level - i; return i >= last ? stops[last] : stops[i] + (stops[i + 1] - stops[i]) * f; }
  return 0.12 + (Math.min(level, 48) - last) / (48 - last) * 1.2;
};
const dialFace = canvasTex(256, 128, (g, w, h) => {
  g.fillStyle = "#b39055"; g.beginPath(); g.moveTo(10, h); g.arc(w / 2, h, 118, Math.PI, 0); g.closePath(); g.fill();
  g.fillStyle = "#1a1d22"; g.beginPath(); g.moveTo(20, h); g.arc(w / 2, h, 108, Math.PI, 0); g.closePath(); g.fill();
  g.textAlign = "center"; g.textBaseline = "middle"; g.font = "600 22px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillStyle = "#e9e2cf";
  [["G", 0], ["1", 1], ["2", 2], ["3", 3], ["4", 4], ["48", 48]].forEach(([s, lv]) => {
    const a = dialAngle(lv); g.fillText(s, w / 2 + Math.sin(a) * 88, h - Math.cos(a) * 88);
  });
  g.fillStyle = "rgba(233,226,207,0.5)";
  for (let lv = 8; lv < 48; lv += 8) { const a = dialAngle(lv); g.fillRect(w / 2 + Math.sin(a) * 88 - 1.5, h - Math.cos(a) * 88 - 1.5, 3, 3); }
});
export const dialMat = () => new THREE.MeshBasicMaterial({ map: dialFace, transparent: true, fog: false });
const signCanvas = (number, name) => canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = "#e9ebe6"; g.textAlign = "center"; g.textBaseline = "middle";
  g.font = "700 150px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText(String(number).padStart(2, "0"), w / 2, 88);
  g.fillStyle = "#e8a33d"; g.fillRect(36, 170, w - 72, 5);
  g.fillStyle = "#c9ced6"; g.font = "600 36px 'Barlow Condensed', Arial Narrow, sans-serif"; g.fillText(name.toUpperCase(), w / 2, 212);
});

// Lift doors in a wall at (x, y, z) of parent, facing +z. The wall needs a hole for them: wallWithDoor cuts one.
// open(u) slides the leaves (0 shut, 1 open) behind the wall, spilling the car's light onto the floor in front.
export function makeLift(parent, { x, y = 0, z, number, name, signSide = 1, signAt = null, showDial = true }) {
  const g = new THREE.Group(); g.position.set(x, y, z); parent.add(g);
  const W = CAR.doorW, H = CAR.doorH;
  const leaves = [-1, 1].map((s) => {
    const leaf = new THREE.Group(); g.add(leaf);
    box(W / 2 + 0.01, H, 0.04, leafOut, s * W / 4, H / 2, -0.05, leaf);
    const inner = new THREE.Mesh(new THREE.PlaneGeometry(W / 2 + 0.01, H), leafIn); inner.rotation.y = Math.PI; inner.position.set(s * W / 4, H / 2, -0.072); leaf.add(inner);
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 0.62), shaftM); pane.rotation.y = Math.PI; pane.position.set(s * W / 4, 1.5, -0.073); leaf.add(pane);
    box(0.22, 0.62, 0.006, glassOut, s * W / 4, 1.5, -0.028, leaf);
    for (const dy of [-0.32, 0.32]) box(0.26, 0.02, 0.008, brass, s * W / 4, 1.5 + dy, -0.026, leaf);
    box(0.012, H, 0.008, brass, s * 0.006, H / 2, -0.026, leaf);
    return { leaf, s };
  });
  for (const s of [-1, 1]) box(0.12, H + 0.14, 0.12, brass, s * (W / 2 + 0.06), (H + 0.14) / 2, 0.02, g);
  box(W + 0.24, 0.14, 0.12, brass, 0, H + 0.07, 0.02, g);
  box(W + 0.1, 0.012, 0.14, brass, 0, 0.006, 0.05, g);
  const spill = pool(g, 0xffd9a0, 1.5, 0, 0.03, 1.0, 0); spill.scale.set(0.75, 1.1, 1);
  let arrow = null;
  if (showDial) {
    const face = new THREE.Mesh(new THREE.PlaneGeometry(0.68, 0.34), dialMat()); face.position.set(0, H + 0.34, 0.04); g.add(face);
    arrow = new THREE.Group(); arrow.position.set(0, H + 0.175, 0.05); g.add(arrow);
    const needle = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.26, 0.006), new THREE.MeshBasicMaterial({ color: 0xe8a33d, fog: false })); needle.position.y = 0.13; arrow.add(needle);
    arrow.rotation.z = -dialAngle(number);
  }
  // The call plate: one lit button.
  box(0.1, 0.22, 0.015, brass, W / 2 + 0.3, 1.25, 0.012, g);
  const callM = new THREE.MeshStandardMaterial({ color: 0x2a2520, emissive: 0xffb45c, emissiveIntensity: 0.15 });
  const btn = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.015, 14), callM); btn.rotation.x = Math.PI / 2; btn.position.set(W / 2 + 0.3, 1.25, 0.025); g.add(btn);
  if (signSide || signAt) {
    const s = new THREE.Mesh(new THREE.PlaneGeometry(0.72, 0.72), new THREE.MeshBasicMaterial({ map: signCanvas(number, name), transparent: true }));
    if (signAt) s.position.set(signAt[0], signAt[1], 0.015); else s.position.set(signSide * (W / 2 + 0.72), 2.05, 0.015);
    if (signAt && signAt[2]) s.scale.setScalar(signAt[2]);
    g.add(s);
  }
  return {
    group: g, number, u: 0,
    open(u) { this.u = u; for (const { leaf, s } of leaves) leaf.position.x = s * u * (W / 2 - 0.03); spill.material.opacity = u * 0.34; },
    call(on) { callM.emissiveIntensity = on ? 1.6 : 0.15; },
    show(level) { if (arrow) arrow.rotation.z = -dialAngle(level); },
    world() { return g.getWorldPosition(new THREE.Vector3()); },
  };
}

// The old building's goods hoist: a pallet on a chain from a trolley at height top, at (x, z) of parent. The caller
// builds the beam it hangs from. set(y) moves the pallet.
export function makeHoist(parent, x, z, top) {
  const steel = mat(0x6d737c, { metalness: 0.7, roughness: 0.4 }), planks = mat(0x7a5c3c, { roughness: 0.9 });
  box(0.22, 0.14, 0.26, mat(0xe8a33d, { roughness: 0.5 }), x, top - 0.05, z, parent).castShadow = false;
  const pallet = new THREE.Group(); parent.add(pallet);
  for (let k = 0; k < 5; k++) box(1.2, 0.04, 0.09, planks, 0, 0, -0.22 + k * 0.11, pallet);
  const chain = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), steel); parent.add(chain);
  for (const dx of [-0.5, 0.5]) { const sling = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.9, 5), steel); sling.position.set(dx * 0.5, 0.45, 0); sling.rotation.z = dx * 0.75; pallet.add(sling); }
  return {
    pallet,
    set(y) {
      pallet.position.set(x, y, z);
      const t = top - 0.1, b = y + 0.72;
      chain.position.set(x, (t + b) / 2, z); chain.scale.y = Math.max(0.01, t - b);
    },
  };
}

// A side wall from z0 to z1, cut away towards the front so a room stays open to a camera above it.
export function cutawayWall(parent, x, m, z0, z1, back, front) {
  const sh = new THREE.Shape();
  sh.moveTo(z0, 0); sh.lineTo(z1, 0); sh.lineTo(z1, front); sh.lineTo(z0, back); sh.closePath();
  const w = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, { depth: 0.12, bevelEnabled: false }), m);
  // The shape's x runs along z, back to front; the extrusion is the wall's thickness, along -x.
  w.rotation.y = -Math.PI / 2; w.position.set(x + 0.06, 0, 0); w.castShadow = w.receiveShadow = true;
  parent.add(w); return w;
}

// A wall from x0 to x1, h tall, its front face at z, with a hole for lift doors at doorX. doorY is how far above the
// wall's foot the doors stand.
export function wallWithDoor(parent, m, x0, x1, h, z, doorX, thick = 0.12, doorY = 0) {
  const W = CAR.doorW + 0.02, H = CAR.doorH + 0.02 + doorY, zc = z - thick / 2;
  box(doorX - W / 2 - x0, h, thick, m, (x0 + doorX - W / 2) / 2, h / 2, zc, parent);
  box(x1 - doorX - W / 2, h, thick, m, (x1 + doorX + W / 2) / 2, h / 2, zc, parent);
  box(W, h - H, thick, m, doorX, (h + H) / 2, zc, parent);
}
