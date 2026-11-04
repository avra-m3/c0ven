// The 3D world every floor shares: renderer, scene, camera and the smallest building blocks.

export const $ = (id) => document.getElementById(id);
export const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

// What changes as the game runs and more than one file reads. floor 0 is the street and lobby. thinker: who the
// "thinking" label follows. rig: while set, the lift's camera instead of the floor's (see lift.js).
export const view = { floor: 0, shake: 0, baseFov: 42, thinker: null, rig: null };

export const stage = $("stage");
export const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
stage.appendChild(renderer.domElement);

export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x12161d);
scene.fog = new THREE.Fog(0x12161d, 11, 24);
export const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 420);

export const mat = (color, o = {}) => new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.85, metalness: 0.02 }, o));
export function box(w, h, d, m, x, y, z, parent = scene) {
  const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  b.position.set(x, y, z); b.castShadow = true; b.receiveShadow = true;
  parent.add(b); return b;
}
export function canvasTex(w, h, draw) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c); t.encoding = THREE.sRGBEncoding; return t;
}
export function sign(text, w, h, x, y, z, bg = "#e9ebe6", fg = "#1a1f27", parent = scene) {
  const tex = canvasTex(512, Math.round(512 * h / w), (g, cw, ch) => {
    g.fillStyle = bg; g.fillRect(0, 0, cw, ch);
    g.fillStyle = fg; g.font = `700 ${Math.round(ch * 0.62)}px 'Barlow Condensed', Arial Narrow, sans-serif`;
    g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, cw / 2, ch * 0.54);
  });
  const s = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 }));
  s.position.set(x, y, z); parent.add(s); return s;
}
scene.add(new THREE.HemisphereLight(0x9fb2c8, 0x1a1f27, 0.55));
// A cool rim from behind, so figures separate from the walls.
const rim = new THREE.DirectionalLight(0x8bb6cf, 0.3); rim.position.set(-4, 6, -8); scene.add(rim);
// Soft glow discs for lamps, so lights read as lit at the camera's distance.
export const haloTex = canvasTex(64, 64, (g) => {
  const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.3, "rgba(255,255,255,0.4)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
});
export function halo(color, size, x, y, z, parent = scene, opacity = 0.8) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  s.scale.setScalar(size); s.position.set(x, y, z); parent.add(s); return s;
}
// Light that a lamp throws on a surface: an additive disc, so no real light is needed.
const poolTex = canvasTex(128, 128, (g) => {
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(255,255,255,0.85)"); r.addColorStop(0.45, "rgba(255,255,255,0.35)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
});
export function pool(parent, color, r, x, y, z, opacity) {
  const p = new THREE.Mesh(new THREE.PlaneGeometry(r * 2, r * 2), new THREE.MeshBasicMaterial({ map: poolTex, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  p.rotation.x = -Math.PI / 2; p.position.set(x, y, z); parent.add(p); return p;
}
// Tiled surfaces, 256 pixels to a unit; rep() gives a surface its own repeat.
export function tile(draw) { const t = canvasTex(256, 256, draw); t.wrapS = t.wrapT = THREE.RepeatWrapping; return t; }
export const rep = (t, x, y) => { const c = t.clone(); c.needsUpdate = true; c.repeat.set(x, y); return c; };

// A floor's objects sit under one group, so the floors the player is not on can be hidden: every visible light costs
// every material, and each floor has a shadow pass. A floor's module calls openFloor first and closeFloor last.
const floorRoots = {};
export const openFloor = () => scene.children.length;
export function closeFloor(n, from) {
  const g = new THREE.Group();
  g.add(...scene.children.slice(from));
  scene.add(g); floorRoots[n] = g; g.visible = n === view.floor;
}
export function showFloor(n) { for (const [k, g] of Object.entries(floorRoots)) g.visible = +k === n; }

export const tubes = [];
export function tube(x, y, z) {
  const t = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.05, 0.18), new THREE.MeshBasicMaterial({ color: 0xe6efe9 }));
  t.position.set(x, y, z); scene.add(t); tubes.push(t);
  const l = new THREE.PointLight(0xdfe9e4, 0.45, 9, 1.6); l.position.set(x, y - 0.2, z); scene.add(l);
}
export function keyLight(x, y, z, tx, angle, dist) {
  const k = new THREE.SpotLight(0xe8f0ea, 1.3, dist, angle, 0.6, 1.2);
  k.position.set(x, y, z); k.target.position.set(tx, 0, 0);
  k.castShadow = true; k.shadow.mapSize.set(1024, 1024); k.shadow.bias = -0.0005;
  scene.add(k, k.target);
}

export function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  view.baseFov = camera.fov = w / h < 0.9 ? 62 : 42;
  camera.updateProjectionMatrix();
}
export const up = (o, y) => o.position.clone().add(new THREE.Vector3(0, y, 0));
export const v = new THREE.Vector3();
export const toScreen = (p) => { v.copy(p).project(camera); return [(v.x * 0.5 + 0.5) * innerWidth, (-v.y * 0.5 + 0.5) * innerHeight]; };
