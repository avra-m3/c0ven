// The building's sounds, made in the browser: the lift's bell, doors, motor and music, and the PA, fax and stamp. They
// follow the page's sound button and start only after a click.
import { $ } from "./world.js";

const LEVEL = 0.7;
let ctx = null, out = null, room = null, noise = null, hum = null, music = null;
const on = () => $("sound").getAttribute("aria-pressed") === "true";
const gainOf = (v) => { const g = ctx.createGain(); g.gain.value = v; return g; };
const filter = (type, freq, q = 0.7) => { const f = ctx.createBiquadFilter(); f.type = type; f.frequency.value = freq; f.Q.value = q; return f; };
const hz = (m) => 440 * 2 ** ((m - 69) / 12);   // from a MIDI note

export function unlockSound() {
  if (ctx) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  ctx = new AC();
  out = gainOf(on() ? LEVEL : 0); out.connect(ctx.destination);
  noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
  const d = noise.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  // A small hard room: the echo of a click in it is a burst of noise that dies fast.
  const tail = ctx.createBuffer(2, Math.floor(ctx.sampleRate * 0.5), ctx.sampleRate), echo = ctx.createConvolver();
  for (let c = 0; c < 2; c++) {
    const e = tail.getChannelData(c);
    for (let i = 0; i < e.length; i++) e[i] = (Math.random() * 2 - 1) * Math.exp(-7 * i / e.length);
  }
  echo.buffer = tail; room = gainOf(0.22); room.connect(echo); echo.connect(out);
  ctx.resume();
}
$("sound").addEventListener("click", () => { if (out) out.gain.value = on() ? LEVEL : 0; });

const play = (node) => { node.connect(out); node.connect(room); };

// A short burst of noise in one band: a click or a rattle.
function burst(t, freq, level, life, to) {
  const s = ctx.createBufferSource(), f = filter("bandpass", freq, 1.4), g = gainOf(0);
  g.gain.setValueAtTime(level, t); g.gain.exponentialRampToValueAtTime(0.0005, t + life);
  s.buffer = noise; s.connect(f); f.connect(g);
  if (to) g.connect(to); else play(g);
  s.start(t, Math.random() * 1.8, life + 0.01);
}
// A short falling tone: a latch or a door stop.
function thump(t, freq, level) {
  const o = ctx.createOscillator(), g = gainOf(0);
  o.type = "triangle"; o.frequency.setValueAtTime(freq * 1.7, t); o.frequency.exponentialRampToValueAtTime(freq, t + 0.05);
  g.gain.setValueAtTime(level, t); g.gain.exponentialRampToValueAtTime(0.0005, t + 0.17);
  o.connect(g); play(g); o.start(t); o.stop(t + 0.18);
}

// One strike of the chime: a metal bar's partials, the upper ones dying first, with the tick of its hammer.
function strike(t) {
  for (const [mul, level, life] of [[1, 0.26, 2.4], [1.004, 0.1, 1.9], [2.756, 0.07, 0.8], [5.404, 0.03, 0.3]]) {
    const o = ctx.createOscillator(), g = gainOf(0);
    o.frequency.value = hz(84) * mul;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(level, t + 0.004); g.gain.exponentialRampToValueAtTime(0.0005, t + life);
    o.connect(g); play(g); o.start(t); o.stop(t + life);
  }
  burst(t, 3200, 0.05, 0.02);
}
// One note going up and two going down, as lifts do. The note is in the music's key.
export function ding(down) {
  if (!ctx || !on()) return;
  strike(ctx.currentTime);
  if (down) strike(ctx.currentTime + 0.48);
}

// A door's run: the latch lets go, the motor and rollers follow the leaf's speed, and it stops with a knock, harder as
// it shuts.
export function doorSound(seconds, opening) {
  if (!ctx || !on()) return;
  const t = ctx.currentTime, end = t + seconds, pace = (low, top) => Float32Array.from({ length: 24 }, (_, i) => low + top * Math.sin(Math.PI * i / 23));
  const rollers = ctx.createBufferSource(), band = filter("bandpass", 420, 0.8), dull = filter("lowpass", 1100), flutter = gainOf(0.7), wheel = ctx.createOscillator(), depth = gainOf(0.3), run = gainOf(0);
  rollers.buffer = noise; rollers.loop = true;
  wheel.frequency.value = 26; wheel.connect(depth); depth.connect(flutter.gain);
  run.gain.setValueCurveAtTime(pace(0, 2.2), t, seconds);
  rollers.connect(band); band.connect(dull); dull.connect(flutter); flutter.connect(run); play(run);
  const rattle = ctx.createBufferSource(), high = filter("highpass", 2600), loose = gainOf(0);
  rattle.buffer = noise; rattle.loop = true;
  loose.gain.setValueCurveAtTime(pace(0, 0.012), t, seconds);
  rattle.connect(high); high.connect(loose); play(loose);
  const motor = ctx.createOscillator(), soft = filter("lowpass", 900), whirr = gainOf(0);
  motor.type = "sawtooth"; motor.frequency.setValueCurveAtTime(pace(75, 95), t, seconds);
  whirr.gain.setValueCurveAtTime(pace(0, 0.09), t, seconds);
  motor.connect(soft); soft.connect(whirr); play(whirr);
  for (const s of [rollers, rattle]) { s.start(t, Math.random() * 1.8); s.stop(end); }
  for (const o of [wheel, motor]) { o.start(t); o.stop(end); }
  thump(t, 130, 0.14); burst(t, 1800, 0.12, 0.03);
  thump(end, opening ? 95 : 80, opening ? 0.16 : 0.34); burst(end, 950, opening ? 0.08 : 0.2, 0.1);
}

// The PA reading out a name: a click, then a buzz of speech through a horn, a syllable at a time. hoarse (0 to 1)
// lowers it and takes its body away.
export function paCall(seconds, hoarse = 0) {
  if (!ctx || !on()) return;
  const t = ctx.currentTime, o = ctx.createOscillator(), horn = filter("bandpass", 1150 - 250 * hoarse, 1.5), g = gainOf(0);
  o.type = "sawtooth";
  const syllables = 3, beat = seconds * 0.85 / syllables;
  o.frequency.setValueAtTime(148 - 42 * hoarse, t);
  for (let k = 0; k < syllables; k++) {
    const a = t + k * beat;
    o.frequency.linearRampToValueAtTime((158 - 46 * hoarse) * (k % 2 ? 0.92 : 1.06), a + beat * 0.5);
    g.gain.setValueAtTime(0, a); g.gain.linearRampToValueAtTime(0.11 * (1 - 0.5 * hoarse), a + 0.02); g.gain.setValueAtTime(0.11 * (1 - 0.5 * hoarse), a + beat * 0.6); g.gain.linearRampToValueAtTime(0, a + beat * 0.92);
  }
  o.connect(horn); horn.connect(g); play(g);
  o.start(t); o.stop(t + seconds);
  burst(t, 2400, 0.04, 0.02);
}
// The fax machine's two-tone ring.
export function faxRing() {
  if (!ctx || !on()) return;
  const t = ctx.currentTime;
  for (let k = 0; k < 4; k++) {
    const o = ctx.createOscillator(), g = gainOf(0), a = t + k * 0.11;
    o.type = "square"; o.frequency.value = k % 2 ? 1180 : 1480;
    g.gain.setValueAtTime(0, a); g.gain.linearRampToValueAtTime(0.022, a + 0.01); g.gain.setValueAtTime(0.022, a + 0.09); g.gain.linearRampToValueAtTime(0, a + 0.105);
    o.connect(g); play(g); o.start(a); o.stop(a + 0.11);
  }
}
// A rubber stamp coming down on a desk.
export function stampSound() {
  if (!ctx || !on()) return;
  thump(ctx.currentTime, 110, 0.3); burst(ctx.currentTime, 700, 0.1, 0.05);
}
// A sheet of paper going across a desk.
export function flick() {
  if (!ctx || !on()) return;
  burst(ctx.currentTime, 3200, 0.018, 0.05);
}

// The motor: a low hum that swells with the car's speed. level is 0 to 1.
export function setHum(level) {
  if (!ctx) return;
  if (!hum) {
    hum = { g: gainOf(0), f: filter("lowpass", 260), o: [ctx.createOscillator(), ctx.createOscillator()] };
    hum.o[0].type = "sawtooth"; hum.o[1].type = "sine"; hum.o[0].frequency.value = 58; hum.o[1].frequency.value = 116;
    for (const o of hum.o) { o.connect(hum.f); o.start(); }
    hum.f.connect(hum.g); hum.g.connect(out);
  }
  hum.g.gain.setTargetAtTime(level * 0.06, ctx.currentTime, 0.06);
  hum.o[0].frequency.setTargetAtTime(52 + level * 12, ctx.currentTime, 0.1);
}

// ---------- The music ----------
// A bossa nova in F, sixteen bars of eight steps, which carries on from where the last ride left it.
const STEP = 60 / 112 / 2;   // seconds
const CHORD = {   // bass root, bass fifth, then the keys' notes, all MIDI
  F: [41, 48, [57, 60, 64, 67]], F7: [41, 48, [57, 63, 67, 72]], G: [43, 50, [58, 62, 65, 69]], C: [48, 43, [58, 62, 64, 69]],
  A: [45, 52, [55, 60, 64, 67]], D: [50, 45, [54, 60, 64, 69]], Bb: [46, 53, [57, 62, 65, 67]], Eb: [39, 46, [55, 61, 65, 70]],
};
const FORM = "F F F F G G C C A A D D G G C C F F F7 F7 Bb Bb Eb Eb A A D D G C F F".split(" ");   // two chords a bar
const CLAVE = [0, 3, 6, 10, 13];   // steps of a two-bar cycle
const WALK = [[0, 0, 3], [3, 1, 1], [4, 1, 3], [7, 0, 1]];   // step, root or fifth, length
const TUNE = [   // for each bar: step, MIDI note, length
  [[0, 88, 3], [3, 86, 1], [4, 84, 4]],
  [[2, 81, 2], [4, 84, 1], [5, 81, 3]],
  [[0, 86, 3], [3, 84, 1], [4, 82, 4]],
  [[2, 79, 2], [4, 82, 1], [5, 79, 3]],
  [[0, 88, 3], [3, 84, 1], [4, 81, 4]],
  [[2, 78, 2], [4, 81, 1], [5, 84, 3]],
  [[0, 82, 3], [3, 81, 1], [4, 79, 4]],
  [[2, 82, 1], [3, 81, 1], [4, 79, 2], [6, 76, 2]],
  [[0, 81, 3], [3, 84, 1], [4, 88, 4]],
  [[2, 87, 2], [4, 86, 1], [5, 84, 3]],
  [[0, 86, 3], [3, 84, 1], [4, 89, 4]],
  [[2, 87, 2], [4, 85, 1], [5, 82, 3]],
  [[0, 84, 3], [3, 81, 1], [4, 88, 4]],
  [[2, 86, 2], [4, 84, 1], [5, 81, 3]],
  [[0, 82, 2], [2, 86, 2], [4, 84, 2], [6, 82, 2]],
  [[0, 81, 6]],
];

// An electric piano note: a sine bent by another sine, bright at the strike and round after it.
function keys(t, m, level, length) {
  const o = ctx.createOscillator(), mod = ctx.createOscillator(), bend = gainOf(0), g = gainOf(0), f = hz(m);
  o.frequency.value = f; mod.frequency.value = f;
  bend.gain.setValueAtTime(f * 1.1, t); bend.gain.exponentialRampToValueAtTime(f * 0.08, t + 0.3);
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(level, t + 0.006); g.gain.exponentialRampToValueAtTime(0.0005, t + length + 0.25);
  mod.connect(bend); bend.connect(o.frequency); o.connect(g); g.connect(music.band);
  for (const n of [o, mod]) { n.start(t); n.stop(t + length + 0.26); }
}
// A vibraphone note: the bar's note and its fourth partial, which dies first. The tune's bus adds the tremolo.
function vibes(t, m, length) {
  for (const [mul, level, life] of [[1, 0.16, length + 0.9], [4, 0.025, 0.35]]) {
    const o = ctx.createOscillator(), g = gainOf(0);
    o.frequency.value = hz(m) * mul;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(level, t + 0.005); g.gain.exponentialRampToValueAtTime(0.0005, t + life);
    o.connect(g); g.connect(music.tune); o.start(t); o.stop(t + life);
  }
}
function bass(t, m, length) {
  const o = ctx.createOscillator(), g = gainOf(0);
  o.type = "triangle"; o.frequency.value = hz(m);
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.3, t + 0.012); g.gain.exponentialRampToValueAtTime(0.02, t + length); g.gain.linearRampToValueAtTime(0, t + length + 0.04);
  o.connect(g); g.connect(music.band); o.start(t); o.stop(t + length + 0.05);
}
function queueNotes() {
  while (music.next < ctx.currentTime + 0.5) {
    const t = music.next, s = music.step, at = s % 8, chord = CHORD[FORM[Math.floor(s / 4)]], feel = () => t + Math.random() * 0.012;
    if (CLAVE.includes(s % 16)) {
      for (const m of chord[2]) keys(feel(), m, 0.035, STEP * 1.5);
      burst(t, 1700, 0.05, 0.03, music.band);
    }
    burst(t, 3400, at % 2 ? 0.012 : 0.022, 0.05, music.band);
    for (const [a, which, n] of WALK) if (a === at) bass(t, chord[which], n * STEP);
    for (const [a, m, n] of TUNE[Math.floor(s / 8)]) if (a === at) vibes(feel(), m, n * STEP);
    music.step = (s + 1) % 128; music.next = t + STEP;
  }
}
function buildMusic() {
  // The car's small ceiling speaker on worn tape: no deep bass or top end, and a slow drift in pitch.
  music = { step: 0, next: 0, timer: 0, near: 0, band: gainOf(1), tune: gainOf(0.8), muffle: filter("lowpass", 400), level: gainOf(0) };
  const lfo = ctx.createOscillator(), shake = gainOf(0.2), drift = ctx.createOscillator(), wow = gainOf(0.001), tape = ctx.createDelay(0.05);
  lfo.frequency.value = 5.2; lfo.connect(shake); shake.connect(music.tune.gain); lfo.start();
  drift.frequency.value = 0.6; drift.connect(wow); tape.delayTime.value = 0.01; wow.connect(tape.delayTime); drift.start();
  const low = filter("highpass", 130), body = filter("peaking", 1400);
  body.gain.value = 3;
  music.tune.connect(music.band); music.band.connect(tape); tape.connect(low); low.connect(body); body.connect(music.muffle); music.muffle.connect(music.level);
  play(music.level);
}
// How near the car's music is: 0 is silent and stops it, 1 is inside the car; in between it is quieter and muffled.
export function setMusic(near) {
  if (!ctx) return;
  if (!music) buildMusic();
  if (near && Math.abs(near - music.near) < 0.02) return;
  music.near = near;
  const now = ctx.currentTime;
  music.level.gain.setTargetAtTime(near * 0.75, now, 0.09);
  music.muffle.frequency.setTargetAtTime(400 * 12 ** near, now, 0.09);
  if (near && !music.timer) { music.next = now + 0.04; queueNotes(); music.timer = setInterval(queueNotes, 150); }
  if (!near && music.timer) { clearInterval(music.timer); music.timer = 0; }
}
