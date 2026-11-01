// Time-based helpers. tween, sleep: with reduced motion they finish at once, so anything on a stopwatch uses its own timer.
import { reduced } from "./world.js";

export function tween(seconds, fn) {
  return new Promise((done) => {
    if (reduced || seconds <= 0) { fn(1); done(); return; }
    const t0 = performance.now();
    const tick = (now) => { const u = Math.min(1, (now - t0) / (seconds * 1000)); fn(u); if (u < 1) requestAnimationFrame(tick); else done(); };
    requestAnimationFrame(tick);
  });
}
export const ease = (u) => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
export const lerp = (a, b, u) => a + (b - a) * u;
export const sleep = (s) => new Promise((r) => setTimeout(r, reduced ? 0 : s * 1000));

export const backOut = (u) => 1 + 2.70158 * Math.pow(u - 1, 3) + 1.70158 * Math.pow(u - 1, 2);
