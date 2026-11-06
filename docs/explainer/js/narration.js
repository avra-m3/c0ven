// Narration: one audio file per stretch between choices, captions cued inside it. narration/floorN.json holds a
// floor's lines as [what the voice reads, the caption], which lines each chunk (audio file) holds, and each chunk's
// length with the time each caption word is spoken, in seconds.
import { $ } from "./world.js";

const LINES = {}, CHUNKS = {};
const audio = new Audio();
audio.preload = "auto";
let soundOn = true, skipLine = null;
const clips = {};
// Reads the floors' narration. Every audio file is fetched up front as a blob, so playback starts with no load gap. A
// chunk not yet recorded gets evenly spread word times, and the page runs it on its clock.
export async function loadNarration(numbers) {
  const all = await Promise.all(numbers.map((n) => fetch(`narration/floor${n}.json`).then((r) => r.json())));
  for (const { lines, chunks, timing } of all) {
    for (const [id, [, text]] of Object.entries(lines)) LINES[id] = text;
    for (const [c, ids] of Object.entries(chunks)) {
      if (timing[c]) { CHUNKS[c] = timing[c]; continue; }
      let t = 0;
      const placed = ids.map((i) => {
        const words = lines[i][1].split(" ").length, at = [i, Array.from({ length: words }, (_, k) => +(t + k * 0.36).toFixed(2))];
        t += words * 0.36 + 0.5;
        return at;
      });
      CHUNKS[c] = { dur: +t.toFixed(2), lines: placed, silent: true };
    }
    for (const c of Object.keys(timing)) {
      fetch(`audio/${c}.mp3`).then((r) => r.ok ? r.blob() : null).then((b) => { if (b) clips[c] = URL.createObjectURL(b); }).catch(() => {});
    }
  }
}
export const speaking = () => !!skipLine;
export const skip = () => { if (skipLine) skipLine(); };
// The first click on the page unlocks sound for the one audio element.
export function unlockAudio() { audio.play().catch(() => {}); audio.pause(); }

function caption(lineId) {
  const cap = $("caption");
  const words = LINES[lineId].split(" ").map((w) => `<span class="w">${w}</span>`).join(" ");
  cap.innerHTML = `<span class="who">[PA SYSTEM]</span>${words}<span class="skip">Click or press Space to skip</span>`;
  cap.hidden = false;
  return cap.querySelectorAll(".w");
}

// Plays a chunk; onCue(lineId) runs as each caption starts. Without sound, or if the file fails, a clock stands in.
export function speak(id, onCue) {
  const { dur, lines, silent } = CHUNKS[id];
  const cues = lines.map(([line, ts], i) => [i ? ts[0] : 0, line, ts]);
  return new Promise((resolve) => {
    let done = false, shown = -1, spans = [], said = 0, useAudio = soundOn && !silent, base = 0, t0 = performance.now();
    let lastT = -1, lastMove = performance.now(), endTimer = null;
    const time = () => useAudio ? audio.currentTime : base + (performance.now() - t0) / 1000;
    const toClock = () => { if (!useAudio) return; base = audio.currentTime || 0; t0 = performance.now(); useAudio = false; audio.pause(); };
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(endTimer);
      audio.onended = audio.onerror = null;
      audio.pause();
      skipLine = null;
      resolve();
    };
    skipLine = () => {
      const next = cues[shown + 1];
      if (!next) { finish(); return; }
      if (useAudio) audio.currentTime = next[0];
      else { base = next[0]; t0 = performance.now(); }
    };
    const tick = () => {
      if (done) return;
      const t = time();
      let i = 0;
      while (i + 1 < cues.length && cues[i + 1][0] <= t) i++;
      if (i !== shown) { shown = i; spans = caption(cues[i][1]); said = 0; if (onCue) onCue(cues[i][1]); }
      const ts = cues[i][2];
      while (said < spans.length && ts[said] <= t + 0.05) spans[said++].classList.add("said");
      if (useAudio) {
        if (t !== lastT) { lastT = t; lastMove = performance.now(); }
        else if (performance.now() - lastMove > 4000) toClock();  // a stalled file never holds the floor
      } else if (t >= dur + 0.55) { finish(); return; }
      requestAnimationFrame(tick);
    };
    if (useAudio) {
      audio.src = clips[id] || `audio/${id}.mp3`;
      audio.onended = () => { endTimer = setTimeout(finish, 550); };
      audio.onerror = toClock;
      const p = audio.play();
      if (p && p.catch) p.catch(toClock);
    }
    tick();
  });
}
$("caption").addEventListener("click", () => skipLine && skipLine());
$("sound").addEventListener("click", () => {
  soundOn = !soundOn;
  $("sound").textContent = soundOn ? "Sound on" : "Sound off";
  $("sound").setAttribute("aria-pressed", soundOn);
  if (!soundOn) audio.pause();
});
