// Everything on screen around the 3D view: stopwatch, choices, endings, reveal panel, labels. Also the list of floors
// and moving between them, because a floor's reveal buttons need that.
import { $, reduced, scene, showFloor, stage, toScreen, up, view } from "./world.js";
import { sleep } from "./motion.js";
import { parkCar, ride } from "./lift.js";

export const floors = {};
export const endings = {};
export function addFloor(F) { floors[F.number] = F; endings[F.number] = new Set(); }

export function setFloorHud(n) {
  const F = floors[n];
  $("brandFloor").textContent = `Floor ${String(n).padStart(2, "0")} / 48 · ${F.name}`;
  $("keyF").textContent = F.keys[0]; $("keyT").textContent = F.keys[1];
  $("watchHead").textContent = F.watch || "Time to stamp one word";
  $("revealMap").innerHTML = F.map.map(([o, s]) => `<div><span class="o">${o}</span><span>${s}</span></div>`).join("");
  $("revealLesson").textContent = F.lesson;
}

// ---------- Stopwatch ----------
const pct = (s) => (s / floors[view.floor].scale * 100).toFixed(2) + "%";

export function setClock(f, t) { $("time").textContent = (f + t).toFixed(1) + " s"; $("liveF").style.width = pct(f); $("liveT").style.width = pct(t); }
const ran = new Map();
// A label already on the list keeps its row, which takes the new numbers.
export function addHistory(run) {
  let row = ran.get(run.label);
  if (!row) {
    row = document.createElement("div");
    row.className = "row";
    ran.set(run.label, row);
    $("history").appendChild(row);
  }
  row.innerHTML = `<span>${run.label}</span><div class="bar"><div class="f" style="width:${pct(run.fetch)}"></div><div class="t" style="width:${pct(run.think)}"></div></div><b>${(run.fetch + run.think).toFixed(1)} s</b>`;
}

// ---------- UI pieces ----------
export function choose(opts) {
  const bx = $("choices");
  bx.innerHTML = "";
  opts.forEach(([text, fn], i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "opt" + (i === 0 && opts.length === 1 ? " primary" : "");
    b.innerHTML = `${text} <kbd>${i + 1}</kbd>`;
    b.addEventListener("click", () => { bx.innerHTML = ""; fn(); });
    bx.appendChild(b);
  });
}

function showEndings() { $("endings").innerHTML = `Endings <em>${endings[view.floor].size}</em>/2`; }
export function findEnding(id, name) {
  if (endings[view.floor].has(id)) return;
  endings[view.floor].add(id);

  showEndings();
  $("toastName").textContent = name;
  $("toast").classList.add("on");
  setTimeout(() => $("toast").classList.remove("on"), 3200);
}

export function labels(on) {
  for (const F of Object.values(floors)) {
    const here = on && F.number === view.floor;
    for (const id of F.labels) $(id).classList.toggle("on", here);
    if (F.onLabels) F.onLabels(here);
  }
  stage.classList.toggle("frozen", on);
}

export function revealButtons(list) {
  const acts = $("revealActions");
  acts.innerHTML = "";
  list.forEach(([text, fn, primary]) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "opt" + (primary ? " primary" : "");
    b.textContent = text; b.addEventListener("click", fn); acts.appendChild(b);
  });
}
export function clearStage() { labels(false); $("reveal").hidden = true; $("choices").innerHTML = ""; }

// Labels pinned to the scene: "thinking…", and the reveal's unless the floor places its own (see floor 3).
const thinkLabel = [$("aThink"), () => up(view.thinker.group, 2.2)];
export function placeLabels(F) {
  for (const [el, at] of [thinkLabel, ...(F.anchors || [])]) {
    if (!el.classList.contains("on")) continue;
    const [x, y] = toScreen(at());
    el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
  }
}

// Moves to a floor by lift, resets it, and starts its narration. Replaying a floor, or with reduced motion, it fades.
export async function goFloor(n) {
  clearStage();
  const from = floors[view.floor], F = floors[n];
  if (from.leave) from.leave();
  const enter = () => {
    view.floor = n;
    showFloor(n);
    scene.fog.near = F.camera.fog[0]; scene.fog.far = F.camera.fog[1];
    setFloorHud(n);
    $("history").innerHTML = ""; ran.clear();
    setClock(0, 0); showEndings();
    F.enter();
  };
  if (from === F || reduced) {
    $("fade").classList.add("on");
    await sleep(0.6);
    enter(); parkCar(F);
    $("fade").classList.remove("on");
    await sleep(0.4);
  } else await ride(from, F, enter);
  F.start();
}
