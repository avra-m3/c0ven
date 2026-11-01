# Writes each caption word's start time into narration/floorN.json, for the page to highlight words as they are spoken.
# Usage, from docs/explainer: python3 tools/align.py <floor> <words.json>
# words.json comes from any speech recogniser run over audio/: {chunk: [{"w": word, "start": s, "end": s}, ...]}.
import difflib, json, re, subprocess, sys

floor, words = sys.argv[1:3]
path = f"narration/floor{floor}.json"
d = json.load(open(path))
W = json.load(open(words))
norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower().replace("advisor", "adviser"))
for c, ids in d["chunks"].items():
    if c not in W: continue
    heard = W[c]
    cap = [w for i in ids for w in d["lines"][i][1].split()]
    sm = difflib.SequenceMatcher(None, [norm(w) for w in cap], [norm(x["w"]) for x in heard], autojunk=False)
    t = [None] * len(cap)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size): t[blk.a + k] = heard[blk.b + k]["start"]
    # A caption word the recogniser spelled differently is spread between its matched neighbours.
    k = 0
    while k < len(t):
        if t[k] is not None: k += 1; continue
        e = k
        while e < len(t) and t[e] is None: e += 1
        lo = t[k - 1] if k else 0.0
        hi = t[e] if e < len(t) else heard[-1]["end"]
        for j in range(k, e): t[j] = lo + (hi - lo) * (j - k + 1) / (e - k + 1)
        k = e
    lines, pos = [], 0
    for i in ids:
        n = len(d["lines"][i][1].split())
        lines.append([i, [round(x, 2) for x in t[pos:pos + n]]]); pos += n
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f"audio/{c}.mp3"], capture_output=True, text=True).stdout)
    d["timing"][c] = {"dur": round(dur, 2), "lines": lines}
    print(c, sum(b.size for b in sm.get_matching_blocks()), "/", len(cap), "words matched")
json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
open(path, "a").write("\n")
